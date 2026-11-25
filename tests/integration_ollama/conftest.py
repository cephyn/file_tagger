from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import csv
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import pytest

from file_tagger.adapters.ollama.gateway import (
    DEFAULT_OLLAMA_HOST,
    OllamaGateway,
)
from file_tagger.domain.contracts import ModelProfile
from file_tagger.domain.errors import RemoteOllamaHostError
from file_tagger.infrastructure.app_paths import AppPaths


OPT_IN_ENV = "FILE_TAGGER_RUN_OLLAMA_TESTS"
DEFAULT_EMBED_MODEL = "embeddinggemma-2:270m"
DEFAULT_DECISION_MODEL = "nimble:9b"
DEFAULT_GENERATION_MODEL = "gemma4:e4b"

pytestmark = pytest.mark.ollama


@dataclass(frozen=True, slots=True)
class IntegrationProfile:
    embedding: str
    decision: str
    generation: str
    embedding_profile: ModelProfile
    decision_profile: ModelProfile
    generation_profile: ModelProfile

    @property
    def digests(self) -> dict[str, str]:
        return {
            "embedding": self.embedding_profile.digest,
            "decision": self.decision_profile.digest,
            "generation": self.generation_profile.digest,
        }


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    vector: tuple[float, ...]
    latency_ms: float


@dataclass(frozen=True, slots=True)
class DecisionResult:
    answer: str
    latency_ms: float


@dataclass(frozen=True, slots=True)
class ChatResult:
    content: str
    latency_ms: float
    load_time_ms: float = 0.0
    tokens_per_second: float = 0.0


@dataclass
class RealOllama:
    gateway: OllamaGateway
    report_dir: Path
    metrics: dict[str, Any] = field(default_factory=dict)

    async def close(self) -> None:
        await self.gateway.close()

    def embed(
        self,
        model: str,
        text: str | Sequence[str],
        *,
        dimensions: int | None = None,
    ) -> EmbeddingResult:
        started = perf_counter()
        vectors = asyncio.run(self.gateway.embed(model, text, dimensions=dimensions))
        latency_ms = (perf_counter() - started) * 1000.0
        vector = vectors[0] if len(vectors) == 1 else vectors[0]
        self.metrics.setdefault("embedding_latency_ms", []).append(latency_ms)
        return EmbeddingResult(vector=vector, latency_ms=latency_ms)

    def decide(
        self,
        model: str,
        prompt: str,
        *,
        state: Mapping[str, Any] | None = None,
    ) -> DecisionResult:
        started = perf_counter()
        response = asyncio.run(
            self.gateway.system_one(
                model,
                state=state or {"prompt": prompt},
                questions={
                    "answer": {
                        "type": "choice",
                        "prompt": prompt,
                        "choices": ["accepted", "rejected", "review"],
                    }
                },
            )
        )
        latency_ms = (perf_counter() - started) * 1000.0
        answer = _decision_answer(response)
        self.metrics.setdefault("decision_latency_ms", []).append(latency_ms)
        return DecisionResult(answer=answer, latency_ms=latency_ms)

    def chat(
        self,
        model: str,
        prompt: str,
        *,
        messages: Sequence[Mapping[str, Any]] | None = None,
        schema: Mapping[str, Any] | None = None,
    ) -> ChatResult:
        selected_messages = (
            tuple(messages) if messages is not None else ({"role": "user", "content": prompt},)
        )
        started = perf_counter()
        content = asyncio.run(
            self.gateway.chat(
                model,
                messages=selected_messages,
                schema=schema,
            )
        )
        latency_ms = (perf_counter() - started) * 1000.0
        self.metrics.setdefault("generation_latency_ms", []).append(latency_ms)
        return ChatResult(content=content, latency_ms=latency_ms)

    def record_model_metrics(
        self,
        model: str,
        prompt: str = "Respond briefly: Ollama is ready.",
    ) -> ChatResult:
        started = perf_counter()
        response = asyncio.run(
            self.gateway._client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
        )
        wall_ms = (perf_counter() - started) * 1000.0
        content = getattr(getattr(response, "message", None), "content", "") or ""
        load_ms = _nanoseconds_to_ms(getattr(response, "load_duration", 0))
        eval_count = int(getattr(response, "eval_count", 0) or 0)
        eval_duration = int(getattr(response, "eval_duration", 0) or 0)
        tokens_per_second = (
            eval_count / (eval_duration / 1_000_000_000.0) if eval_duration > 0 else 0.0
        )
        metrics = {
            "wall_time_ms": wall_ms,
            "load_time_ms": load_ms,
            "tokens_per_second": tokens_per_second,
            "eval_count": eval_count,
        }
        self.metrics.setdefault("model_metrics", {})[model] = metrics
        return ChatResult(
            content=content,
            latency_ms=wall_ms,
            load_time_ms=load_ms,
            tokens_per_second=tokens_per_second,
        )

    def write_reports(
        self,
        profile: IntegrationProfile,
        *,
        recall: Mapping[str, Any] | None = None,
    ) -> tuple[Path, Path]:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "host": self.gateway.host,
            "models": {
                role: {
                    "model": getattr(profile, role),
                    "digest": profile.digests[role],
                    "context_length": getattr(profile, f"{role}_profile").context_length,
                    "dimensions": getattr(profile, f"{role}_profile").dimensions,
                    "capabilities": list(getattr(profile, f"{role}_profile").capabilities),
                }
                for role in ("embedding", "decision", "generation")
            },
            "metrics": self.metrics,
            "recall": recall or {},
        }
        json_path = self.report_dir / "ollama_quality.json"
        json_path.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        csv_path = self.report_dir / "ollama_quality.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["category", "model", "metric", "value"])
            for role in ("embedding", "decision", "generation"):
                model_profile = getattr(profile, f"{role}_profile")
                writer.writerow(["digest", model_profile.model, "sha256", model_profile.digest])
            for name, values in self.metrics.items():
                if isinstance(values, list):
                    for index, value in enumerate(values):
                        writer.writerow(["metric", name, str(index), value])
                elif isinstance(values, dict):
                    for model, model_values in values.items():
                        if isinstance(model_values, dict):
                            for metric, value in model_values.items():
                                writer.writerow(["metric", str(model), metric, value])
        return json_path, csv_path


def _nanoseconds_to_ms(value: Any) -> float:
    try:
        return float(value) / 1_000_000.0
    except (TypeError, ValueError):
        return 0.0


def _decision_answer(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, Mapping):
        for key in ("answer", "choice", "selected", "answers"):
            value = response.get(key)
            if isinstance(value, str) and value:
                return value
            if isinstance(value, Mapping):
                first = next(iter(value.values()), "")
                if isinstance(first, str):
                    return first
    for attr in ("answer", "answer_text", "choice", "selected"):
        value = getattr(response, attr, None)
        if isinstance(value, str) and value:
            return value
    return str(response)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "ollama: requires opt-in local Ollama integration tests",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    if os.environ.get(OPT_IN_ENV) == "1":
        return
    skip_marker = pytest.mark.skip(
        reason="set FILE_TAGGER_RUN_OLLAMA_TESTS=1 to run real local Ollama tests"
    )
    for item in items:
        if "ollama" in item.keywords:
            item.add_marker(skip_marker)


def _close_gateway(gateway: OllamaGateway) -> None:
    try:
        asyncio.run(gateway.close())
    except RuntimeError as exc:
        if "Event loop is closed" not in str(exc):
            raise


@pytest.fixture(scope="session")
def integration_profile() -> IntegrationProfile:
    if os.environ.get(OPT_IN_ENV) != "1":
        pytest.skip("set FILE_TAGGER_RUN_OLLAMA_TESTS=1 to run real local Ollama tests")

    host = os.environ.get("FILE_TAGGER_OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
    try:
        gateway = OllamaGateway(host=host)
    except RemoteOllamaHostError as exc:
        pytest.fail(f"Remote Ollama hosts require loopback: {exc}")

    try:
        required_models = {
            "embedding": os.environ.get(
                "FILE_TAGGER_OLLAMA_EMBED_MODEL",
                DEFAULT_EMBED_MODEL,
            ),
            "decision": os.environ.get(
                "FILE_TAGGER_OLLAMA_DECISION_MODEL",
                DEFAULT_DECISION_MODEL,
            ),
            "generation": os.environ.get(
                "FILE_TAGGER_OLLAMA_GENERATION_MODEL",
                DEFAULT_GENERATION_MODEL,
            ),
        }
        try:
            for model in required_models.values():
                gateway._validate_model(model)
        except Exception as exc:
            pytest.fail(f"Cloud or missing Ollama model configuration: {exc}")

        try:
            readiness = asyncio.run(gateway.readiness(required_models))
        except Exception as exc:
            pytest.fail(f"Could not connect to local Ollama: {exc}")

        if not readiness.ready:
            pytest.fail(
                "Local Ollama integration profile is not ready: "
                f"missing={readiness.missing_models}, "
                f"incompatible={readiness.incompatible_models}, "
                f"errors={readiness.errors}"
            )

        profile = IntegrationProfile(
            embedding=required_models["embedding"],
            decision=required_models["decision"],
            generation=required_models["generation"],
            embedding_profile=asyncio.run(
                gateway.model_profile(
                    "embedding",
                    required_models["embedding"],
                )
            ),
            decision_profile=asyncio.run(
                gateway.model_profile(
                    "decision",
                    required_models["decision"],
                )
            ),
            generation_profile=asyncio.run(
                gateway.model_profile(
                    "generation",
                    required_models["generation"],
                )
            ),
        )
        yield profile
    finally:
        _close_gateway(gateway)


@pytest.fixture(scope="session")
def real_ollama(integration_profile: IntegrationProfile):
    host = os.environ.get("FILE_TAGGER_OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
    gateway = OllamaGateway(host=host)
    report_dir = AppPaths.discover().data_dir / "reports"
    wrapper = RealOllama(
        gateway=gateway,
        report_dir=report_dir,
    )
    try:
        yield wrapper
    finally:
        recall = wrapper.metrics.get("recall_at_k")
        try:
            wrapper.write_reports(
                integration_profile,
                recall=recall if isinstance(recall, dict) else None,
            )
        finally:
            _close_gateway(wrapper.gateway)
