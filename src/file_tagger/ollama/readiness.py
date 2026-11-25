from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from .benchmarks import BenchmarkResult, run_model_benchmark
from ..domain.contracts import HardwareReport, ModelProfile, OllamaReadiness
from ..domain.errors import (
    ModelIncompatibleError,
    ModelMissingError,
    OllamaUnavailableError,
)
from ..persistence.models import ModelProfileRecord
from .model_recommender import ModelRecommendation


ROLE_FEATURES = {
    "embedding": ("indexing", "search"),
    "decision": ("tagging",),
    "generation": ("rag", "search"),
}


class RequiredModels(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    embedding: str = Field(min_length=1)
    decision: str = Field(min_length=1)
    generation: str = Field(min_length=1)

    def as_mapping(self) -> dict[str, str]:
        return {
            "embedding": self.embedding,
            "decision": self.decision,
            "generation": self.generation,
        }


class ReadinessReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ready: bool
    blocked_roles: tuple[str, ...] = ()
    instructions: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    profiles: tuple[ModelProfile, ...] = ()
    benchmarks: tuple[BenchmarkResult, ...] = ()
    ollama: OllamaReadiness | None = None
    hardware: HardwareReport | None = None
    recommendation: ModelRecommendation | None = None
    selected_models: RequiredModels | None = None


def _cloud_model(model: str) -> bool:
    normalized = model.lower()
    return any(
        marker in normalized
        for marker in (":cloud", "-cloud", "cloud:", ".cloud")
    )


def _blocked_features(roles: set[str]) -> tuple[str, ...]:
    features = {
        feature
        for role in roles
        for feature in ROLE_FEATURES.get(role, ())
    }
    return tuple(sorted(features))


class OllamaReadinessService:
    def __init__(
        self,
        gateway: Any,
        *,
        session: Session | None = None,
    ) -> None:
        self.gateway = gateway
        self.session = session

    async def check(
        self,
        required: RequiredModels,
        *,
        slow_threshold_ms: float = 10_000,
    ) -> ReadinessReport:
        model_map = required.as_mapping()
        gateway_report = await self.gateway.readiness(model_map)
        missing_models = set(gateway_report.missing_models)
        incompatible_models = set(gateway_report.incompatible_models)
        errors = list(gateway_report.errors)
        instructions: list[str] = []
        warnings: list[str] = []
        profiles: list[ModelProfile] = []
        benchmarks: list[BenchmarkResult] = []
        blocked_model_roles: set[str] = set()

        if not gateway_report.available or gateway_report.errors:
            blocked_model_roles.update(model_map)
            instructions.extend(
                (
                    "Install or start Ollama 0.35.0 or newer.",
                    "Keep Ollama bound to a local loopback address.",
                )
            )

        for role, model in model_map.items():
            if model in missing_models:
                blocked_model_roles.add(role)
                instructions.append(f"Install the model manually: ollama pull {model}")
                continue
            if model in incompatible_models:
                blocked_model_roles.add(role)
                instructions.append(
                    f"Select a model for {model} with the required {role} capability."
                )
                continue
            if _cloud_model(model):
                blocked_model_roles.add(role)
                errors.append(f"Cloud model variants are not supported: {model}.")
                continue

            try:
                profile = await self.gateway.model_profile(role, model)
            except ModelMissingError:
                blocked_model_roles.add(role)
                instructions.append(f"Install the model manually: ollama pull {model}")
                continue
            except ModelIncompatibleError:
                blocked_model_roles.add(role)
                errors.append(f"Model {model} lacks the required {role} capability.")
                instructions.append(
                    f"Select a different {role} model with the required capability."
                )
                continue
            except OllamaUnavailableError as exc:
                blocked_model_roles.add(role)
                errors.append(str(exc))
                continue

            profiles.append(profile)
            benchmark = await run_model_benchmark(
                self.gateway,
                profile,
                slow_threshold_ms=slow_threshold_ms,
            )
            benchmarks.append(benchmark)
            if not benchmark.success:
                blocked_model_roles.add(role)
                errors.append(
                    benchmark.detail
                    or f"Readiness benchmark failed for model {model}."
                )
            elif benchmark.slow:
                warnings.append(
                    f"{profile.role} benchmark for {profile.model} was slow "
                    f"({benchmark.elapsed_ms:.0f} ms)."
                )

        catalog_profiles = await self._compatible_installed_profiles()
        known_profiles = {
            (profile.role, profile.model, profile.digest)
            for profile in profiles
        }
        for profile in catalog_profiles:
            identity = (profile.role, profile.model, profile.digest)
            if identity not in known_profiles:
                profiles.append(profile)
                known_profiles.add(identity)

        if self.session is not None:
            self._persist_profiles(profiles, benchmarks)

        return ReadinessReport(
            ready=not blocked_model_roles and not errors,
            blocked_roles=_blocked_features(blocked_model_roles),
            instructions=tuple(dict.fromkeys(instructions)),
            warnings=tuple(warnings),
            errors=tuple(dict.fromkeys(errors)),
            profiles=tuple(profiles),
            benchmarks=tuple(benchmarks),
            ollama=gateway_report,
            selected_models=required,
        )

    async def _compatible_installed_profiles(self) -> list[ModelProfile]:
        list_models = getattr(self.gateway, "list_models", None)
        if not callable(list_models):
            return []
        try:
            installed_models = await list_models()
        except OllamaUnavailableError:
            return []

        profiles: list[ModelProfile] = []
        for model in installed_models:
            if _cloud_model(model):
                continue
            for role in ROLE_FEATURES:
                try:
                    profiles.append(await self.gateway.model_profile(role, model))
                except (ModelMissingError, ModelIncompatibleError):
                    continue
                except OllamaUnavailableError:
                    break
        return profiles

    def _persist_profiles(
        self,
        profiles: list[ModelProfile],
        benchmarks: list[BenchmarkResult],
    ) -> None:
        if self.session is None:
            return
        benchmark_by_model = {
            (result.role, result.model): result for result in benchmarks
        }
        for profile in profiles:
            record = self.session.scalar(
                select(ModelProfileRecord).where(
                    ModelProfileRecord.role == profile.role,
                    ModelProfileRecord.model == profile.model,
                    ModelProfileRecord.digest == profile.digest,
                    ModelProfileRecord.dimensions == profile.dimensions,
                )
            )
            benchmark = benchmark_by_model.get((profile.role, profile.model))
            index_version = (
                f"{profile.model}@{profile.digest}:{profile.dimensions}"
                if profile.role == "embedding"
                else f"{profile.model}@{profile.digest}"
            )
            benchmark_data: Mapping[str, Any] = (
                benchmark.model_dump(mode="json") if benchmark else {}
            )
            if record is None:
                record = ModelProfileRecord(
                    role=profile.role,
                    model=profile.model,
                    digest=profile.digest,
                    dimensions=profile.dimensions,
                    context_length=profile.context_length,
                    index_version=index_version,
                    capabilities=list(profile.capabilities),
                    benchmark_data=dict(benchmark_data),
                )
                self.session.add(record)
            else:
                record.context_length = profile.context_length
                record.index_version = index_version
                record.capabilities = list(profile.capabilities)
                record.benchmark_data = dict(benchmark_data)
        self.session.flush()
