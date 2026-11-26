from time import perf_counter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..domain.contracts import ModelProfile
from ..domain.errors import FileTaggerError


class BenchmarkResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str
    model: str
    elapsed_ms: float = Field(ge=0)
    success: bool
    slow: bool = False
    detail: str | None = None


async def run_model_benchmark(
    gateway: Any,
    profile: ModelProfile,
    *,
    slow_threshold_ms: float = 10_000,
) -> BenchmarkResult:
    started = perf_counter()
    try:
        if profile.role == "embedding":
            await gateway.embed(
                profile.model,
                "File Tagger local readiness probe.",
                dimensions=profile.dimensions,
            )
        elif profile.role == "decision":
            await gateway.system_one(
                profile.model,
                state={"title": "Readiness probe"},
                questions={
                    "ready": {
                        "type": "choice",
                        "instructions": "Decide whether local Ollama inference is ready.",
                        "criteria": {
                            "yes": "Local Ollama returned an available inference result.",
                            "no": "Local Ollama did not return an available inference result.",
                        },
                    }
                },
            )
        else:
            await gateway.chat(
                profile.model,
                messages=[
                    {
                        "role": "user",
                        "content": "Reply with a short JSON readiness result.",
                    }
                ],
                schema={
                    "type": "object",
                    "properties": {"ready": {"type": "boolean"}},
                    "required": ["ready"],
                },
            )
    except (ConnectionError, TimeoutError, FileTaggerError) as exc:
        elapsed_ms = (perf_counter() - started) * 1_000
        return BenchmarkResult(
            role=profile.role,
            model=profile.model,
            elapsed_ms=elapsed_ms,
            success=False,
            detail=f"{exc.__class__.__name__}: readiness probe failed",
        )

    elapsed_ms = (perf_counter() - started) * 1_000
    return BenchmarkResult(
        role=profile.role,
        model=profile.model,
        elapsed_ms=elapsed_ms,
        success=True,
        slow=elapsed_ms >= slow_threshold_ms,
    )
