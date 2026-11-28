from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from ..domain.contracts import HardwareReport
from ..domain.errors import ModelRecommendationError


GIB = 1024**3
LOW_DISK_RESERVE = 4 * GIB


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model: str
    ram_gib: float
    disk_gib: float
    vram_gib: float
    quality_weight: int


EMBEDDING_MODELS = (
    ModelSpec("embeddinggemma-2:270m", 1.0, 0.5, 0.75, 100),
    ModelSpec("embeddinggemma-2:440m", 2.0, 1.0, 1.5, 200),
    ModelSpec("embeddinggemma-2:740m", 3.5, 2.0, 2.5, 300),
)

DECISION_MODELS = (
    ModelSpec("nimble:9b", 12.0, 10.0, 10.0, 300),
    ModelSpec("tev1:4b", 6.0, 4.5, 4.5, 200),
    ModelSpec("tev1:0.8b", 2.0, 1.5, 1.5, 100),
)

GENERATION_MODELS = (
    ModelSpec("inkling-2-mini", 26.0, 20.0, 20.0, 600),
    ModelSpec("gemma4:31b", 42.0, 34.0, 34.0, 500),
    ModelSpec("gemma4:26b", 34.0, 27.0, 27.0, 400),
    ModelSpec("gemma4:12b", 18.0, 14.0, 14.0, 300),
    ModelSpec("gemma4:e4b", 8.5, 7.0, 7.0, 200),
    ModelSpec("gemma4:e2b", 3.5, 3.0, 3.0, 100),
)


class RecommendedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str
    model: str
    ram_gib: float
    disk_gib: float
    execution_plan: str
    quality_weight: int
    is_override: bool = False
    warnings: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()


class ModelRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    embedding: RecommendedModel
    decision: RecommendedModel
    generation: RecommendedModel


def _ram_gib(hardware: HardwareReport) -> float:
    return hardware.available_ram_bytes / GIB


def _disk_gib(hardware: HardwareReport) -> float:
    return hardware.available_disk_bytes / GIB


def _reserve_ram(hardware: HardwareReport) -> float:
    return min(4.0, max(0.25, _ram_gib(hardware) * 0.1))


def _is_feasible(spec: ModelSpec, hardware: HardwareReport) -> bool:
    return (
        spec.ram_gib + _reserve_ram(hardware) <= _ram_gib(hardware)
        and spec.disk_gib + LOW_DISK_RESERVE / GIB <= _disk_gib(hardware)
    )


def _execution_plan(spec: ModelSpec, hardware: HardwareReport) -> str:
    if hardware.vram_bytes is not None and hardware.vram_bytes / GIB >= spec.vram_gib:
        return "gpu_resident"
    return "cpu_or_offload"


def _recommend_spec(
    *,
    role: str,
    candidates: tuple[ModelSpec, ...],
    hardware: HardwareReport,
    override: str | None = None,
    excluded_models: frozenset[str] = frozenset(),
    preferred_model: str | None = None,
) -> RecommendedModel:
    by_name = {spec.model: spec for spec in candidates}
    if override is not None:
        normalized = override.lower()
        if any(marker in normalized for marker in (":cloud", "-cloud", "cloud:")):
            raise ModelRecommendationError(
                "Cloud-hosted generation variants are not supported."
            )
        spec = by_name.get(override)
        if spec is None:
            raise ModelRecommendationError(f"Unknown local {role} model: {override}.")
        override_warnings: tuple[str, ...] = ()
        if not _is_feasible(spec, hardware):
            override_warnings = (
                f"User override requires {spec.ram_gib:g} GiB available RAM and "
                f"{spec.disk_gib:g} GiB model disk space, but detected hardware "
                "is below the measured recommendation gate.",
            )
        return RecommendedModel(
            role=role,
            model=spec.model,
            ram_gib=spec.ram_gib,
            disk_gib=spec.disk_gib,
            execution_plan=_execution_plan(spec, hardware),
            quality_weight=spec.quality_weight,
            is_override=True,
            warnings=override_warnings,
            reasons=("User-selected model override.",),
        )

    ordered = sorted(
        (spec for spec in candidates if spec.model not in excluded_models),
        key=lambda spec: spec.quality_weight,
        reverse=True,
    )
    if preferred_model is not None:
        ordered.sort(key=lambda spec: spec.model != preferred_model)
    for spec in ordered:
        if not _is_feasible(spec, hardware):
            continue
        reasons = [
            f"Fits {_ram_gib(hardware):g} GiB available RAM with system headroom.",
            f"Fits {_disk_gib(hardware):g} GiB available disk with reserve.",
        ]
        if role == "generation" and spec.model == "inkling-2-mini":
            reasons.append("Inkling receives the highest generation quality weight.")
        recommendation_warnings: tuple[str, ...] = ()
        if hardware.vram_bytes is None:
            recommendation_warnings = (
                "No discrete GPU VRAM was detected; expect CPU/offload execution.",
            )
        return RecommendedModel(
            role=role,
            model=spec.model,
            ram_gib=spec.ram_gib,
            disk_gib=spec.disk_gib,
            execution_plan=_execution_plan(spec, hardware),
            quality_weight=spec.quality_weight,
            warnings=recommendation_warnings,
            reasons=tuple(reasons),
        )
    raise ModelRecommendationError(
        f"No feasible local {role} model fits RAM and disk headroom."
    )


def recommend_models(
    hardware: HardwareReport,
    *,
    use_visual_augmentation: bool = False,
    full_multimodal: bool = False,
    generation_override: str | None = None,
    decision_override: str | None = None,
    embedding_override: str | None = None,
    excluded_generation_models: frozenset[str] | set[str] = frozenset(),
) -> ModelRecommendation:
    if hardware.available_disk_bytes < LOW_DISK_RESERVE:
        raise ModelRecommendationError(
            "Available disk space is below the required 4 GiB operating reserve."
        )

    embedding_preference = "embeddinggemma-2:270m"
    if use_visual_augmentation:
        embedding_preference = "embeddinggemma-2:440m"
    if full_multimodal:
        embedding_preference = "embeddinggemma-2:740m"
        use_visual_augmentation = True

    embedding = _recommend_spec(
        role="embedding",
        candidates=EMBEDDING_MODELS,
        hardware=hardware,
        override=embedding_override,
        preferred_model=embedding_preference,
    )
    decision = _recommend_spec(
        role="decision",
        candidates=DECISION_MODELS,
        hardware=hardware,
        override=decision_override,
    )
    generation = _recommend_spec(
        role="generation",
        candidates=GENERATION_MODELS,
        hardware=hardware,
        override=generation_override,
        excluded_models=frozenset(excluded_generation_models),
    )
    return ModelRecommendation(
        embedding=embedding,
        decision=decision,
        generation=generation,
    )
