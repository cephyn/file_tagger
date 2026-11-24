import pytest

from file_tagger.domain.contracts import HardwareReport
from file_tagger.domain.errors import ModelRecommendationError
from file_tagger.ollama.model_recommender import recommend_models


GIB = 1024**3


def hardware(
    *,
    available_ram_gb: float,
    free_disk_gb: float = 100,
    vram_gb: float | None = None,
) -> HardwareReport:
    return HardwareReport(
        cpu_architecture="AMD64",
        logical_processors=8,
        total_ram_bytes=int(max(available_ram_gb, 16) * GIB),
        available_ram_bytes=int(available_ram_gb * GIB),
        available_disk_bytes=int(free_disk_gb * GIB),
        gpu_name="NVIDIA GPU" if vram_gb is not None else None,
        vram_bytes=int(vram_gb * GIB) if vram_gb is not None else None,
    )


@pytest.mark.parametrize(
    ("available_ram_gb", "expected_generation"),
    [(8, "gemma4:e2b"), (24, "gemma4:12b"), (32, "inkling-2-mini")],
)
def test_generation_model_follows_feasible_size(
    available_ram_gb: int, expected_generation: str
) -> None:
    recommendation = recommend_models(hardware(available_ram_gb=available_ram_gb))

    assert recommendation.generation.model == expected_generation


def test_low_ram_prefers_tev1_fallback_and_small_embedding() -> None:
    recommendation = recommend_models(hardware(available_ram_gb=4))

    assert recommendation.decision.model == "tev1:0.8b"
    assert recommendation.embedding.model == "embeddinggemma-2:270m"
    assert recommendation.generation.model == "gemma4:e2b"
    assert recommendation.generation.warnings


@pytest.mark.parametrize(
    ("available_ram_gb", "excluded", "expected_generation"),
    [
        (10, frozenset(), "gemma4:e4b"),
        (40, frozenset({"inkling-2-mini"}), "gemma4:26b"),
        (
            64,
            frozenset({"inkling-2-mini", "gemma4:26b"}),
            "gemma4:31b",
        ),
    ],
)
def test_generation_ladder_supports_every_feasible_tier(
    available_ram_gb: int,
    excluded: frozenset[str],
    expected_generation: str,
) -> None:
    recommendation = recommend_models(
        hardware(available_ram_gb=available_ram_gb),
        excluded_generation_models=excluded,
    )

    assert recommendation.generation.model == expected_generation


def test_gpu_memory_changes_execution_note_but_not_feasibility() -> None:
    cpu_only = recommend_models(hardware(available_ram_gb=24))
    gpu_backed = recommend_models(
        hardware(available_ram_gb=24, vram_gb=24)
    )

    assert cpu_only.generation.model == gpu_backed.generation.model
    assert cpu_only.generation.execution_plan == "cpu_or_offload"
    assert gpu_backed.generation.execution_plan == "gpu_resident"


def test_visual_profiles_select_only_feasible_larger_embedding() -> None:
    constrained = recommend_models(
        hardware(available_ram_gb=8), use_visual_augmentation=True
    )
    multimodal = recommend_models(
        hardware(available_ram_gb=12),
        use_visual_augmentation=True,
        full_multimodal=True,
    )

    assert constrained.embedding.model == "embeddinggemma-2:440m"
    assert multimodal.embedding.model == "embeddinggemma-2:740m"


def test_low_disk_rejects_recommendation() -> None:
    with pytest.raises(ModelRecommendationError, match="disk"):
        recommend_models(hardware(available_ram_gb=8, free_disk_gb=2))


def test_user_override_is_allowed_with_explanation() -> None:
    recommendation = recommend_models(
        hardware(available_ram_gb=8),
        generation_override="gemma4:12b",
    )

    assert recommendation.generation.model == "gemma4:12b"
    assert recommendation.generation.is_override is True
    assert recommendation.generation.warnings


def test_incompatible_cloud_override_is_rejected() -> None:
    with pytest.raises(ModelRecommendationError, match="Cloud"):
        recommend_models(
            hardware(available_ram_gb=32),
            generation_override="gemma4:cloud",
        )
