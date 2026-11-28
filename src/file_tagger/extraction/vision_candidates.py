from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


MINIMUM_USABLE_TEXT_CHARS = 100
DOMINANT_VISUAL_AREA = 0.5
VISUAL_EMBEDDING_PROFILES = frozenset(
    {
        "embeddinggemma-2:440m",
        "embeddinggemma-2:740m",
    }
)
VISUAL_CAPABILITIES = frozenset({"image", "multimodal", "vision"})


class VisionCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    page_number: int = Field(gt=0)
    text_chars: int = Field(ge=0)
    image_area: float = Field(ge=0, le=1)
    table_area: float = Field(ge=0, le=1)
    tables: int = Field(ge=0)
    needs_vision_embedding: bool
    reasons: tuple[str, ...] = ()


class VisualAugmentationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool
    reason: Literal[
        "visual augmentation enabled",
        "visual augmentation disabled",
    ]
    candidates: tuple[VisionCandidate, ...] = ()

    @property
    def selected_page_numbers(self) -> tuple[int, ...]:
        return tuple(candidate.page_number for candidate in self.candidates)

    @property
    def selected_candidates(self) -> tuple[VisionCandidate, ...]:
        return self.candidates


def classify_page(
    *,
    text_chars: int,
    image_area: float,
    tables: int,
    page_number: int = 1,
    table_area: float = 0,
    table_structure_valid: bool = True,
    low_quality_text: bool = False,
) -> VisionCandidate:
    reasons: list[str] = []
    if text_chars < MINIMUM_USABLE_TEXT_CHARS:
        reasons.append("insufficient extractable text")
    if low_quality_text:
        reasons.append("low-quality extracted text")
    if image_area >= DOMINANT_VISUAL_AREA:
        reasons.append("dominant visual content")
    if table_area >= DOMINANT_VISUAL_AREA and tables > 0:
        reasons.append("dominant table content")
    if tables > 0 and not table_structure_valid:
        reasons.append("table structure unavailable")

    return VisionCandidate(
        page_number=page_number,
        text_chars=text_chars,
        image_area=image_area,
        table_area=table_area,
        tables=tables,
        needs_vision_embedding=bool(reasons),
        reasons=tuple(reasons),
    )


def embedding_profile_name(embedding_profile: object) -> str:
    if isinstance(embedding_profile, str):
        return embedding_profile.strip().casefold()
    model = getattr(embedding_profile, "model", None)
    if isinstance(model, str):
        return model.strip().casefold()
    raise TypeError("embedding_profile must be a model name or expose a 'model' string")


def supports_visual_augmentation(embedding_profile: object) -> bool:
    if embedding_profile_name(embedding_profile) in VISUAL_EMBEDDING_PROFILES:
        return True
    capabilities = getattr(embedding_profile, "capabilities", ())
    if not isinstance(capabilities, (list, tuple, set, frozenset)):
        return False
    return any(
        isinstance(capability, str)
        and capability.strip().casefold() in VISUAL_CAPABILITIES
        for capability in capabilities
    )


def select_vision_candidates(
    candidates: Iterable[VisionCandidate],
    *,
    embedding_profile: object,
) -> VisualAugmentationResult:
    classified = tuple(candidates)
    if not supports_visual_augmentation(embedding_profile):
        return VisualAugmentationResult(
            enabled=False,
            reason="visual augmentation disabled",
        )
    return VisualAugmentationResult(
        enabled=True,
        reason="visual augmentation enabled",
        candidates=tuple(
            candidate
            for candidate in classified
            if candidate.needs_vision_embedding
        ),
    )
