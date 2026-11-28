"""Typed local-model tagging services."""

from .cache import (
    CachedTagRecommendations,
    DatabaseTagRecommendationCache,
    InMemoryTagRecommendationCache,
    StoredTagRecommendations,
)
from .candidate_service import (
    DEFAULT_CANDIDATE_MODEL,
    CandidateService,
    recommend_tags,
    recommend_tags_async,
)
from .decision_service import (
    DEFAULT_DECISION_MODEL,
    TagDecisionResult,
    decide_tags,
    decide_tags_async,
)

__all__ = (
    "CachedTagRecommendations",
    "CandidateService",
    "DEFAULT_CANDIDATE_MODEL",
    "DEFAULT_DECISION_MODEL",
    "DatabaseTagRecommendationCache",
    "InMemoryTagRecommendationCache",
    "StoredTagRecommendations",
    "TagDecisionResult",
    "decide_tags",
    "decide_tags_async",
    "recommend_tags",
    "recommend_tags_async",
)
