"""Typed local-model tagging services."""

from .decision_service import (
    DEFAULT_DECISION_MODEL,
    TagDecisionResult,
    decide_tags,
    decide_tags_async,
)

__all__ = (
    "DEFAULT_DECISION_MODEL",
    "TagDecisionResult",
    "decide_tags",
    "decide_tags_async",
)
