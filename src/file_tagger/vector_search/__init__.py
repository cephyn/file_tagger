"""
Vector search package for file tagging application.
Provides semantic search capabilities.
"""

__all__ = ["VectorSearch"]


def __getattr__(name: str) -> object:
    if name == "VectorSearch":
        from .vector_search import VectorSearch

        return VectorSearch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")