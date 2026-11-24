from .chat import ChatWithResultsDialog, markdown_to_html
from .hybrid import HybridSearch, SearchFilters, hybrid_search, normalize_fts_query
from .results import Citation, MatchedSection, SearchResult, make_result, merge_results
from .rrf import FusedHit, reciprocal_rank_fusion

__all__ = [
    "ChatWithResultsDialog",
    "Citation",
    "FusedHit",
    "HybridSearch",
    "MatchedSection",
    "SearchFilters",
    "SearchResult",
    "hybrid_search",
    "make_result",
    "markdown_to_html",
    "merge_results",
    "normalize_fts_query",
    "reciprocal_rank_fusion",
]
