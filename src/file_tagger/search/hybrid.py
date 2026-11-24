from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import struct
from typing import Literal

from sqlalchemy import text
from sqlalchemy.engine import RowMapping
from sqlalchemy.orm import Session

from ..domain.contracts import SearchCandidate
from ..infrastructure.database import normalize_unicode_key
from .rrf import FusedHit, reciprocal_rank_fusion

VECTOR_DIMENSIONS = frozenset({256, 512, 768})
DEFAULT_CANDIDATE_LIMIT = 50
DEFAULT_RRF_K = 60
MAX_QUERY_TERMS = 64


@dataclass(frozen=True)
class SearchFilters:
    tags: tuple[str, ...] = ()
    path: str | None = None
    file_types: tuple[str, ...] = ()


def normalize_fts_query(query: str) -> str:
    normalized = normalize_unicode_key(query)
    assert normalized is not None
    tokens: list[str] = []
    current: list[str] = []
    for character in normalized:
        if character.isalnum():
            current.append(character)
        elif current:
            tokens.append("".join(current))
            current = []
            if len(tokens) >= MAX_QUERY_TERMS:
                break
    if current and len(tokens) < MAX_QUERY_TERMS:
        tokens.append("".join(current))

    unique = tuple(dict.fromkeys(tokens))
    return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in unique)


class HybridSearch:
    def __init__(
        self,
        session: Session,
        *,
        candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        if candidate_limit <= 0:
            raise ValueError("The candidate limit must be positive.")
        if rrf_k <= 0:
            raise ValueError("The reciprocal-rank constant must be positive.")
        self._session = session
        self._candidate_limit = candidate_limit
        self._rrf_k = rrf_k

    def search(
        self,
        query: str,
        query_embedding: Sequence[float] | None = None,
        *,
        tags: Sequence[str] = (),
        path: str | None = None,
        file_type: str | None = None,
        file_types: Sequence[str] = (),
        filters: SearchFilters | None = None,
        limit: int = 10,
    ) -> tuple[SearchCandidate, ...]:
        if limit < 0:
            raise ValueError("The result limit cannot be negative.")
        if limit == 0:
            return ()

        selected_filters = _merge_filters(
            filters,
            tags=tags,
            path=path,
            file_type=file_type,
            file_types=file_types,
        )
        lexical_query = normalize_fts_query(query)
        if not lexical_query:
            return ()
        rankings: dict[str, tuple[str, ...]] = {
            "lexical": self._lexical_rankings(
                lexical_query,
                selected_filters,
            )
        }

        embedding = tuple(query_embedding or ())
        if embedding:
            rankings["vector"] = self._vector_rankings(
                embedding,
                selected_filters,
            )

        fused = reciprocal_rank_fusion(rankings, k=self._rrf_k)
        if not fused:
            return ()
        hydrated = self._hydrate_candidates(fused, selected_filters)
        return self._group_by_file(fused, hydrated, limit)

    def _lexical_rankings(
        self,
        lexical_query: str,
        filters: SearchFilters,
    ) -> tuple[str, ...]:
        conditions, parameters = _filter_sql(filters)
        parameters.update(
            {
                "lexical_query": lexical_query,
                "candidate_limit": self._candidate_limit,
            }
        )
        rows = self._session.execute(
            text(
                f"""
                SELECT c.stable_id
                FROM chunks_fts
                JOIN chunks AS c ON c.id = chunks_fts.rowid
                JOIN pages AS p ON p.id = c.page_id
                JOIN documents AS d ON d.id = c.document_id
                JOIN files AS f ON f.id = d.file_id
                WHERE chunks_fts MATCH :lexical_query
                  AND {conditions}
                ORDER BY bm25(chunks_fts), c.id
                LIMIT :candidate_limit
                """
            ),
            parameters,
        ).mappings()
        return tuple(str(row["stable_id"]) for row in rows)

    def _vector_rankings(
        self,
        query_embedding: tuple[float, ...],
        filters: SearchFilters,
    ) -> tuple[str, ...]:
        dimensions = len(query_embedding)
        if dimensions not in VECTOR_DIMENSIONS:
            supported = ", ".join(str(item) for item in sorted(VECTOR_DIMENSIONS))
            raise ValueError(
                f"Unsupported query embedding dimension {dimensions}; "
                f"expected one of {supported}."
            )
        conditions, parameters = _filter_sql(filters)
        parameters.update(
            {
                "query_embedding": struct.pack(
                    f"<{dimensions}f",
                    *query_embedding,
                ),
                "candidate_limit": self._candidate_limit,
            }
        )
        rows = self._session.execute(
            text(
                f"""
                WITH eligible AS (
                    SELECT c.id, c.stable_id
                    FROM chunks AS c
                    JOIN pages AS p ON p.id = c.page_id
                    JOIN documents AS d ON d.id = c.document_id
                    JOIN files AS f ON f.id = d.file_id
                    WHERE {conditions}
                ),
                nearest AS (
                    SELECT chunk_id, distance
                    FROM chunk_embeddings_{dimensions}
                    WHERE embedding MATCH :query_embedding
                      AND k = :candidate_limit
                      AND chunk_id IN (SELECT id FROM eligible)
                )
                SELECT e.stable_id
                FROM nearest AS n
                JOIN eligible AS e ON e.id = n.chunk_id
                ORDER BY n.distance, e.id
                LIMIT :candidate_limit
                """
            ),
            parameters,
        ).mappings()
        return tuple(str(row["stable_id"]) for row in rows)

    def _hydrate_candidates(
        self,
        fused: tuple[FusedHit, ...],
        filters: SearchFilters,
    ) -> Mapping[str, RowMapping]:
        conditions, parameters = _filter_sql(filters)
        placeholders = []
        for index, hit in enumerate(fused):
            name = f"chunk_{index}"
            placeholders.append(f":{name}")
            parameters[name] = hit.chunk_id
        rows = self._session.execute(
            text(
                f"""
                SELECT
                    c.stable_id AS chunk_id,
                    f.canonical_path AS file_path,
                    p.page_number,
                    c.heading,
                    c.text
                FROM chunks AS c
                JOIN pages AS p ON p.id = c.page_id
                JOIN documents AS d ON d.id = c.document_id
                JOIN files AS f ON f.id = d.file_id
                WHERE c.stable_id IN ({", ".join(placeholders)})
                  AND {conditions}
                """
            ),
            parameters,
        ).mappings()
        return {str(row["chunk_id"]): row for row in rows}

    @staticmethod
    def _group_by_file(
        fused: tuple[FusedHit, ...],
        hydrated: Mapping[str, RowMapping],
        limit: int,
    ) -> tuple[SearchCandidate, ...]:
        candidates: list[SearchCandidate] = []
        seen_files: set[str] = set()
        for hit in fused:
            row = hydrated.get(hit.chunk_id)
            if row is None:
                continue
            file_path = str(row["file_path"])
            if file_path in seen_files:
                continue
            source: Literal["fts", "vector", "hybrid"] = (
                "hybrid"
                if len(hit.source_ranks) > 1
                else "fts"
                if "lexical" in hit.source_ranks
                else "vector"
            )
            candidates.append(
                SearchCandidate(
                    chunk_id=hit.chunk_id,
                    file_path=file_path,
                    page_number=int(row["page_number"]),
                    heading=(
                        str(row["heading"]) if row["heading"] is not None else None
                    ),
                    text=str(row["text"]),
                    source=source,
                    rank=len(candidates) + 1,
                    score=hit.score,
                )
            )
            seen_files.add(file_path)
            if len(candidates) >= limit:
                break
        return tuple(candidates)


def hybrid_search(
    session: Session,
    query: str,
    query_embedding: Sequence[float] | None = None,
    *,
    tags: Sequence[str] = (),
    path: str | None = None,
    file_type: str | None = None,
    file_types: Sequence[str] = (),
    filters: SearchFilters | None = None,
    limit: int = 10,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    rrf_k: int = DEFAULT_RRF_K,
) -> tuple[SearchCandidate, ...]:
    return HybridSearch(
        session,
        candidate_limit=candidate_limit,
        rrf_k=rrf_k,
    ).search(
        query,
        query_embedding,
        tags=tags,
        path=path,
        file_type=file_type,
        file_types=file_types,
        filters=filters,
        limit=limit,
    )


def _merge_filters(
    filters: SearchFilters | None,
    *,
    tags: Sequence[str],
    path: str | None,
    file_type: str | None,
    file_types: Sequence[str],
) -> SearchFilters:
    selected_tags = tuple(filters.tags if filters is not None else ())
    selected_types = tuple(filters.file_types if filters is not None else ())
    selected_path = filters.path if filters is not None else None
    merged_types = (*selected_types, *(file_types or ()))
    if file_type:
        merged_types = (*merged_types, file_type)
    merged_tags = (*selected_tags, *(tags or ()))
    return SearchFilters(
        tags=tuple(dict.fromkeys(tag.strip() for tag in merged_tags if tag.strip())),
        path=path if path is not None else selected_path,
        file_types=tuple(
            dict.fromkeys(item.strip() for item in merged_types if item.strip())
        ),
    )


def _filter_sql(
    filters: SearchFilters,
) -> tuple[str, dict[str, object]]:
    clauses = [
        "d.revision = (SELECT max(latest.revision) FROM documents AS latest "
        "WHERE latest.file_id = f.id)"
    ]
    parameters: dict[str, object] = {}
    for index, tag in enumerate(filters.tags):
        name = f"tag_{index}"
        clauses.append(
            "EXISTS ("
            "SELECT 1 FROM file_tags AS ft "
            "JOIN tags AS t ON t.id = ft.tag_id "
            "WHERE ft.file_id = f.id "
            "AND unicode_normalize(t.name) = unicode_normalize(:" + name + ")"
            ")"
        )
        parameters[name] = tag

    if filters.path is not None and filters.path.strip():
        normalized_path = normalize_unicode_key(filters.path.strip())
        assert normalized_path is not None
        clauses.append(
            "unicode_normalize(f.canonical_path) "
            "LIKE unicode_normalize(:path_filter) ESCAPE '\\'"
        )
        parameters["path_filter"] = f"%{_escape_like(normalized_path)}%"

    type_clauses: list[str] = []
    for index, file_type in enumerate(filters.file_types):
        normalized = normalize_unicode_key(file_type.strip())
        assert normalized is not None
        name = f"file_type_{index}"
        if "/" in normalized:
            type_clauses.append(
                f"unicode_normalize(f.mime_type) = unicode_normalize(:{name})"
            )
            parameters[name] = normalized
        else:
            extension = normalized.removeprefix(".")
            type_clauses.append(
                f"unicode_normalize(f.canonical_path) "
                f"LIKE unicode_normalize(:{name}) ESCAPE '\\'"
            )
            parameters[name] = f"%.{_escape_like(extension)}"
    if type_clauses:
        clauses.append(f"({' OR '.join(type_clauses)})")
    return " AND ".join(clauses), parameters


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
