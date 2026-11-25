from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import sqlite3
import struct
from typing import Any, Literal

from sqlalchemy import text
from sqlalchemy.engine import RowMapping
from sqlalchemy.orm import Session

from ..domain.contracts import ModelProfile
from ..infrastructure.database import normalize_unicode_key
from .results import SearchResult, make_result, merge_results
from .rrf import FusedHit, reciprocal_rank_fusion

VECTOR_DIMENSIONS = frozenset({256, 512, 768})
DEFAULT_CANDIDATE_LIMIT = 50
DEFAULT_RRF_K = 60
MAX_QUERY_TERMS = 64


@dataclass(frozen=True)
class SearchFilters:
    tags: tuple[str, ...] = ()
    path: str | None = None
    paths: tuple[str, ...] = ()
    file_types: tuple[str, ...] = ()
    excluded_file_types: tuple[str, ...] = ()


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
        model_digest: str | None = None,
        index_version: str | None = None,
        embedding_profile: ModelProfile | None = None,
        profile: ModelProfile | None = None,
    ) -> None:
        if candidate_limit <= 0:
            raise ValueError("The candidate limit must be positive.")
        if rrf_k <= 0:
            raise ValueError("The reciprocal-rank constant must be positive.")
        self._session = session
        self._candidate_limit = candidate_limit
        self._rrf_k = rrf_k
        self._model_digest = model_digest
        self._index_version = index_version
        self._profile = embedding_profile or profile

    @property
    def embedding_profile(self) -> ModelProfile | None:
        return self._profile

    @property
    def model_digest(self) -> str | None:
        return self._model_digest

    @property
    def index_version(self) -> str | None:
        return self._index_version

    @property
    def index_fingerprint(self) -> str:
        digest = sha256()
        with _sqlite_read_snapshot(self._session):
            data_version = self._session.execute(
                text("PRAGMA data_version")
            ).scalar_one()
            _update_fingerprint(digest, "data_version", ((data_version,),))
            corpus_rows = self._session.execute(
                text(
                    """
                    SELECT
                        f.id AS file_id,
                        f.canonical_path,
                        f.content_hash AS file_content_hash,
                        f.size_bytes,
                        f.mtime_ns,
                        f.mime_type,
                        f.indexing_state,
                        d.id AS document_id,
                        d.revision,
                        d.file_hash AS document_hash,
                        d.extraction_version,
                        c.id AS chunk_id,
                        c.stable_id,
                        c.content_hash AS chunk_content_hash,
                        c.model_digest,
                        c.index_version
                    FROM files AS f
                    LEFT JOIN documents AS d
                      ON d.id = (
                          SELECT latest.id
                          FROM documents AS latest
                          WHERE latest.file_id = f.id
                          ORDER BY latest.revision DESC, latest.id DESC
                          LIMIT 1
                      )
                    LEFT JOIN chunks AS c ON c.document_id = d.id
                    ORDER BY f.id, c.ordinal, c.id
                    """
                )
            )
            _update_fingerprint(digest, "corpus", corpus_rows)
            tag_rows = self._session.execute(
                text(
                    """
                    SELECT ft.file_id, t.name
                    FROM file_tags AS ft
                    JOIN tags AS t ON t.id = ft.tag_id
                    ORDER BY ft.file_id, unicode_normalize(t.name), t.id
                    """
                )
            )
            _update_fingerprint(digest, "tags", tag_rows)
        return digest.hexdigest()

    def search(
        self,
        query: str,
        query_embedding: Sequence[float] | None = None,
        *,
        tags: Sequence[str] = (),
        path: str | None = None,
        file_type: str | None = None,
        file_types: Sequence[str] = (),
        excluded_file_types: Sequence[str] = (),
        filters: SearchFilters | None = None,
        limit: int = 10,
        model_digest: str | None = None,
        index_version: str | None = None,
        embedding_profile: ModelProfile | None = None,
        profile: ModelProfile | None = None,
    ) -> tuple[SearchResult, ...]:
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
            excluded_file_types=excluded_file_types,
        )
        lexical_query = normalize_fts_query(query)
        if not lexical_query:
            return ()
        with _sqlite_read_snapshot(self._session):
            rankings: dict[str, tuple[str, ...]] = {
                "lexical": self._lexical_rankings(
                    lexical_query,
                    selected_filters,
                )
            }

            embedding = tuple(query_embedding or ())
            if embedding:
                identity = self._vector_identity(
                    embedding,
                    model_digest=model_digest,
                    index_version=index_version,
                    embedding_profile=embedding_profile,
                    profile=profile,
                )
                if identity is not None:
                    rankings["vector"] = self._vector_rankings(
                        embedding,
                        selected_filters,
                        *identity,
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
        model_digest: str,
        index_version: str,
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
                "model_digest": model_digest,
                "index_version": index_version,
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
                      AND c.model_digest = :model_digest
                      AND c.index_version = :index_version
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

    def _vector_identity(
        self,
        query_embedding: tuple[float, ...],
        *,
        model_digest: str | None,
        index_version: str | None,
        embedding_profile: ModelProfile | None,
        profile: ModelProfile | None,
    ) -> tuple[str, str] | None:
        selected_profile = embedding_profile or profile or self._profile
        selected_digest = model_digest or self._model_digest
        selected_version = index_version or self._index_version
        if selected_profile is not None:
            if selected_profile.role != "embedding":
                return None
            if selected_profile.dimensions is None:
                return None
            if selected_digest is not None and selected_digest != selected_profile.digest:
                raise ValueError("The query embedding digest does not match its profile.")
            if len(query_embedding) != selected_profile.dimensions:
                raise ValueError(
                    "The query embedding dimension does not match its profile."
                )
            selected_digest = selected_profile.digest
            profile_version = selected_profile.default_index_version
            selected_version = selected_version or profile_version
        if not selected_digest or not selected_version:
            return None
        return selected_digest, selected_version

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
                    f.indexing_state,
                    p.page_number,
                    c.heading,
                    c.section,
                    c.text,
                    f.mime_type,
                    COALESCE(
                        (
                            SELECT json_group_array(ordered.name)
                            FROM (
                                SELECT t.name
                                FROM file_tags AS ft
                                JOIN tags AS t ON t.id = ft.tag_id
                                WHERE ft.file_id = f.id
                                ORDER BY unicode_normalize(t.name), t.id
                            ) AS ordered
                        ),
                        '[]'
                    ) AS tags
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
    ) -> tuple[SearchResult, ...]:
        candidates: list[SearchResult] = []
        for hit in fused:
            row = hydrated.get(hit.chunk_id)
            if row is None:
                continue
            source: Literal["fts", "vector", "hybrid"] = (
                "hybrid"
                if len(hit.source_ranks) > 1
                else "fts"
                if "lexical" in hit.source_ranks
                else "vector"
            )
            candidates.append(
                make_result(
                    row,
                    source=source,
                    rank=hit.rank,
                    score=hit.score,
                )
            )
        return merge_results(candidates, limit=limit)


def hybrid_search(
    session: Session,
    query: str,
    query_embedding: Sequence[float] | None = None,
    *,
    tags: Sequence[str] = (),
    path: str | None = None,
    file_type: str | None = None,
    file_types: Sequence[str] = (),
    excluded_file_types: Sequence[str] = (),
    filters: SearchFilters | None = None,
    limit: int = 10,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    rrf_k: int = DEFAULT_RRF_K,
    model_digest: str | None = None,
    index_version: str | None = None,
    embedding_profile: ModelProfile | None = None,
    profile: ModelProfile | None = None,
) -> tuple[SearchResult, ...]:
    return HybridSearch(
        session,
        candidate_limit=candidate_limit,
        rrf_k=rrf_k,
        model_digest=model_digest,
        index_version=index_version,
        embedding_profile=embedding_profile,
        profile=profile,
    ).search(
        query,
        query_embedding,
        tags=tags,
        path=path,
        file_type=file_type,
        file_types=file_types,
        excluded_file_types=excluded_file_types,
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
    excluded_file_types: Sequence[str],
) -> SearchFilters:
    selected_tags = tuple(filters.tags if filters is not None else ())
    selected_types = tuple(filters.file_types if filters is not None else ())
    selected_excluded_types = tuple(
        filters.excluded_file_types if filters is not None else ()
    )
    selected_paths = tuple(filters.paths if filters is not None else ())
    if filters is not None and filters.path is not None:
        selected_paths = (*selected_paths, filters.path)
    if path is not None:
        selected_paths = (*selected_paths, path)
    merged_paths = tuple(
        dict.fromkeys(item.strip() for item in selected_paths if item.strip())
    )
    merged_types = (*selected_types, *(file_types or ()))
    if file_type:
        merged_types = (*merged_types, file_type)
    merged_excluded_types = (
        *selected_excluded_types,
        *(excluded_file_types or ()),
    )
    merged_tags = (*selected_tags, *(tags or ()))
    return SearchFilters(
        tags=tuple(dict.fromkeys(tag.strip() for tag in merged_tags if tag.strip())),
        path=merged_paths[0] if len(merged_paths) == 1 else None,
        paths=merged_paths if len(merged_paths) > 1 else (),
        file_types=tuple(
            dict.fromkeys(item.strip() for item in merged_types if item.strip())
        ),
        excluded_file_types=tuple(
            dict.fromkeys(
                item.strip()
                for item in merged_excluded_types
                if item.strip()
            )
        ),
    )


@contextmanager
def _sqlite_read_snapshot(session: Session) -> Iterator[None]:
    owns_transaction = not session.in_transaction()
    if owns_transaction:
        with session.begin():
            _begin_sqlite_snapshot(session)
            yield
        return

    _begin_sqlite_snapshot(session)
    yield


def _begin_sqlite_snapshot(session: Session) -> None:
    connection = session.connection()
    dbapi_connection = connection.connection.driver_connection
    if not isinstance(dbapi_connection, sqlite3.Connection):
        raise TypeError("Hybrid search requires a SQLite database.")
    if not dbapi_connection.in_transaction:
        connection.exec_driver_sql("BEGIN")
    connection.exec_driver_sql("SELECT count(*) FROM files").scalar_one()


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

    selected_paths = tuple(filters.paths)
    if filters.path is not None:
        selected_paths = (*selected_paths, filters.path)
    path_clauses: list[str] = []
    for index, selected_path in enumerate(
        dict.fromkeys(item for item in selected_paths if item.strip())
    ):
        normalized_path = normalize_unicode_key(selected_path.strip())
        assert normalized_path is not None
        name = f"path_filter_{index}"
        path_clauses.append(
            "unicode_normalize(f.canonical_path) "
            f"LIKE unicode_normalize(:{name}) ESCAPE '\\'"
        )
        parameters[name] = f"%{_escape_like(normalized_path)}%"
    if path_clauses:
        clauses.append(f"({' OR '.join(path_clauses)})")

    type_clauses = _file_type_clauses(
        filters.file_types,
        name_prefix="file_type",
        parameters=parameters,
    )
    if type_clauses:
        clauses.append(f"({' OR '.join(type_clauses)})")
    excluded_type_clauses = _file_type_clauses(
        filters.excluded_file_types,
        name_prefix="excluded_file_type",
        parameters=parameters,
    )
    if excluded_type_clauses:
        clauses.append(f"NOT ({' OR '.join(excluded_type_clauses)})")
    return " AND ".join(clauses), parameters


def _file_type_clauses(
    file_types: Sequence[str],
    *,
    name_prefix: str,
    parameters: dict[str, object],
) -> list[str]:
    clauses: list[str] = []
    for index, file_type in enumerate(file_types):
        normalized = normalize_unicode_key(file_type.strip())
        assert normalized is not None
        name = f"{name_prefix}_{index}"
        if "/" in normalized:
            clauses.append(
                "(f.mime_type IS NOT NULL AND "
                f"unicode_normalize(f.mime_type) = unicode_normalize(:{name}))"
            )
            parameters[name] = normalized
        else:
            extension = normalized.removeprefix(".")
            clauses.append(
                "unicode_normalize(f.canonical_path) "
                f"LIKE unicode_normalize(:{name}) ESCAPE '\\'"
            )
            parameters[name] = f"%.{_escape_like(extension)}"
    return clauses


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _update_fingerprint(
    digest: Any,
    category: str,
    rows: Any,
) -> None:
    category_bytes = category.encode("utf-8")
    digest.update(len(category_bytes).to_bytes(8, "big"))
    digest.update(category_bytes)
    for row in rows:
        for value in row:
            encoded = b"" if value is None else str(value).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
