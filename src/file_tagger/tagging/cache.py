from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import AliasChoices, Field, model_validator
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from ..domain.contracts import DomainRecord, TagDecision
from ..persistence.models import File, TagSuggestionCache


def utc_now() -> datetime:
    return datetime.now(UTC)


class CachedTagRecommendations(DomainRecord):
    file_id: int | None = Field(default=None, gt=0)
    file_path: str | None = Field(default=None, min_length=1)
    file_hash: str = Field(min_length=1)
    document_state_digest: str | None = Field(default=None, min_length=1)
    extraction_version: str | None = Field(default=None, min_length=1)
    candidate_model_digest: str | None = Field(default=None, min_length=1)
    decision_model_digest: str | None = Field(default=None, min_length=1)
    candidate_set_digest: str | None = Field(default=None, min_length=1)
    max_candidates: int | None = Field(default=None, gt=0)
    confirmation_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    model_digest: str | None = Field(default=None, min_length=1)
    decisions: tuple[TagDecision, ...] = Field(
        default=(),
        validation_alias=AliasChoices("decisions", "recommendations", "suggestions"),
    )
    cached_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def require_model_identity(self) -> "CachedTagRecommendations":
        if not any(
            (
                self.candidate_model_digest,
                self.decision_model_digest,
                self.model_digest,
            )
        ):
            raise ValueError("At least one tag model digest is required.")
        return self

    @property
    def effective_candidate_model_digest(self) -> str | None:
        return self.candidate_model_digest or self.model_digest

    @property
    def effective_decision_model_digest(self) -> str | None:
        return self.decision_model_digest or self.model_digest

    @property
    def recommendations(self) -> tuple[TagDecision, ...]:
        return self.decisions

    @property
    def suggestions(self) -> tuple[TagDecision, ...]:
        return self.decisions

    @property
    def accepted(self) -> tuple[TagDecision, ...]:
        return tuple(item for item in self.decisions if item.decision == "accept")

    @property
    def rejected(self) -> tuple[TagDecision, ...]:
        return tuple(item for item in self.decisions if item.decision == "reject")

    @property
    def review(self) -> tuple[TagDecision, ...]:
        return tuple(item for item in self.decisions if item.decision == "review")

    @property
    def requires_confirmation(self) -> tuple[str, ...]:
        return tuple(item.tag for item in self.review)


StoredTagRecommendations = CachedTagRecommendations


class TagRecommendationCache(Protocol):
    def lookup(
        self,
        file_hash: str,
        *,
        file_id: int | None = None,
        file_path: str | None = None,
        document_state_digest: str | None = None,
        extraction_version: str | None = None,
        candidate_model_digest: str | None = None,
        decision_model_digest: str | None = None,
        candidate_set_digest: str | None = None,
        max_candidates: int | None = None,
        confirmation_threshold: float | None = None,
        model_digest: str | None = None,
    ) -> CachedTagRecommendations | None: ...

    def store(self, stored: CachedTagRecommendations) -> None: ...


class InMemoryTagRecommendationCache:
    def __init__(self) -> None:
        self._entries: list[CachedTagRecommendations] = []

    def lookup(
        self,
        file_hash: str,
        *,
        file_id: int | None = None,
        file_path: str | None = None,
        document_state_digest: str | None = None,
        extraction_version: str | None = None,
        candidate_model_digest: str | None = None,
        decision_model_digest: str | None = None,
        candidate_set_digest: str | None = None,
        max_candidates: int | None = None,
        confirmation_threshold: float | None = None,
        model_digest: str | None = None,
    ) -> CachedTagRecommendations | None:
        if not file_hash:
            return None
        for stored in reversed(self._entries):
            if _matches(
                stored,
                file_hash=file_hash,
                file_id=file_id,
                file_path=file_path,
                document_state_digest=document_state_digest,
                extraction_version=extraction_version,
                candidate_model_digest=candidate_model_digest,
                decision_model_digest=decision_model_digest,
                candidate_set_digest=candidate_set_digest,
                max_candidates=max_candidates,
                confirmation_threshold=confirmation_threshold,
                model_digest=model_digest,
            ):
                return stored
        return None

    def store(self, stored: CachedTagRecommendations) -> None:
        self._entries = [
            entry for entry in self._entries if not _same_identity(entry, stored)
        ]
        self._entries.append(stored)

    def save(self, stored: CachedTagRecommendations) -> None:
        self.store(stored)


class DatabaseTagRecommendationCache:
    def __init__(self, session: Session) -> None:
        self._session = session

    def lookup(
        self,
        file_hash: str,
        *,
        file_id: int | None = None,
        file_path: str | None = None,
        document_state_digest: str | None = None,
        extraction_version: str | None = None,
        candidate_model_digest: str | None = None,
        decision_model_digest: str | None = None,
        candidate_set_digest: str | None = None,
        max_candidates: int | None = None,
        confirmation_threshold: float | None = None,
        model_digest: str | None = None,
    ) -> CachedTagRecommendations | None:
        if not file_hash:
            return None
        statement = select(TagSuggestionCache).where(
            TagSuggestionCache.file_hash == file_hash
        )
        if file_id is not None:
            statement = statement.where(TagSuggestionCache.file_id == file_id)
        if file_path is not None:
            statement = statement.where(TagSuggestionCache.file_path == file_path)
        if document_state_digest is not None:
            statement = statement.where(
                TagSuggestionCache.document_state_digest == document_state_digest
            )
        if extraction_version is not None:
            statement = statement.where(
                TagSuggestionCache.extraction_version == extraction_version
            )
        selected_candidate = candidate_model_digest or model_digest
        selected_decision = decision_model_digest or model_digest
        if selected_candidate is not None:
            statement = statement.where(
                TagSuggestionCache.candidate_model_digest == selected_candidate
            )
        if selected_decision is not None:
            statement = statement.where(
                TagSuggestionCache.decision_model_digest == selected_decision
            )
        if candidate_set_digest is not None:
            statement = statement.where(
                TagSuggestionCache.candidate_set_digest == candidate_set_digest
            )
        if max_candidates is not None:
            statement = statement.where(
                TagSuggestionCache.max_candidates == max_candidates
            )
        if confirmation_threshold is not None:
            statement = statement.where(
                TagSuggestionCache.confirmation_threshold
                == confirmation_threshold
            )
        row = self._session.scalars(
            statement.order_by(TagSuggestionCache.cached_at.desc()).limit(1)
        ).first()
        if row is None:
            return None
        payload = _database_payload(row)
        if payload is None:
            return None
        try:
            return CachedTagRecommendations.model_validate(payload)
        except (TypeError, ValueError):
            return None

    def store(self, stored: CachedTagRecommendations) -> None:
        file_id = self._resolve_file_id(stored)
        values = {
            "file_id": file_id,
            "file_path": stored.file_path or self._file_path(file_id),
            "file_hash": stored.file_hash,
            "document_state_digest": stored.document_state_digest,
            "extraction_version": stored.extraction_version,
            "candidate_model_digest": stored.effective_candidate_model_digest,
            "decision_model_digest": stored.effective_decision_model_digest,
            "candidate_set_digest": stored.candidate_set_digest,
            "max_candidates": stored.max_candidates,
            "confirmation_threshold": stored.confirmation_threshold,
            "cached_at": stored.cached_at,
            "suggestions": stored.model_dump(mode="json", exclude_none=True),
        }
        statement = sqlite_insert(TagSuggestionCache).values(**values)
        statement = statement.on_conflict_do_update(
            index_elements=[TagSuggestionCache.file_id],
            set_={
                "file_path": statement.excluded.file_path,
                "file_hash": statement.excluded.file_hash,
                "document_state_digest": (
                    statement.excluded.document_state_digest
                ),
                "extraction_version": statement.excluded.extraction_version,
                "candidate_model_digest": (
                    statement.excluded.candidate_model_digest
                ),
                "decision_model_digest": statement.excluded.decision_model_digest,
                "candidate_set_digest": statement.excluded.candidate_set_digest,
                "max_candidates": statement.excluded.max_candidates,
                "confirmation_threshold": (
                    statement.excluded.confirmation_threshold
                ),
                "cached_at": statement.excluded.cached_at,
                "suggestions": statement.excluded.suggestions,
            },
        )
        self._session.execute(statement)
        self._session.flush()

    def save(self, stored: CachedTagRecommendations) -> None:
        self.store(stored)

    def _resolve_file_id(self, stored: CachedTagRecommendations) -> int:
        if stored.file_id is not None:
            exact = self._session.get(File, stored.file_id)
            if exact is None:
                raise ValueError("The cache entry references an unknown file.")
            if exact.content_hash != stored.file_hash:
                raise ValueError("The cache file identity does not match its content hash.")
            if stored.file_path and exact.canonical_path != stored.file_path:
                raise ValueError("The cache file identity does not match its path.")
            return exact.id
        if stored.file_path:
            exact = self._session.scalar(
                select(File).where(File.canonical_path == stored.file_path)
            )
            if exact is not None and exact.content_hash == stored.file_hash:
                return exact.id
        matches = self._session.scalars(
            select(File).where(File.content_hash == stored.file_hash)
        ).all()
        if len(matches) == 1:
            return matches[0].id
        raise ValueError(
            "A tag recommendation cache entry requires a unique indexed file identity."
        )

    def _file_path(self, file_id: int) -> str:
        file = self._session.get(File, file_id)
        if file is None:
            raise ValueError("The cache entry references an unknown file.")
        return file.canonical_path


def _same_identity(
    left: CachedTagRecommendations,
    right: CachedTagRecommendations,
) -> bool:
    return all(
        (
            left.file_hash == right.file_hash,
            left.file_id == right.file_id,
            left.file_path == right.file_path,
            left.document_state_digest == right.document_state_digest,
            left.extraction_version == right.extraction_version,
            left.effective_candidate_model_digest
            == right.effective_candidate_model_digest,
            left.effective_decision_model_digest
            == right.effective_decision_model_digest,
            left.candidate_set_digest == right.candidate_set_digest,
            left.max_candidates == right.max_candidates,
            left.confirmation_threshold == right.confirmation_threshold,
        )
    )


def _matches(
    stored: CachedTagRecommendations,
    *,
    file_hash: str,
    file_id: int | None,
    file_path: str | None,
    document_state_digest: str | None,
    extraction_version: str | None,
    candidate_model_digest: str | None,
    decision_model_digest: str | None,
    candidate_set_digest: str | None,
    max_candidates: int | None,
    confirmation_threshold: float | None,
    model_digest: str | None,
) -> bool:
    if stored.file_hash != file_hash:
        return False
    if file_id is not None and stored.file_id != file_id:
        return False
    if file_path is not None and stored.file_path != file_path:
        return False
    if (
        document_state_digest is not None
        and stored.document_state_digest != document_state_digest
    ):
        return False
    if extraction_version is not None:
        if stored.extraction_version != extraction_version:
            return False
    selected_candidate = candidate_model_digest or model_digest
    selected_decision = decision_model_digest or model_digest
    if selected_candidate is not None:
        if stored.effective_candidate_model_digest != selected_candidate:
            return False
    if selected_decision is not None:
        if stored.effective_decision_model_digest != selected_decision:
            return False
    if candidate_set_digest is not None:
        if stored.candidate_set_digest != candidate_set_digest:
            return False
    if max_candidates is not None:
        if stored.max_candidates != max_candidates:
            return False
    if confirmation_threshold is not None:
        if stored.confirmation_threshold != confirmation_threshold:
            return False
    return True


def _database_payload(row: TagSuggestionCache) -> dict[str, Any] | None:
    if not isinstance(row.suggestions, Mapping):
        return None
    payload = dict(row.suggestions)
    payload.update(
        {
            "file_id": row.file_id,
            "file_path": row.file_path,
            "file_hash": row.file_hash,
            "document_state_digest": row.document_state_digest,
            "extraction_version": row.extraction_version,
            "candidate_model_digest": row.candidate_model_digest,
            "decision_model_digest": row.decision_model_digest,
            "candidate_set_digest": row.candidate_set_digest,
            "max_candidates": row.max_candidates,
            "confirmation_threshold": row.confirmation_threshold,
            "cached_at": row.cached_at,
        }
    )
    return payload


InMemoryCache = InMemoryTagRecommendationCache
DatabaseCache = DatabaseTagRecommendationCache
Cache = InMemoryTagRecommendationCache
