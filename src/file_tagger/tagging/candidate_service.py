import asyncio
from collections.abc import Coroutine, Mapping, Sequence
import hashlib
import inspect
import json
from typing import Any, Protocol, TypeVar
import unicodedata

from pydantic import AliasChoices, ConfigDict, Field, TypeAdapter

from ..adapters.ollama.gateway import ModelRole, OllamaGateway
from ..domain.contracts import DomainRecord, ModelProfile, TagDecision
from .cache import CachedTagRecommendations, TagRecommendationCache
from .decision_service import (
    DEFAULT_DECISION_MODEL,
    MAX_SYSTEM_ONE_QUESTIONS,
    decide_tags_async,
)


decide_tags = decide_tags_async


DEFAULT_CANDIDATE_MODEL = "inkling-2-mini"
DEFAULT_MAX_TAG_CANDIDATES = 24
DEFAULT_SEMANTIC_RESULT_LIMIT = 12
DEFAULT_CONFIRMATION_THRESHOLD = 0.7

T = TypeVar("T")


class CandidateSearch(Protocol):
    def search(
        self,
        query: str,
        query_embedding: Sequence[float] | None = None,
        **kwargs: Any,
    ) -> Sequence[object]: ...


class CandidateGateway(Protocol):
    async def model_profile(self, role: ModelRole, model: str) -> ModelProfile: ...

    async def chat(
        self,
        model: str,
        *,
        messages: Sequence[Mapping[str, Any]],
        schema: Mapping[str, Any] | None = None,
        keep_alive: float | str | None = "5m",
    ) -> str: ...

    async def system_one(
        self,
        model: str,
        *,
        state: Any,
        questions: Mapping[str, Any],
        keep_alive: float | str | None = "5m",
    ) -> Any: ...


class CandidateEnvelope(DomainRecord):
    model_config = ConfigDict(extra="forbid", frozen=True)
    candidates: tuple[str, ...] = Field(
        max_length=MAX_SYSTEM_ONE_QUESTIONS,
        validation_alias=AliasChoices("candidates", "tags"),
    )


_CANDIDATE_ENVELOPE = TypeAdapter(CandidateEnvelope)


def recommend_tags(
    document_state: str,
    *,
    file_hash: str,
    extraction_version: str,
    search: CandidateSearch,
    candidate_model: str = DEFAULT_CANDIDATE_MODEL,
    decision_model: str = DEFAULT_DECISION_MODEL,
    candidate_model_digest: str | None = None,
    decision_model_digest: str | None = None,
    cache: TagRecommendationCache | None = None,
    gateway: CandidateGateway | None = None,
    decision_gateway: Any | None = None,
    max_candidates: int = DEFAULT_MAX_TAG_CANDIDATES,
    semantic_limit: int = DEFAULT_SEMANTIC_RESULT_LIMIT,
    confirmation_threshold: float = DEFAULT_CONFIRMATION_THRESHOLD,
    query_embedding: Sequence[float] | None = None,
    embedding_model_digest: str | None = None,
    embedding_index_version: str | None = None,
    embedding_profile: ModelProfile | None = None,
    file_id: int | None = None,
    file_path: str | None = None,
    force_refresh: bool = False,
) -> CachedTagRecommendations:
    return _run_sync(
        recommend_tags_async(
            document_state,
            file_hash=file_hash,
            extraction_version=extraction_version,
            search=search,
            candidate_model=candidate_model,
            decision_model=decision_model,
            candidate_model_digest=candidate_model_digest,
            decision_model_digest=decision_model_digest,
            cache=cache,
            gateway=gateway,
            decision_gateway=decision_gateway,
            max_candidates=max_candidates,
            semantic_limit=semantic_limit,
            confirmation_threshold=confirmation_threshold,
            query_embedding=query_embedding,
            embedding_model_digest=embedding_model_digest,
            embedding_index_version=embedding_index_version,
            embedding_profile=embedding_profile,
            file_id=file_id,
            file_path=file_path,
            force_refresh=force_refresh,
        )
    )


async def recommend_tags_async(
    document_state: str,
    *,
    file_hash: str,
    extraction_version: str,
    search: CandidateSearch,
    candidate_model: str = DEFAULT_CANDIDATE_MODEL,
    decision_model: str = DEFAULT_DECISION_MODEL,
    candidate_model_digest: str | None = None,
    decision_model_digest: str | None = None,
    cache: TagRecommendationCache | None = None,
    gateway: CandidateGateway | None = None,
    decision_gateway: Any | None = None,
    max_candidates: int = DEFAULT_MAX_TAG_CANDIDATES,
    semantic_limit: int = DEFAULT_SEMANTIC_RESULT_LIMIT,
    confirmation_threshold: float = DEFAULT_CONFIRMATION_THRESHOLD,
    query_embedding: Sequence[float] | None = None,
    embedding_model_digest: str | None = None,
    embedding_index_version: str | None = None,
    embedding_profile: ModelProfile | None = None,
    file_id: int | None = None,
    file_path: str | None = None,
    force_refresh: bool = False,
) -> CachedTagRecommendations:
    _validate_request(
        document_state=document_state,
        file_hash=file_hash,
        extraction_version=extraction_version,
        candidate_model=candidate_model,
        decision_model=decision_model,
        max_candidates=max_candidates,
        semantic_limit=semantic_limit,
        confirmation_threshold=confirmation_threshold,
    )
    owns_gateway = gateway is None
    selected_gateway: CandidateGateway = gateway or OllamaGateway()
    selected_decision_gateway = decision_gateway or selected_gateway
    document_state_digest = hashlib.sha256(document_state.encode("utf-8")).hexdigest()
    try:
        existing = _semantic_existing_tags(
            search,
            document_state,
            limit=semantic_limit,
            query_embedding=query_embedding,
            embedding_model_digest=embedding_model_digest,
            embedding_index_version=embedding_index_version,
            embedding_profile=embedding_profile,
        )
        bounded_existing = existing[:max_candidates]
        candidate_set_digest = _candidate_set_digest(bounded_existing)
        resolved_candidate_digest = await _model_digest(
            selected_gateway,
            role="generation",
            model=candidate_model,
            digest=candidate_model_digest,
        )
        resolved_decision_digest = await _model_digest(
            selected_decision_gateway,
            role="decision",
            model=decision_model,
            digest=decision_model_digest,
        )
        if cache is not None and not force_refresh:
            cached = cache.lookup(
                file_hash,
                file_id=file_id,
                file_path=file_path,
                document_state_digest=document_state_digest,
                extraction_version=extraction_version,
                candidate_model_digest=resolved_candidate_digest,
                decision_model_digest=resolved_decision_digest,
                candidate_set_digest=candidate_set_digest,
                max_candidates=max_candidates,
                confirmation_threshold=confirmation_threshold,
            )
            if cached is not None:
                refreshed = _refresh_cached_spelling(cached, bounded_existing)
                if refreshed != cached:
                    cache.store(refreshed)
                return refreshed

        available_for_generation = max_candidates - len(bounded_existing)
        generated: tuple[str, ...] = ()
        if available_for_generation > 0:
            generated = await _generate_candidates(
                selected_gateway,
                document_state=document_state,
                model=candidate_model,
                limit=available_for_generation,
                existing=bounded_existing,
            )

        candidates, generated_keys = _merge_candidates(
            bounded_existing,
            generated,
            max_candidates=max_candidates,
        )
        decisions: tuple[TagDecision, ...] = ()
        if candidates:
            decision_result = decide_tags(
                document_state,
                candidates,
                decision_model,
                gateway=selected_decision_gateway,
            )
            decided = (
                await decision_result
                if inspect.isawaitable(decision_result)
                else decision_result
            )
            decisions = tuple(
                _typed_decision(
                    decision,
                    is_new=_tag_key(decision.tag) in generated_keys,
                    confirmation_threshold=confirmation_threshold,
                )
                for decision in decided.decisions
            )

        stored = CachedTagRecommendations(
            file_id=file_id,
            file_path=file_path,
            file_hash=file_hash,
            document_state_digest=document_state_digest,
            extraction_version=extraction_version,
            candidate_model_digest=resolved_candidate_digest,
            decision_model_digest=resolved_decision_digest,
            candidate_set_digest=candidate_set_digest,
            max_candidates=max_candidates,
            confirmation_threshold=confirmation_threshold,
            decisions=decisions,
        )
        if cache is not None:
            cache.store(stored)
        return stored
    finally:
        if owns_gateway:
            await _close_gateway(selected_gateway)


class CandidateService:
    def __init__(
        self,
        search: CandidateSearch,
        *,
        gateway: CandidateGateway | None = None,
        decision_gateway: Any | None = None,
        cache: TagRecommendationCache | None = None,
        candidate_model: str = DEFAULT_CANDIDATE_MODEL,
        decision_model: str = DEFAULT_DECISION_MODEL,
        candidate_model_digest: str | None = None,
        decision_model_digest: str | None = None,
        max_candidates: int = DEFAULT_MAX_TAG_CANDIDATES,
        semantic_limit: int = DEFAULT_SEMANTIC_RESULT_LIMIT,
        confirmation_threshold: float = DEFAULT_CONFIRMATION_THRESHOLD,
    ) -> None:
        self._search = search
        self._gateway = gateway
        self._decision_gateway = decision_gateway
        self._cache = cache
        self._candidate_model = candidate_model
        self._decision_model = decision_model
        self._candidate_model_digest = candidate_model_digest
        self._decision_model_digest = decision_model_digest
        self._max_candidates = max_candidates
        self._semantic_limit = semantic_limit
        self._confirmation_threshold = confirmation_threshold

    def recommend(
        self,
        document_state: str,
        *,
        file_hash: str,
        extraction_version: str,
        **kwargs: Any,
    ) -> CachedTagRecommendations:
        defaults: dict[str, Any] = {
            "search": self._search,
            "gateway": self._gateway,
            "decision_gateway": self._decision_gateway,
            "cache": self._cache,
            "candidate_model": self._candidate_model,
            "decision_model": self._decision_model,
            "candidate_model_digest": self._candidate_model_digest,
            "decision_model_digest": self._decision_model_digest,
            "max_candidates": self._max_candidates,
            "semantic_limit": self._semantic_limit,
            "confirmation_threshold": self._confirmation_threshold,
        }
        defaults.update(kwargs)
        return recommend_tags(
            document_state,
            file_hash=file_hash,
            extraction_version=extraction_version,
            **defaults,
        )

    def recommend_async(
        self,
        document_state: str,
        *,
        file_hash: str,
        extraction_version: str,
        **kwargs: Any,
    ) -> Coroutine[Any, Any, CachedTagRecommendations]:
        defaults: dict[str, Any] = {
            "search": self._search,
            "gateway": self._gateway,
            "decision_gateway": self._decision_gateway,
            "cache": self._cache,
            "candidate_model": self._candidate_model,
            "decision_model": self._decision_model,
            "candidate_model_digest": self._candidate_model_digest,
            "decision_model_digest": self._decision_model_digest,
            "max_candidates": self._max_candidates,
            "semantic_limit": self._semantic_limit,
            "confirmation_threshold": self._confirmation_threshold,
        }
        defaults.update(kwargs)
        return recommend_tags_async(
            document_state,
            file_hash=file_hash,
            extraction_version=extraction_version,
            **defaults,
        )


async def _model_digest(
    gateway: Any,
    *,
    role: ModelRole,
    model: str,
    digest: str | None,
) -> str:
    if digest is not None and digest.strip():
        return digest.strip()
    profile_method = getattr(gateway, "model_profile", None)
    if profile_method is None:
        raise ValueError(
            f"A {role} model digest is required when the gateway cannot provide one."
        )
    profile = await profile_method(role, model)
    if not isinstance(profile, ModelProfile):
        try:
            profile = ModelProfile.model_validate(profile)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"The {role} model profile is invalid.") from exc
    return profile.digest


def _semantic_existing_tags(
    search: CandidateSearch,
    document_state: str,
    *,
    limit: int,
    query_embedding: Sequence[float] | None,
    embedding_model_digest: str | None,
    embedding_index_version: str | None,
    embedding_profile: ModelProfile | None,
) -> tuple[str, ...]:
    kwargs: dict[str, Any] = {"limit": limit}
    if embedding_model_digest is not None:
        kwargs["model_digest"] = embedding_model_digest
    if embedding_index_version is not None:
        kwargs["index_version"] = embedding_index_version
    if embedding_profile is not None:
        kwargs["embedding_profile"] = embedding_profile
    results = search.search(document_state, query_embedding, **kwargs)
    tags: list[str] = []
    seen: set[str] = set()
    for result in results:
        raw_tags = (
            result.get("tags", ())
            if isinstance(result, Mapping)
            else getattr(result, "tags", ())
        )
        if isinstance(raw_tags, str):
            raw_tags = (raw_tags,)
        if not isinstance(raw_tags, Sequence):
            continue
        for raw_tag in raw_tags:
            normalized = _display_tag(raw_tag)
            if normalized is None:
                continue
            key = _tag_key(normalized)
            if key in seen:
                continue
            seen.add(key)
            tags.append(normalized)
    return tuple(tags)


def _candidate_set_digest(tags: Sequence[str]) -> str:
    normalized = [_tag_key(tag) for tag in tags]
    canonical = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _refresh_cached_spelling(
    cached: CachedTagRecommendations,
    existing: Sequence[str],
) -> CachedTagRecommendations:
    current_spelling = {_tag_key(tag): tag for tag in existing}
    changed = False
    decisions: list[TagDecision] = []
    for decision in cached.decisions:
        current = current_spelling.get(_tag_key(decision.tag))
        if not decision.is_new and current is not None and current != decision.tag:
            decision = decision.model_copy(update={"tag": current})
            changed = True
        decisions.append(decision)
    if not changed:
        return cached
    return cached.model_copy(update={"decisions": tuple(decisions)})


async def _generate_candidates(
    gateway: CandidateGateway,
    *,
    document_state: str,
    model: str,
    limit: int,
    existing: Sequence[str],
) -> tuple[str, ...]:
    schema = CandidateEnvelope.model_json_schema()
    exclusions = ", ".join(existing) if existing else "none"
    raw = await gateway.chat(
        model,
        messages=(
            {
                "role": "system",
                "content": (
                    "Generate concise, reusable file tags. Return JSON only. "
                    "Do not repeat existing tags and do not invent facts unsupported "
                    "by the document."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Generate no more than {limit} new tag candidates for this "
                    f"document state. Exclude these existing tags exactly or by case: "
                    f"{exclusions}.\n\n{document_state}"
                ),
            },
        ),
        schema=schema,
    )
    try:
        envelope = _CANDIDATE_ENVELOPE.validate_json(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Tag candidate generation returned invalid structured data.") from exc
    generated: list[str] = []
    seen = {_tag_key(item) for item in existing}
    for candidate in envelope.candidates:
        normalized = _display_tag(candidate)
        if normalized is None:
            continue
        key = _tag_key(normalized)
        if key in seen:
            continue
        seen.add(key)
        generated.append(normalized)
        if len(generated) >= limit:
            break
    return tuple(generated)


def _merge_candidates(
    existing: Sequence[str],
    generated: Sequence[str],
    *,
    max_candidates: int,
) -> tuple[tuple[str, ...], frozenset[str]]:
    selected: list[str] = []
    keys: set[str] = set()
    generated_keys: set[str] = set()
    for source, values in (("existing", existing), ("generated", generated)):
        for value in values:
            normalized = _display_tag(value)
            if normalized is None:
                continue
            key = _tag_key(normalized)
            if key in keys:
                continue
            keys.add(key)
            selected.append(normalized)
            if source == "generated":
                generated_keys.add(key)
            if len(selected) >= max_candidates:
                return tuple(selected), frozenset(generated_keys)
    return tuple(selected), frozenset(generated_keys)


def _typed_decision(
    decision: TagDecision,
    *,
    is_new: bool,
    confirmation_threshold: float,
) -> TagDecision:
    outcome = decision.decision
    if decision.confidence < confirmation_threshold:
        outcome = "review"
    return TagDecision(
        tag=decision.tag,
        decision=outcome,
        confidence=decision.confidence,
        rationale=decision.rationale,
        is_new=is_new,
    )


def _display_tag(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split())
    if not normalized or len(normalized) > 255:
        return None
    return normalized


def _tag_key(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip().casefold()


def _validate_request(
    *,
    document_state: str,
    file_hash: str,
    extraction_version: str,
    candidate_model: str,
    decision_model: str,
    max_candidates: int,
    semantic_limit: int,
    confirmation_threshold: float,
) -> None:
    for value, label in (
        (document_state, "Document state"),
        (file_hash, "File hash"),
        (extraction_version, "Extraction version"),
        (candidate_model, "Candidate model"),
        (decision_model, "Decision model"),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must be a nonempty string.")
    if max_candidates < 1 or max_candidates > MAX_SYSTEM_ONE_QUESTIONS:
        raise ValueError(
            f"max_candidates must be between 1 and {MAX_SYSTEM_ONE_QUESTIONS}."
        )
    if semantic_limit < 1:
        raise ValueError("semantic_limit must be positive.")
    if not 0.0 <= confirmation_threshold <= 1.0:
        raise ValueError("confirmation_threshold must be between 0 and 1.")


async def _close_gateway(gateway: CandidateGateway) -> None:
    close = getattr(gateway, "close", None)
    if close is None:
        return
    result = close()
    if hasattr(result, "__await__"):
        await result


def _run_sync(coroutine: Coroutine[Any, Any, T]) -> T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coroutine).result()
