import asyncio
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Coroutine,
    Mapping,
    Sequence,
)
from contextlib import asynccontextmanager
from dataclasses import dataclass
from hashlib import sha256
import inspect
import json
import re
import threading
from typing import Any, Callable, Protocol, TypeAlias, TypeVar, cast
import unicodedata

from pydantic import ValidationError

from ..adapters.ollama.gateway import OllamaGateway
from ..domain.contracts import ModelProfile
from ..domain.errors import (
    GenerationContextRejectedError,
    ModelIncompatibleError,
    StructuredOutputError,
    UncitedAnswerError,
)
from ..search.hybrid import SearchFilters
from ..search.results import SearchCitation, SearchResult, make_result
from .citations import (
    CitedAnswer,
    NOT_FOUND_ANSWER,
    answer_schema,
    parse_cited_answer,
)
from .planner import DEFAULT_UNPROFILED_CONTEXT_LENGTH, SearchPlan
from .planner import build_search_plan_async

DEFAULT_GENERATION_MODEL = "gemma4:e4b"
DEFAULT_EVIDENCE_LIMIT = 24
DEFAULT_RERANK_TOP_K = 12
DEFAULT_PER_QUERY_LIMIT = 20
DEFAULT_MAX_SECTIONS_PER_FILE = 2
MAX_CONTEXT_RETRIES = 3
OUTPUT_TOKEN_RESERVE = 1_024
ASCII_BYTES_PER_TOKEN = 4
MAX_QUESTION_LENGTH = 1_024

T = TypeVar("T")
JsonMapping: TypeAlias = Mapping[str, Any]
ChatHistory: TypeAlias = tuple[tuple[str, str], ...]
IndexIdentity: TypeAlias = tuple[
    str | None,
    str | None,
    int | None,
    str | None,
    str | None,
]
EvidenceInput: TypeAlias = (
    SearchResult
    | SearchCitation
    | Sequence[SearchResult | SearchCitation]
)
Planner: TypeAlias = Callable[..., Awaitable[SearchPlan] | SearchPlan]


class SearchGateway(Protocol):
    def search(
        self,
        query: str,
        query_embedding: Sequence[float] | None = None,
        *,
        limit: int = 10,
        filters: SearchFilters | None = None,
        model_digest: str | None = None,
        index_version: str | None = None,
        embedding_profile: ModelProfile | None = None,
    ) -> Sequence[SearchResult]: ...


class GenerationGateway(Protocol):
    async def chat(
        self,
        model: str,
        *,
        messages: Sequence[Mapping[str, Any]],
        schema: Mapping[str, Any] | None = None,
        keep_alive: float | str | None = "5m",
    ) -> str: ...
    def chat_stream(
        self,
        model: str,
        *,
        messages: Sequence[Mapping[str, Any]],
        schema: Mapping[str, Any] | None = None,
        keep_alive: float | str | None = "5m",
    ) -> AsyncIterator[str]: ...


class EmbeddingGateway(Protocol):
    async def embed(
        self,
        model: str,
        text: str | Sequence[str],
        *,
        dimensions: int | None = None,
        truncate: bool = False,
        keep_alive: float | str | None = "5m",
    ) -> tuple[tuple[float, ...], ...]: ...


@dataclass(frozen=True, slots=True)
class _SessionStateKey:
    session_id: str
    original_question: str
    markdown_context: str
    index_identity: IndexIdentity
    evidence_identity: str | None


@dataclass(frozen=True, slots=True)
class _SessionState:
    key: _SessionStateKey
    version: int
    evidence: tuple[SearchResult, ...]
    history: ChatHistory
    answer: CitedAnswer | None = None


class RagService:
    def __init__(
        self,
        search: SearchGateway | None = None,
        gateway: GenerationGateway | None = None,
        embedding_gateway: Any | None = None,
        planner: Planner = build_search_plan_async,
        model: str = DEFAULT_GENERATION_MODEL,
        evidence_limit: int = DEFAULT_EVIDENCE_LIMIT,
        per_query_limit: int = DEFAULT_PER_QUERY_LIMIT,
        max_sections_per_file: int = DEFAULT_MAX_SECTIONS_PER_FILE,
        embedding_profile: ModelProfile | None = None,
        model_digest: str | None = None,
        index_version: str | None = None,
    ) -> None:
        if evidence_limit <= 0:
            raise ValueError("The evidence limit must be positive.")
        if per_query_limit <= 0:
            raise ValueError("The per-query search limit must be positive.")
        if max_sections_per_file <= 0:
            raise ValueError("The per-file evidence limit must be positive.")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("A local generation model name is required.")
        self._search = search
        self._gateway: GenerationGateway = gateway or OllamaGateway()
        self._planner = planner
        self.model = model.strip()
        self.evidence_limit = evidence_limit
        self.per_query_limit = per_query_limit
        self.max_sections_per_file = max_sections_per_file
        self._embedding_gateway = embedding_gateway or gateway or self._gateway
        self._embedding_profile = embedding_profile
        self._model_digest = model_digest
        self._index_version = index_version
        self._session_states: dict[str, _SessionState] = {}
        self._session_locks: dict[str, threading.Lock] = {}
        self._session_registry_lock = threading.Lock()

    @property
    def evidence(self) -> tuple[SearchResult, ...]:
        with self._session_registry_lock:
            return tuple(
                result
                for state in self._session_states.values()
                for result in state.evidence
            )

    def get_session_answer(self, session_id: str) -> CitedAnswer | None:
        checked_session = _validate_session_identity(session_id, None)
        with self._session_registry_lock:
            state = self._session_states.get(checked_session)
            if state is None:
                return None
            answer = state.answer
        return answer

    def close_session(self, session_id: str) -> None:
        checked_session = _validate_session_identity(session_id, None)
        with self._session_registry_lock:
            self._session_states.pop(checked_session, None)
            self._session_locks.pop(checked_session, None)

    def close_all_sessions(self) -> None:
        with self._session_registry_lock:
            self._session_states.clear()
            self._session_locks.clear()

    def answer_question(
        self,
        question: str,
        markdown_context: str | EvidenceInput = "",
        evidence: EvidenceInput | None = None,
        *,
        rerank: bool = False,
        rerank_top_k: int = DEFAULT_RERANK_TOP_K,
        max_context_retries: int = MAX_CONTEXT_RETRIES,
        session_id: str | None = None,
        conversation_id: str | None = None,
        retrieval_question: str | None = None,
        cancellation_check: Callable[[], bool] | None = None,
    ) -> CitedAnswer:
        return _run_sync(
            self.answer_question_async(
                question,
                markdown_context,
                evidence,
                rerank=rerank,
                rerank_top_k=rerank_top_k,
                max_context_retries=max_context_retries,
                session_id=session_id,
                conversation_id=conversation_id,
                retrieval_question=retrieval_question,
                cancellation_check=cancellation_check,
            )
        )

    async def answer_question_async(
        self,
        question: str,
        markdown_context: str | EvidenceInput = "",
        evidence: EvidenceInput | None = None,
        *,
        rerank: bool = False,
        rerank_top_k: int = DEFAULT_RERANK_TOP_K,
        max_context_retries: int = MAX_CONTEXT_RETRIES,
        session_id: str | None = None,
        conversation_id: str | None = None,
        retrieval_question: str | None = None,
        cancellation_check: Callable[[], bool] | None = None,
    ) -> CitedAnswer:
        checked_question = _validate_question(question)
        if not isinstance(markdown_context, str):
            if evidence is not None:
                raise TypeError(
                    "Evidence cannot be supplied in both positional arguments."
                )
            evidence = markdown_context
            checked_context = ""
        else:
            checked_context = markdown_context.strip()
        checked_session = _validate_session_identity(session_id, conversation_id)
        if session_id is None and conversation_id is None:
            checked_session = (
                "default:"
                + sha256(
                    f"{checked_question}\0{checked_context}".encode("utf-8")
                ).hexdigest()
            )
        explicit_retrieval_question = (
            _validate_question(retrieval_question)
            if retrieval_question is not None
            else None
        )
        supplied_evidence = (
            _bounded_evidence(
                _coerce_evidence(evidence),
                limit=self.evidence_limit,
                max_sections_per_file=self.max_sections_per_file,
            )
            if evidence is not None
            else None
        )

        async with self._session_guard(checked_session):
            with self._session_registry_lock:
                previous_state = self._session_states.get(checked_session)
            profile = _embedding_profile(self._search, self._embedding_profile)
            identity = _index_identity(
                self._search,
                profile,
                model_digest=self._model_digest,
                index_version=self._index_version,
            )
            can_reuse_retrieval = (
                previous_state is not None
                and previous_state.key.markdown_context == checked_context
                and previous_state.key.index_identity == identity
            )
            original_question = (
                explicit_retrieval_question
                or (
                    previous_state.key.original_question
                    if can_reuse_retrieval and previous_state is not None
                    else checked_question
                )
            )
            supplied_identity = (
                _evidence_identity(supplied_evidence)
                if supplied_evidence is not None
                else None
            )
            evidence_identity = (
                supplied_identity
                if supplied_evidence is not None
                else (
                    previous_state.key.evidence_identity
                    if can_reuse_retrieval and previous_state is not None
                    else None
                )
            )
            key = _SessionStateKey(
                session_id=checked_session,
                original_question=original_question,
                markdown_context=checked_context,
                index_identity=identity,
                evidence_identity=evidence_identity,
            )
            reuse = previous_state is not None and previous_state.key == key
            if reuse:
                assert previous_state is not None
                selected = previous_state.evidence
                previous_history = previous_state.history
                version = previous_state.version
            else:
                if supplied_evidence is not None:
                    selected = supplied_evidence
                else:
                    if self._search is None:
                        raise ValueError(
                            "Search evidence or a HybridSearch service is required."
                        )
                    planned = self._planner(
                        original_question,
                        checked_context,
                        gateway=self._gateway,
                    )
                    plan = (
                        planned
                        if isinstance(planned, SearchPlan)
                        else await planned
                    )
                    selected = await execute_search_plan_async(
                        plan,
                        self._search,
                        embedding_gateway=self._embedding_gateway,
                        embedding_profile=profile,
                        model_digest=self._model_digest,
                        index_version=self._index_version,
                        per_query_limit=self.per_query_limit,
                        limit=self.evidence_limit,
                        max_sections_per_file=self.max_sections_per_file,
                        cancellation_check=cancellation_check,
                    )
                previous_history = ()
                version = previous_state.version if previous_state is not None else 0

            answer = await _generate_answer_async(
                checked_question,
                selected,
                model=self.model,
                gateway=self._gateway,
                markdown_context=checked_context,
                rerank=rerank,
                rerank_top_k=rerank_top_k,
                max_context_retries=max_context_retries,
                history=previous_history,
                cancellation_check=cancellation_check,
            )
            updated = _SessionState(
                key=key,
                version=version + 1,
                evidence=selected,
                history=(*previous_history[-4:], (checked_question, answer.text)),
                answer=answer,
            )
            with self._session_registry_lock:
                self._session_states[checked_session] = updated
            return answer

    async def answer_question_stream_async(
        self,
        question: str,
        markdown_context: str | EvidenceInput = "",
        evidence: EvidenceInput | None = None,
        *,
        rerank: bool = False,
        rerank_top_k: int = DEFAULT_RERANK_TOP_K,
        max_context_retries: int = MAX_CONTEXT_RETRIES,
        session_id: str | None = None,
        conversation_id: str | None = None,
        retrieval_question: str | None = None,
        cancellation_check: Callable[[], bool] | None = None,
    ) -> AsyncIterator[str]:
        checked_question = _validate_question(question)
        if not isinstance(markdown_context, str):
            if evidence is not None:
                raise TypeError(
                    "Evidence cannot be supplied in both positional arguments."
                )
            evidence = markdown_context
            checked_context = ""
        else:
            checked_context = markdown_context.strip()
        checked_session = _validate_session_identity(session_id, conversation_id)
        if session_id is None and conversation_id is None:
            checked_session = (
                "default:"
                + sha256(
                    f"{checked_question}\0{checked_context}".encode("utf-8")
                ).hexdigest()
            )
        explicit_retrieval_question = (
            _validate_question(retrieval_question)
            if retrieval_question is not None
            else None
        )
        supplied_evidence = (
            _bounded_evidence(
                _coerce_evidence(evidence),
                limit=self.evidence_limit,
                max_sections_per_file=self.max_sections_per_file,
            )
            if evidence is not None
            else None
        )

        async with self._session_guard(checked_session):
            with self._session_registry_lock:
                previous_state = self._session_states.get(checked_session)
            profile = _embedding_profile(self._search, self._embedding_profile)
            identity = _index_identity(
                self._search,
                profile,
                model_digest=self._model_digest,
                index_version=self._index_version,
            )
            can_reuse_retrieval = (
                previous_state is not None
                and previous_state.key.markdown_context == checked_context
                and previous_state.key.index_identity == identity
            )
            original_question = (
                explicit_retrieval_question
                or (
                    previous_state.key.original_question
                    if can_reuse_retrieval and previous_state is not None
                    else checked_question
                )
            )
            supplied_identity = (
                _evidence_identity(supplied_evidence)
                if supplied_evidence is not None
                else None
            )
            evidence_identity = (
                supplied_identity
                if supplied_evidence is not None
                else (
                    previous_state.key.evidence_identity
                    if can_reuse_retrieval and previous_state is not None
                    else None
                )
            )
            key = _SessionStateKey(
                session_id=checked_session,
                original_question=original_question,
                markdown_context=checked_context,
                index_identity=identity,
                evidence_identity=evidence_identity,
            )
            reuse = previous_state is not None and previous_state.key == key
            if reuse:
                assert previous_state is not None
                selected = previous_state.evidence
                previous_history = previous_state.history
                version = previous_state.version
            else:
                if supplied_evidence is not None:
                    selected = supplied_evidence
                else:
                    if self._search is None:
                        raise ValueError(
                            "Search evidence or a HybridSearch service is required."
                        )
                    planned = self._planner(
                        original_question,
                        checked_context,
                        gateway=self._gateway,
                    )
                    plan = (
                        planned
                        if isinstance(planned, SearchPlan)
                        else await planned
                    )
                    selected = await execute_search_plan_async(
                        plan,
                        self._search,
                        embedding_gateway=self._embedding_gateway,
                        embedding_profile=profile,
                        model_digest=self._model_digest,
                        index_version=self._index_version,
                        per_query_limit=self.per_query_limit,
                        limit=self.evidence_limit,
                        max_sections_per_file=self.max_sections_per_file,
                        cancellation_check=cancellation_check,
                    )
                previous_history = ()
                version = previous_state.version if previous_state is not None else 0

            answer: CitedAnswer | None = None
            async for fragment, final_answer in _generate_answer_stream_async(
                checked_question,
                selected,
                model=self.model,
                gateway=self._gateway,
                markdown_context=checked_context,
                rerank=rerank,
                rerank_top_k=rerank_top_k,
                max_context_retries=max_context_retries,
                history=previous_history,
                cancellation_check=cancellation_check,
            ):
                if fragment:
                    yield fragment
                if final_answer is not None:
                    answer = final_answer
            if answer is None:
                answer = CitedAnswer(
                    question=checked_question,
                    text=NOT_FOUND_ANSWER,
                    evidence=selected,
                )
            updated = _SessionState(
                key=key,
                version=version + 1,
                evidence=selected,
                history=(*previous_history[-4:], (checked_question, answer.text)),
                answer=answer,
            )
            with self._session_registry_lock:
                self._session_states[checked_session] = updated

    @asynccontextmanager
    async def _session_guard(self, session_id: str) -> AsyncIterator[None]:
        with self._session_registry_lock:
            lock = self._session_locks.setdefault(session_id, threading.Lock())
        acquired = asyncio.create_task(asyncio.to_thread(lock.acquire))
        try:
            await asyncio.shield(acquired)
        except asyncio.CancelledError:
            await asyncio.shield(acquired)
            lock.release()
            raise
        try:
            yield
        finally:
            lock.release()

    async def answer_plan_async(
        self,
        plan: SearchPlan,
        markdown_context: str = "",
        *,
        rerank: bool = False,
        session_id: str | None = None,
        conversation_id: str | None = None,
        retrieval_question: str | None = None,
        cancellation_check: Callable[[], bool] | None = None,
    ) -> CitedAnswer:
        if not isinstance(plan, SearchPlan):
            raise TypeError("plan must be a SearchPlan.")
        if self._search is None:
            raise ValueError("A HybridSearch service is required.")
        evidence = await execute_search_plan_async(
            plan,
            self._search,
            embedding_gateway=self._embedding_gateway,
            embedding_profile=_embedding_profile(
                self._search,
                self._embedding_profile,
            ),
            model_digest=self._model_digest,
            index_version=self._index_version,
            per_query_limit=self.per_query_limit,
            limit=self.evidence_limit,
            max_sections_per_file=self.max_sections_per_file,
            cancellation_check=cancellation_check,
        )
        return await self.answer_question_async(
            plan.question,
            markdown_context,
            evidence,
            rerank=rerank,
            session_id=session_id,
            conversation_id=conversation_id,
            retrieval_question=retrieval_question or plan.question,
            cancellation_check=cancellation_check,
        )

    def answer_plan(
        self,
        plan: SearchPlan,
        markdown_context: str = "",
        *,
        rerank: bool = False,
        session_id: str | None = None,
        conversation_id: str | None = None,
        retrieval_question: str | None = None,
        cancellation_check: Callable[[], bool] | None = None,
    ) -> CitedAnswer:
        return _run_sync(
            self.answer_plan_async(
                plan,
                markdown_context,
                rerank=rerank,
                session_id=session_id,
                conversation_id=conversation_id,
                retrieval_question=retrieval_question,
                cancellation_check=cancellation_check,
            )
        )


def execute_search_plan(
    plan: SearchPlan,
    search: SearchGateway,
    *,
    embedding_gateway: Any | None = None,
    embedding_profile: ModelProfile | None = None,
    model_digest: str | None = None,
    index_version: str | None = None,
    per_query_limit: int = DEFAULT_PER_QUERY_LIMIT,
    limit: int = DEFAULT_EVIDENCE_LIMIT,
    max_sections_per_file: int = DEFAULT_MAX_SECTIONS_PER_FILE,
    cancellation_check: Callable[[], bool] | None = None,
) -> tuple[SearchResult, ...]:
    return _run_sync(
        execute_search_plan_async(
            plan,
            search,
            embedding_gateway=embedding_gateway,
            embedding_profile=embedding_profile,
            model_digest=model_digest,
            index_version=index_version,
            per_query_limit=per_query_limit,
            limit=limit,
            max_sections_per_file=max_sections_per_file,
            cancellation_check=cancellation_check,
        )
    )


async def execute_search_plan_async(
    plan: SearchPlan,
    search: SearchGateway,
    *,
    embedding_gateway: Any | None = None,
    embedding_profile: ModelProfile | None = None,
    model_digest: str | None = None,
    index_version: str | None = None,
    per_query_limit: int = DEFAULT_PER_QUERY_LIMIT,
    limit: int = DEFAULT_EVIDENCE_LIMIT,
    max_sections_per_file: int = DEFAULT_MAX_SECTIONS_PER_FILE,
    cancellation_check: Callable[[], bool] | None = None,
) -> tuple[SearchResult, ...]:
    if not isinstance(plan, SearchPlan):
        raise TypeError("plan must be a SearchPlan.")
    if per_query_limit <= 0:
        raise ValueError("The per-query search limit must be positive.")
    if limit < 0:
        raise ValueError("The evidence limit cannot be negative.")
    if max_sections_per_file <= 0:
        raise ValueError("The per-file evidence limit must be positive.")
    if limit == 0:
        return ()
    if cancellation_check is not None and cancellation_check():
        raise asyncio.CancelledError

    selected_filters, retrieval_constraints = _plan_search_options(plan)
    queries = tuple(
        dict.fromkeys(
            (*plan.exact_terms, *retrieval_constraints, *plan.paraphrase_queries)
        )
    )
    selected_profile = _embedding_profile(search, embedding_profile)
    identity = _index_identity(
        search,
        selected_profile,
        model_digest=model_digest,
        index_version=index_version,
    )
    embeddings: tuple[tuple[float, ...], ...] | None = None
    if selected_profile is not None:
        embeddings = await _embed_queries(
            embedding_gateway,
            selected_profile,
            queries,
            cancellation_check=cancellation_check,
        )

    candidates: list[SearchResult] = []
    for index, query in enumerate(queries):
        if cancellation_check is not None and cancellation_check():
            raise asyncio.CancelledError
        if embeddings is None:
            if selected_filters is None:
                candidates.extend(search.search(query, limit=per_query_limit))
            else:
                candidates.extend(
                    search.search(
                        query,
                        filters=selected_filters,
                        limit=per_query_limit,
                    )
                )
            continue
        if selected_filters is None:
            candidates.extend(
                search.search(
                    query,
                    embeddings[index],
                    limit=per_query_limit,
                    model_digest=identity[1],
                    index_version=identity[3],
                )
            )
        else:
            candidates.extend(
                search.search(
                    query,
                    embeddings[index],
                    filters=selected_filters,
                    limit=per_query_limit,
                    model_digest=identity[1],
                    index_version=identity[3],
                )
            )
    candidates = [
        result
        for result in _expand_evidence(candidates)
        if not _result_is_excluded(result, plan.exclusions)
    ]
    return _bounded_evidence(
        candidates,
        limit=limit,
        max_sections_per_file=max_sections_per_file,
    )


def _embedding_profile(
    search: SearchGateway | None,
    explicit: ModelProfile | None,
) -> ModelProfile | None:
    selected = explicit
    if selected is None and search is not None:
        raw = getattr(search, "embedding_profile", None)
        if raw is None:
            raw = getattr(search, "profile", None)
        selected = raw
    if selected is None:
        return None
    if not isinstance(selected, ModelProfile):
        raise ModelIncompatibleError(
            "The active indexed embedding profile is invalid."
        )
    if selected.role != "embedding" or selected.dimensions is None:
        raise ModelIncompatibleError(
            "The active indexed profile is not a dimensioned embedding model."
        )
    return selected


def _index_identity(
    search: SearchGateway | None,
    profile: ModelProfile | None,
    *,
    model_digest: str | None,
    index_version: str | None,
) -> IndexIdentity:
    digest = model_digest
    version = index_version
    fingerprint = _index_fingerprint(search)
    if search is not None:
        digest = digest or cast(str | None, getattr(search, "model_digest", None))
        version = version or cast(
            str | None,
            getattr(search, "index_version", None),
        )
    if profile is None:
        return (None, digest, None, version, fingerprint)
    if digest is not None and digest != profile.digest:
        raise ModelIncompatibleError(
            "The active indexed model digest does not match its embedding profile."
        )
    profile_version = profile.default_index_version
    selected_version = version or profile_version
    if selected_version is None:
        raise ModelIncompatibleError(
            "The active indexed embedding profile has no index version."
        )
    return (
        profile.model,
        profile.digest,
        profile.dimensions,
        selected_version,
        fingerprint,
    )


def _index_fingerprint(search: SearchGateway | None) -> str | None:
    if search is None:
        return None
    fingerprint = getattr(search, "index_fingerprint", None)
    if callable(fingerprint):
        fingerprint = fingerprint()
    if fingerprint is None:
        return None
    if not isinstance(fingerprint, str) or not fingerprint.strip():
        raise ModelIncompatibleError("The active index fingerprint is invalid.")
    return fingerprint


def _plan_search_options(
    plan: SearchPlan,
) -> tuple[SearchFilters | None, tuple[str, ...]]:
    tags: list[str] = []
    paths: list[str] = []
    file_types: list[str] = []
    excluded_file_types: list[str] = []
    retrieval_constraints: list[str] = []
    for constraint in (*plan.filters, *plan.normalized_constraints):
        key, value = _typed_constraint(constraint)
        if key in {"tag", "tags"}:
            tags.extend(_constraint_values(value))
        elif key in {"path", "folder", "directory"}:
            paths.append(value)
        elif key in {
            "filetype",
            "filetypes",
            "extension",
            "extensions",
            "ext",
            "mime",
            "mimetype",
            "type",
        }:
            file_types.extend(_constraint_values(value))
        else:
            retrieval_constraints.append(constraint)
    for exclusion in plan.exclusions:
        key, value = _typed_constraint(exclusion)
        if key in {
            "filetype",
            "filetypes",
            "extension",
            "extensions",
            "ext",
            "mime",
            "mimetype",
            "type",
        }:
            excluded_file_types.extend(_constraint_values(value))
    unique_paths = tuple(
        {
            path.casefold(): path
            for path in paths
            if path.strip()
        }.values()
    )
    selected = (
        SearchFilters(
            tags=tuple(dict.fromkeys(tags)),
            path=unique_paths[0] if len(unique_paths) == 1 else None,
            paths=unique_paths if len(unique_paths) > 1 else (),
            file_types=tuple(dict.fromkeys(file_types)),
            excluded_file_types=tuple(dict.fromkeys(excluded_file_types)),
        )
        if tags or unique_paths or file_types or excluded_file_types
        else None
    )
    return selected, tuple(retrieval_constraints)


def _typed_constraint(constraint: str) -> tuple[str, str]:
    key, separator, value = constraint.partition(":")
    if not separator or not value.strip():
        return "", constraint.strip()
    normalized_key = re.sub(r"[\s_-]+", "", unicodedata.normalize("NFKC", key))
    return normalized_key.casefold(), value.strip()


def _constraint_values(value: str) -> tuple[str, ...]:
    return tuple(
        item.strip()
        for item in re.split(r"[,|]", value)
        if item.strip()
    )


def _result_is_excluded(
    result: SearchResult,
    exclusions: Sequence[str],
) -> bool:
    text = " ".join(
        (
            result.file_path,
            *result.tags,
            *(section.heading or "" for section in result.matched_sections),
            *(section.text for section in result.matched_sections),
        )
    )
    for exclusion in exclusions:
        key, value = _typed_constraint(exclusion)
        if not value:
            continue
        if key in {"tag", "tags"}:
            excluded_tags = {
                _normalize_match(item) for item in _constraint_values(value)
            }
            if any(_normalize_match(tag) in excluded_tags for tag in result.tags):
                return True
        elif key in {"path", "folder", "directory"}:
            if _normalize_match(value) in _normalize_match(result.file_path):
                return True
        elif key in {
            "filetype",
            "filetypes",
            "extension",
            "extensions",
            "ext",
            "mime",
            "mimetype",
            "type",
        }:
            normalized_path = _normalize_path(result.file_path)
            for file_type in _constraint_values(value):
                normalized_type = unicodedata.normalize(
                    "NFKC",
                    file_type,
                ).strip().casefold()
                if "/" in normalized_type:
                    if (
                        result.mime_type is not None
                        and unicodedata.normalize(
                            "NFKC",
                            result.mime_type,
                        ).strip().casefold()
                        == normalized_type
                    ):
                        return True
                elif normalized_path.endswith(
                    f".{normalized_type.removeprefix('.')}"
                ):
                    return True
        elif _normalize_match(value) in _normalize_match(text):
            return True
    return False


def _normalize_match(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(
        token
        for token in re.findall(r"[^\W_]+", normalized)
    )


def _normalize_path(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold().replace("\\", "/")


async def _embed_queries(
    gateway: Any | None,
    profile: ModelProfile,
    queries: Sequence[str],
    *,
    cancellation_check: Callable[[], bool] | None = None,
) -> tuple[tuple[float, ...], ...]:
    if cancellation_check is not None and cancellation_check():
        raise asyncio.CancelledError
    dimensions = profile.dimensions
    assert dimensions is not None
    embed = getattr(gateway, "embed", None)
    if not callable(embed):
        raise ModelIncompatibleError(
            "The configured gateway cannot embed retrieval queries for the active index."
        )
    result = embed(
        profile.model,
        tuple(queries),
        dimensions=dimensions,
        truncate=False,
    )
    if inspect.isawaitable(result):
        result = await result
    if cancellation_check is not None and cancellation_check():
        raise asyncio.CancelledError
    if not isinstance(result, Sequence) or isinstance(result, (str, bytes)):
        raise ModelIncompatibleError(
            "The embedding gateway returned invalid retrieval query vectors."
        )
    if len(result) != len(queries):
        raise ModelIncompatibleError(
            "The embedding gateway did not return one vector per retrieval query."
        )
    vectors: list[tuple[float, ...]] = []
    for raw_vector in result:
        if not isinstance(raw_vector, Sequence) or isinstance(
            raw_vector,
            (str, bytes),
        ):
            raise ModelIncompatibleError(
                "The embedding gateway returned an invalid retrieval query vector."
            )
        vector = tuple(float(value) for value in raw_vector)
        if len(vector) != dimensions:
            raise ModelIncompatibleError(
                "The retrieval query embedding dimensions do not match the active index."
            )
        vectors.append(vector)
    return tuple(vectors)


def _evidence_identity(evidence: Sequence[SearchResult]) -> str:
    canonical = json.dumps(
        [
            {
                "path": result.file_path,
                "chunk_id": result.chunk_id,
                "text": [
                    section.text for section in result.matched_sections
                ],
            }
            for result in evidence
        ],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _validate_session_identity(
    session_id: str | None,
    conversation_id: str | None,
) -> str:
    for identity in (session_id, conversation_id):
        if identity is not None and (
            not isinstance(identity, str) or not identity.strip()
        ):
            raise ValueError(
                "A nonempty conversation or session identity is required."
            )
    if session_id is not None and conversation_id is not None:
        if session_id != conversation_id:
            raise ValueError(
                "session_id and conversation_id must identify the same conversation."
            )
        conversation_id = None
    selected = session_id or conversation_id or "default"
    return selected.strip()


def answer_question(
    question: str,
    evidence: EvidenceInput,
    model: str = DEFAULT_GENERATION_MODEL,
    *,
    gateway: GenerationGateway | None = None,
    rerank: bool = False,
    rerank_top_k: int = DEFAULT_RERANK_TOP_K,
    max_context_retries: int = MAX_CONTEXT_RETRIES,
) -> CitedAnswer:
    return _run_sync(
        answer_question_async(
            question,
            evidence,
            model,
            gateway=gateway,
            rerank=rerank,
            rerank_top_k=rerank_top_k,
            max_context_retries=max_context_retries,
        )
    )


async def answer_question_async(
    question: str,
    evidence: EvidenceInput,
    model: str = DEFAULT_GENERATION_MODEL,
    *,
    gateway: GenerationGateway | None = None,
    rerank: bool = False,
    rerank_top_k: int = DEFAULT_RERANK_TOP_K,
    max_context_retries: int = MAX_CONTEXT_RETRIES,
) -> CitedAnswer:
    checked_question = _validate_question(question)
    checked_model = _validate_model(model)
    selected = _bounded_evidence(
        _coerce_evidence(evidence),
        limit=DEFAULT_EVIDENCE_LIMIT,
        max_sections_per_file=DEFAULT_MAX_SECTIONS_PER_FILE,
    )
    if not selected:
        return CitedAnswer(question=checked_question, text=NOT_FOUND_ANSWER)

    owns_gateway = gateway is None
    selected_gateway: GenerationGateway = gateway or OllamaGateway()
    try:
        return await _generate_answer_async(
            checked_question,
            selected,
            model=checked_model,
            gateway=selected_gateway,
            rerank=rerank,
            rerank_top_k=rerank_top_k,
            max_context_retries=max_context_retries,
        )
    finally:
        if owns_gateway:
            await _close_gateway(selected_gateway)


async def _generate_answer_stream_async(
    question: str,
    evidence: Sequence[SearchResult],
    *,
    model: str,
    gateway: GenerationGateway,
    markdown_context: str,
    rerank: bool,
    rerank_top_k: int,
    max_context_retries: int,
    history: ChatHistory = (),
    cancellation_check: Callable[[], bool] | None = None,
) -> AsyncIterator[tuple[str, CitedAnswer | None]]:
    if rerank_top_k <= 0:
        raise ValueError("The rerank top-k must be positive.")
    if max_context_retries < 0:
        raise ValueError("The context retry limit cannot be negative.")

    selected = tuple(evidence)
    if cancellation_check is not None and cancellation_check():
        raise asyncio.CancelledError
    if not selected:
        yield "", CitedAnswer(
            question=question,
            text=NOT_FOUND_ANSWER,
            evidence=(),
        )
        return

    profile = await _generation_profile(gateway, model)
    if rerank:
        selected = await _rerank_evidence_async(
            question,
            selected[:rerank_top_k],
            selected[rerank_top_k:],
            model=model,
            gateway=gateway,
            markdown_context=markdown_context,
            profile=profile,
            max_context_retries=max_context_retries,
        )

    remaining = selected
    retries = 0
    while True:
        bounded = _fit_evidence_to_profile(
            question,
            remaining,
            history=history,
            markdown_context=markdown_context,
            profile=profile,
        )
        messages = _answer_messages(
            question,
            bounded,
            history,
            markdown_context=markdown_context,
        )
        chat_stream = getattr(gateway, "chat_stream", None)
        if not callable(chat_stream):
            response = await gateway.chat(
                model,
                messages=messages,
                schema=answer_schema(),
            )
            yield response, parse_cited_answer(question, response, bounded)
            return

        accumulated = ""
        emitted_answer = ""
        mode: str | None = None
        try:
            async for fragment in chat_stream(
                model,
                messages=messages,
                schema=answer_schema(),
            ):
                if cancellation_check is not None and cancellation_check():
                    raise asyncio.CancelledError
                if not isinstance(fragment, str) or not fragment:
                    continue
                accumulated += fragment
                if mode is None:
                    stripped = accumulated.lstrip()
                    if stripped:
                        mode = "json" if stripped.startswith("{") else "plain"
                if mode == "plain":
                    yield fragment, None
                    emitted_answer += fragment
                    continue

                match = re.search(r'"answer"\s*:\s*"', accumulated)
                if match is None:
                    continue
                raw_answer = accumulated[match.end():]
                closing = _find_unescaped_quote(raw_answer)
                candidate = raw_answer if closing is None else raw_answer[:closing]
                try:
                    decoded = json.loads('"' + candidate + '"')
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if isinstance(decoded, str) and decoded.startswith(emitted_answer):
                    delta = decoded[len(emitted_answer):]
                    if delta:
                        emitted_answer = decoded
                        yield delta, None

            if mode == "json" or (mode is None and accumulated.lstrip().startswith("{")):
                answer = parse_cited_answer(question, accumulated, bounded)
            else:
                answer = CitedAnswer(
                    question=question,
                    text=accumulated,
                    evidence=bounded,
                )
            yield "", answer
            return
        except GenerationContextRejectedError:
            if len(bounded) <= 1 or retries >= max_context_retries:
                raise
            remaining = bounded[: max(1, len(bounded) // 2)]
            retries += 1


def _find_unescaped_quote(text: str) -> int | None:
    escaped = False
    for index, char in enumerate(text):
        if char == '"' and not escaped:
            return index
        if char == "\\" and not escaped:
            escaped = True
        else:
            escaped = False
    return None


async def _generate_answer_async(
    question: str,
    evidence: Sequence[SearchResult],
    *,
    model: str,
    gateway: GenerationGateway,
    markdown_context: str = "",
    rerank: bool,
    rerank_top_k: int,
    max_context_retries: int,
    history: ChatHistory = (),
    cancellation_check: Callable[[], bool] | None = None,
) -> CitedAnswer:
    if rerank_top_k <= 0:
        raise ValueError("The rerank top-k must be positive.")
    if max_context_retries < 0:
        raise ValueError("The context retry limit cannot be negative.")

    selected = tuple(evidence)
    if cancellation_check is not None and cancellation_check():
        raise asyncio.CancelledError
    if not selected:
        return CitedAnswer(question=question, text=NOT_FOUND_ANSWER)

    profile = await _generation_profile(gateway, model)
    if rerank:
        selected = await _rerank_evidence_async(
            question,
            selected[:rerank_top_k],
            selected[rerank_top_k:],
            model=model,
            gateway=gateway,
            markdown_context=markdown_context,
            profile=profile,
            max_context_retries=max_context_retries,
        )

    remaining = selected
    retries = 0
    while True:
        bounded = _fit_evidence_to_profile(
            question,
            remaining,
            history=history,
            markdown_context=markdown_context,
            profile=profile,
        )
        messages = _answer_messages(
            question,
            bounded,
            history,
            markdown_context=markdown_context,
        )
        try:
            response = await gateway.chat(
                model,
                messages=messages,
                schema=answer_schema(),
            )
            if cancellation_check is not None and cancellation_check():
                raise asyncio.CancelledError
            return parse_cited_answer(question, response, bounded)
        except GenerationContextRejectedError:
            if len(bounded) <= 1 or retries >= max_context_retries:
                raise
            remaining = bounded[: max(1, len(bounded) // 2)]
            retries += 1


async def _rerank_evidence_async(
    question: str,
    candidates: Sequence[SearchResult],
    remainder: Sequence[SearchResult],
    *,
    model: str,
    gateway: GenerationGateway,
    markdown_context: str,
    profile: ModelProfile | None,
    max_context_retries: int,
) -> tuple[SearchResult, ...]:
    bounded = tuple(candidates)
    retries = 0
    while bounded:
        messages = _rerank_messages(
            question,
            bounded,
            markdown_context=markdown_context,
        )
        if profile is not None and not _request_fits(
            messages,
            _rerank_schema(),
            profile.context_length,
        ):
            if len(bounded) > 1:
                bounded = bounded[: max(1, len(bounded) // 2)]
                continue
        try:
            response = await gateway.chat(
                model,
                messages=messages,
                schema=_rerank_schema(),
            )
        except GenerationContextRejectedError:
            if len(bounded) <= 1 or retries >= max_context_retries:
                raise
            bounded = bounded[: max(1, len(bounded) // 2)]
            retries += 1
            continue
        return _parse_rerank(response, bounded, remainder)
    return tuple(remainder)


def _parse_rerank(
    response: str,
    candidates: Sequence[SearchResult],
    remainder: Sequence[SearchResult],
) -> tuple[SearchResult, ...]:
    try:
        payload = json.loads(response)
    except (TypeError, json.JSONDecodeError) as exc:
        raise StructuredOutputError(
            "Evidence reranking returned malformed structured data."
        ) from exc
    if not isinstance(payload, Mapping):
        raise StructuredOutputError(
            "Evidence reranking returned invalid structured data."
        )
    ranked_ids = payload.get("ranked_chunk_ids")
    if not isinstance(ranked_ids, list) or not ranked_ids:
        raise StructuredOutputError(
            "Evidence reranking returned no ranked chunk IDs."
        )
    if any(not isinstance(item, str) for item in ranked_ids):
        raise StructuredOutputError(
            "Evidence reranking returned invalid chunk IDs."
        )
    if len(set(cast(list[str], ranked_ids))) != len(ranked_ids):
        raise StructuredOutputError(
            "Evidence reranking returned duplicate chunk IDs."
        )

    available = {result.chunk_id: result for result in candidates}
    unknown = set(cast(list[str], ranked_ids)) - set(available)
    if unknown:
        raise StructuredOutputError(
            f"Evidence reranking returned unknown chunk IDs: {sorted(unknown)!r}."
        )
    selected = [available[chunk_id] for chunk_id in cast(list[str], ranked_ids)]
    selected_ids = set(cast(list[str], ranked_ids))
    selected.extend(
        result
        for result in (*candidates, *remainder)
        if result.chunk_id not in selected_ids
    )
    return tuple(selected)


def _bounded_evidence(
    evidence: Sequence[SearchResult],
    *,
    limit: int,
    max_sections_per_file: int,
) -> tuple[SearchResult, ...]:
    if limit == 0:
        return ()
    expanded = sorted(
        _expand_evidence(evidence),
        key=lambda item: (
            item.rank,
            -item.score,
            item.file_path.casefold(),
            item.chunk_id,
        ),
    )
    selected: list[SearchResult] = []
    seen_chunks: set[str] = set()
    file_counts: dict[str, int] = {}
    for result in expanded:
        if result.chunk_id in seen_chunks:
            continue
        path_key = result.file_path.casefold()
        if file_counts.get(path_key, 0) >= max_sections_per_file:
            continue
        selected.append(result)
        seen_chunks.add(result.chunk_id)
        file_counts[path_key] = file_counts.get(path_key, 0) + 1
        if len(selected) >= limit:
            break
    return tuple(selected)


def _expand_evidence(
    evidence: Sequence[SearchResult],
) -> tuple[SearchResult, ...]:
    expanded: list[SearchResult] = []
    for result in evidence:
        if result.stale:
            continue
        for section in result.matched_sections:
            expanded.append(
                make_result(
                    {
                        "file_path": result.file_path,
                        "stable_id": section.chunk_id,
                        "page_number": section.page_number,
                        "heading": section.heading,
                        "section": section.section,
                        "text": section.text,
                        "tags": result.tags,
                        "mime_type": result.mime_type,
                    },
                    source=result.source,
                    rank=result.rank,
                    score=result.score,
                )
            )
    return tuple(expanded)


def _coerce_evidence(
    evidence: EvidenceInput,
) -> tuple[SearchResult, ...]:
    if isinstance(evidence, (SearchResult, SearchCitation)):
        items: tuple[SearchResult | SearchCitation, ...] = (evidence,)
    elif isinstance(evidence, (str, bytes)):
        raise TypeError("Evidence must contain search results or citations.")
    else:
        items = tuple(evidence)

    results: list[SearchResult] = []
    for item in items:
        if isinstance(item, SearchResult):
            results.append(item)
            continue
        if isinstance(item, SearchCitation):
            results.append(
                make_result(
                    {
                        "file_path": item.path,
                        "stable_id": item.chunk_id,
                        "page_number": item.page_number,
                        "heading": item.heading,
                        "section": item.section,
                        "text": item.text,
                    },
                    source="hybrid",
                )
            )
            continue
        raise TypeError("Evidence must contain search results or citations.")
    return tuple(results)


def _fit_evidence_to_profile(
    question: str,
    evidence: Sequence[SearchResult],
    *,
    history: ChatHistory,
    markdown_context: str,
    profile: ModelProfile | None,
) -> tuple[SearchResult, ...]:
    selected = tuple(evidence)
    context_length = (
        profile.context_length
        if profile is not None
        else DEFAULT_UNPROFILED_CONTEXT_LENGTH
    )
    while len(selected) > 1:
        messages = _answer_messages(
            question,
            selected,
            history,
            markdown_context=markdown_context,
        )
        if _request_fits(messages, answer_schema(), context_length):
            return selected
        selected = selected[: max(1, len(selected) // 2)]
    return selected


def _answer_messages(
    question: str,
    evidence: Sequence[SearchResult],
    history: ChatHistory,
    *,
    markdown_context: str = "",
) -> tuple[dict[str, str], ...]:
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "Answer only from the indexed evidence below. Return JSON only in the "
                "requested schema. Every factual claim must include chunk_ids and an "
                "evidence_quote equal to the complete full text of every cited chunk, "
                "including all punctuation and symbols. Never quote only a sentence or "
                "fragment. The claim and every cited chunk must match that quote after "
                "only Unicode NFKC, case, and whitespace normalization; never omit "
                "hedges, uncertainty, or negation. Put any visible source marker "
                "immediately after the matching claim. Never invent chunk IDs, quotes, "
                "citations, facts, or files. "
                f'If the evidence is insufficient, answer "{NOT_FOUND_ANSWER}" with no '
                "claims."
            ),
        }
    ]
    for previous_question, previous_answer in history[-4:]:
        messages.extend(
            (
                {"role": "user", "content": previous_question},
                {"role": "assistant", "content": previous_answer},
            )
        )
    messages.append(
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Markdown context:\n{markdown_context or '(none)'}\n\n"
                f"Indexed evidence:\n{json.dumps(_evidence_payload(evidence), ensure_ascii=False)}"
            ),
        }
    )
    return tuple(messages)


def _rerank_messages(
    question: str,
    evidence: Sequence[SearchResult],
    *,
    markdown_context: str = "",
) -> tuple[dict[str, str], ...]:
    return (
        {
            "role": "system",
            "content": (
                "Rank only the supplied indexed evidence for the question. Return JSON "
                "only. Include each selected chunk ID exactly once in descending relevance."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Markdown context:\n{markdown_context or '(none)'}\n\n"
                f"Candidates:\n{json.dumps(_evidence_payload(evidence), ensure_ascii=False)}"
            ),
        },
    )


def _evidence_payload(
    evidence: Sequence[SearchResult],
) -> tuple[dict[str, Any], ...]:
    payload: list[dict[str, Any]] = []
    for result in evidence:
        for citation in result.citations:
            payload.append(
                {
                    "chunk_id": citation.chunk_id,
                    "citation_id": citation.citation_id,
                    "path": citation.path,
                    "page_number": citation.page_number,
                    "heading": citation.heading,
                    "section": citation.section,
                    "text": citation.text,
                }
            )
    return tuple(payload)


def _rerank_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["ranked_chunk_ids"],
        "properties": {
            "ranked_chunk_ids": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "minLength": 1},
            }
        },
    }


def _request_fits(
    messages: Sequence[Mapping[str, Any]],
    schema: JsonMapping,
    context_length: int,
) -> bool:
    prompt = json.dumps(
        {"messages": messages, "format": schema},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return _estimate_token_count(prompt) + OUTPUT_TOKEN_RESERVE <= context_length


async def _generation_profile(
    gateway: GenerationGateway,
    model: str,
) -> ModelProfile | None:
    profile_method = getattr(gateway, "model_profile", None)
    if not callable(profile_method):
        return None
    raw_profile = await profile_method("generation", model)
    try:
        profile = (
            raw_profile
            if isinstance(raw_profile, ModelProfile)
            else ModelProfile.model_validate(raw_profile)
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise ModelIncompatibleError(
            f"Model {model} returned an invalid generation profile."
        ) from exc
    if profile.role != "generation" or profile.model != model:
        raise ModelIncompatibleError(
            f"Model {model} returned a mismatched generation profile."
        )
    if "completion" not in profile.capabilities:
        raise ModelIncompatibleError(
            f"Model {model} lacks the required completion capability."
        )
    return profile


def _validate_question(question: str) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("A nonempty question is required.")
    checked = " ".join(question.split())
    if len(checked) > MAX_QUESTION_LENGTH:
        raise ValueError(
            f"A question must not exceed {MAX_QUESTION_LENGTH} characters."
        )
    return checked


def _validate_model(model: str) -> str:
    if not isinstance(model, str) or not model.strip():
        raise ValueError("A local generation model name is required.")
    return model.strip()


def _estimate_token_count(text: str) -> int:
    ascii_bytes = sum(character.isascii() for character in text)
    non_ascii_bytes = len(text.encode("utf-8")) - ascii_bytes
    return max(
        1,
        (ascii_bytes + ASCII_BYTES_PER_TOKEN - 1) // ASCII_BYTES_PER_TOKEN
        + non_ascii_bytes,
    )


async def _close_gateway(gateway: GenerationGateway) -> None:
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


__all__ = (
    "CitedAnswer",
    "DEFAULT_GENERATION_MODEL",
    "EvidenceInput",
    "GenerationGateway",
    "NOT_FOUND_ANSWER",
    "RagService",
    "SearchGateway",
    "UncitedAnswerError",
    "answer_question",
    "answer_question_async",
    "execute_search_plan",
    "execute_search_plan_async",
)
