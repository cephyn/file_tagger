import asyncio
from collections.abc import (
    Awaitable,
    Coroutine,
    Mapping,
    Sequence,
)
import json
from typing import Any, Callable, Protocol, TypeAlias, TypeVar, cast

from pydantic import ValidationError

from ..adapters.ollama.gateway import OllamaGateway
from ..domain.contracts import ModelProfile
from ..domain.errors import (
    GenerationContextRejectedError,
    ModelIncompatibleError,
    StructuredOutputError,
    UncitedAnswerError,
)
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
        *,
        limit: int = 10,
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


class RagService:
    def __init__(
        self,
        search: SearchGateway | None = None,
        gateway: GenerationGateway | None = None,
        planner: Planner = build_search_plan_async,
        model: str = DEFAULT_GENERATION_MODEL,
        evidence_limit: int = DEFAULT_EVIDENCE_LIMIT,
        per_query_limit: int = DEFAULT_PER_QUERY_LIMIT,
        max_sections_per_file: int = DEFAULT_MAX_SECTIONS_PER_FILE,
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
        self._evidence_cache: dict[tuple[str, str], tuple[SearchResult, ...]] = {}
        self._history: dict[tuple[str, str], ChatHistory] = {}

    @property
    def evidence(self) -> tuple[SearchResult, ...]:
        return tuple(
            result
            for cached in self._evidence_cache.values()
            for result in cached
        )

    def answer_question(
        self,
        question: str,
        markdown_context: str | EvidenceInput = "",
        evidence: EvidenceInput | None = None,
        *,
        rerank: bool = False,
        rerank_top_k: int = DEFAULT_RERANK_TOP_K,
        max_context_retries: int = MAX_CONTEXT_RETRIES,
    ) -> CitedAnswer:
        return _run_sync(
            self.answer_question_async(
                question,
                markdown_context,
                evidence,
                rerank=rerank,
                rerank_top_k=rerank_top_k,
                max_context_retries=max_context_retries,
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
        key = (checked_question, checked_context)

        if evidence is not None:
            selected = _bounded_evidence(
                _coerce_evidence(evidence),
                limit=self.evidence_limit,
                max_sections_per_file=self.max_sections_per_file,
            )
            self._evidence_cache[key] = selected
        elif key in self._evidence_cache:
            selected = self._evidence_cache[key]
        else:
            if self._search is None:
                raise ValueError(
                    "Search evidence or a HybridSearch service is required."
                )
            planned = self._planner(
                checked_question,
                checked_context,
                gateway=self._gateway,
            )
            plan = (
                planned
                if isinstance(planned, SearchPlan)
                else await planned
            )
            selected = execute_search_plan(
                plan,
                self._search,
                per_query_limit=self.per_query_limit,
                limit=self.evidence_limit,
                max_sections_per_file=self.max_sections_per_file,
            )
            self._evidence_cache[key] = selected

        previous = self._history.get(key, ())
        answer = await _generate_answer_async(
            checked_question,
            selected,
            model=self.model,
            gateway=self._gateway,
            markdown_context=checked_context,
            rerank=rerank,
            rerank_top_k=rerank_top_k,
            max_context_retries=max_context_retries,
            history=previous,
        )
        self._history[key] = (
            *previous[-4:],
            (checked_question, answer.text),
        )
        return answer

    async def answer_plan_async(
        self,
        plan: SearchPlan,
        markdown_context: str = "",
        *,
        rerank: bool = False,
    ) -> CitedAnswer:
        if not isinstance(plan, SearchPlan):
            raise TypeError("plan must be a SearchPlan.")
        if self._search is None:
            raise ValueError("A HybridSearch service is required.")
        evidence = execute_search_plan(
            plan,
            self._search,
            per_query_limit=self.per_query_limit,
            limit=self.evidence_limit,
            max_sections_per_file=self.max_sections_per_file,
        )
        return await self.answer_question_async(
            plan.question,
            markdown_context,
            evidence,
            rerank=rerank,
        )

    def answer_plan(
        self,
        plan: SearchPlan,
        markdown_context: str = "",
        *,
        rerank: bool = False,
    ) -> CitedAnswer:
        return _run_sync(
            self.answer_plan_async(
                plan,
                markdown_context,
                rerank=rerank,
            )
        )


def execute_search_plan(
    plan: SearchPlan,
    search: SearchGateway,
    *,
    per_query_limit: int = DEFAULT_PER_QUERY_LIMIT,
    limit: int = DEFAULT_EVIDENCE_LIMIT,
    max_sections_per_file: int = DEFAULT_MAX_SECTIONS_PER_FILE,
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

    queries = tuple(dict.fromkeys(plan.paraphrase_queries))
    candidates: list[SearchResult] = []
    for query in queries:
        candidates.extend(search.search(query, limit=per_query_limit))
    return _bounded_evidence(
        candidates,
        limit=limit,
        max_sections_per_file=max_sections_per_file,
    )


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
) -> CitedAnswer:
    if rerank_top_k <= 0:
        raise ValueError("The rerank top-k must be positive.")
    if max_context_retries < 0:
        raise ValueError("The context retry limit cannot be negative.")

    selected = tuple(evidence)
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
                        "text": section.snippet,
                        "tags": result.tags,
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
                        "text": item.excerpt,
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
                "requested schema. Every factual claim must include one or more chunk_ids "
                "from the evidence. Never invent chunk IDs, citations, facts, or files. "
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
                    "text": citation.excerpt,
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
)
