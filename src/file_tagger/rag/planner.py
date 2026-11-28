import asyncio
from collections.abc import Coroutine, Mapping, Sequence
import json
from typing import Annotated, Any, Protocol, TypeAlias, TypeVar
import unicodedata

from pydantic import AliasChoices, ConfigDict, Field, ValidationError, field_validator

from ..adapters.gemini.gateway import GEMINI_GENERATION_MODEL
from ..adapters.ollama.gateway import OllamaGateway
from ..domain.contracts import DomainRecord, ModelProfile
from ..domain.errors import (
    GenerationContextRejectedError,
    GeminiUnavailableError,
    ModelIncompatibleError,
    ModelMissingError,
    StructuredOutputError,
)
from .context import NormalizedMarkdownContext, normalize_markdown_context


DEFAULT_PLANNER_MODEL = "inkling-2-mini"
DEFAULT_FALLBACK_MODEL = "gemma4:e4b"
DEFAULT_UNPROFILED_CONTEXT_LENGTH = 32_768
MAX_PLAN_RESPONSE_BYTES = 16_384
MAX_QUESTION_LENGTH = 1_024
MAX_INTENT_ITEMS = 8
MAX_EXACT_TERM_ITEMS = 16
MAX_PARAPHRASE_ITEMS = 8
MAX_AUXILIARY_ITEMS = 16
MAX_EVIDENCE_ITEMS = 8
MAX_NAME_LENGTH = 64
MAX_TERM_LENGTH = 128
MAX_QUERY_LENGTH = 256
MAX_AUXILIARY_LENGTH = 160
ASCII_BYTES_PER_TOKEN = 4
OUTPUT_TOKEN_RESERVE = 4_096

PlanName: TypeAlias = Annotated[str, Field(min_length=1, max_length=MAX_NAME_LENGTH)]
PlanTerm: TypeAlias = Annotated[str, Field(min_length=1, max_length=MAX_TERM_LENGTH)]
PlanQuery: TypeAlias = Annotated[str, Field(min_length=1, max_length=MAX_QUERY_LENGTH)]
PlanConstraint: TypeAlias = Annotated[
    str,
    Field(min_length=1, max_length=MAX_AUXILIARY_LENGTH),
]

T = TypeVar("T")
JsonMapping: TypeAlias = Mapping[str, Any]


class PlannerGateway(Protocol):
    async def chat(
        self,
        model: str,
        *,
        messages: Sequence[Mapping[str, Any]],
        schema: JsonMapping | None = None,
        keep_alive: float | str | None = "5m",
    ) -> str: ...


class SearchPlan(DomainRecord):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_by_name=True)

    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)
    intent_names: tuple[PlanName, ...] = Field(
        min_length=1,
        max_length=MAX_INTENT_ITEMS,
        validation_alias=AliasChoices("intent_names", "intents"),
    )
    exact_terms: tuple[PlanTerm, ...] = Field(
        default=(),
        max_length=MAX_EXACT_TERM_ITEMS,
        validation_alias=AliasChoices("exact_terms", "exact_entities"),
    )
    paraphrase_queries: tuple[PlanQuery, ...] = Field(
        min_length=1,
        max_length=MAX_PARAPHRASE_ITEMS,
        validation_alias=AliasChoices("paraphrase_queries", "subqueries"),
    )
    filters: tuple[PlanConstraint, ...] = Field(
        default=(),
        max_length=MAX_AUXILIARY_ITEMS,
    )
    exclusions: tuple[PlanConstraint, ...] = Field(
        default=(),
        max_length=MAX_AUXILIARY_ITEMS,
    )
    normalized_constraints: tuple[PlanConstraint, ...] = Field(
        default=(),
        max_length=MAX_AUXILIARY_ITEMS,
    )
    evidence_categories: tuple[PlanName, ...] = Field(
        min_length=1,
        max_length=MAX_EVIDENCE_ITEMS,
    )

    @field_validator("question", mode="before")
    @classmethod
    def normalize_question(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        return " ".join(unicodedata.normalize("NFKC", value).split())

    @field_validator(
        "intent_names",
        "exact_terms",
        "paraphrase_queries",
        "filters",
        "exclusions",
        "normalized_constraints",
        "evidence_categories",
        mode="before",
    )
    @classmethod
    def normalize_terms(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            return value
        checked = tuple(item for item in value if isinstance(item, str))
        if len(checked) != len(value):
            return value
        return _unique_nonempty(checked)

    @field_validator(
        "intent_names",
        "normalized_constraints",
        "evidence_categories",
    )
    @classmethod
    def canonicalize_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(term.casefold() for term in value)

    @property
    def intents(self) -> tuple[str, ...]:
        return self.intent_names

    @property
    def exact_entities(self) -> tuple[str, ...]:
        return self.exact_terms

    @property
    def subqueries(self) -> tuple[str, ...]:
        return self.paraphrase_queries


def build_search_plan(
    question: str,
    markdown_context: str = "",
    *,
    model: str = DEFAULT_PLANNER_MODEL,
    fallback_model: str | None = DEFAULT_FALLBACK_MODEL,
    gateway: PlannerGateway | None = None,
) -> SearchPlan:
    return _run_sync(
        build_search_plan_async(
            question,
            markdown_context,
            model=model,
            fallback_model=fallback_model,
            gateway=gateway,
        )
    )


async def build_search_plan_async(
    question: str,
    markdown_context: str = "",
    *,
    model: str = DEFAULT_PLANNER_MODEL,
    fallback_model: str | None = DEFAULT_FALLBACK_MODEL,
    gateway: PlannerGateway | None = None,
) -> SearchPlan:
    checked_question, checked_model, checked_fallback = _validate_request(
        question,
        model,
        fallback_model,
    )
    if (
        gateway is None
        or isinstance(gateway, OllamaGateway)
    ) and (
        checked_model == GEMINI_GENERATION_MODEL
        or (
            checked_fallback is not None
            and checked_fallback == GEMINI_GENERATION_MODEL
        )
    ):
        raise GeminiUnavailableError(
            "The Gemini generation gateway is unavailable."
        )
    context = normalize_markdown_context(markdown_context)
    owns_gateway = gateway is None
    selected_gateway: PlannerGateway = gateway or OllamaGateway()
    messages = _planning_messages(checked_question, context)
    models: tuple[str, ...] = (checked_model,)
    if _canonical_model_name(checked_model) == DEFAULT_PLANNER_MODEL and (
        checked_fallback is not None
    ):
        if checked_fallback != checked_model:
            models += (checked_fallback,)

    try:
        for index, current_model in enumerate(models):
            try:
                profile = await _generation_profile(selected_gateway, current_model)
                _enforce_prompt_budget(
                    current_model,
                    messages,
                    SearchPlan.model_json_schema(),
                    profile,
                )
                response = await selected_gateway.chat(
                    current_model,
                    messages=messages,
                    schema=SearchPlan.model_json_schema(),
                )
                return _parse_plan(response, checked_question, context)
            except (ModelMissingError, ModelIncompatibleError):
                if index + 1 < len(models):
                    continue
                raise
        raise ModelMissingError("No local search planning model is available.")
    finally:
        if owns_gateway:
            await _close_gateway(selected_gateway)


def _validate_request(
    question: str,
    model: str,
    fallback_model: str | None,
) -> tuple[str, str, str | None]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("A nonempty search question is required.")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("A search planning model name is required.")
    checked_question = " ".join(unicodedata.normalize("NFKC", question).split())
    if len(checked_question) > MAX_QUESTION_LENGTH:
        raise ValueError(
            f"A search question must not exceed {MAX_QUESTION_LENGTH} characters."
        )
    checked_model = model.strip()
    _reject_cloud_model(checked_model)
    if fallback_model is None:
        return checked_question, checked_model, None
    if not isinstance(fallback_model, str) or not fallback_model.strip():
        raise ValueError("The Gemma search planning fallback must be a model name.")
    checked_fallback = fallback_model.strip()
    _reject_cloud_model(checked_fallback)
    if not checked_fallback.casefold().startswith("gemma4:"):
        raise ValueError("The search planning fallback must be a Gemma 4 model.")
    return checked_question, checked_model, checked_fallback


def _reject_cloud_model(model: str) -> None:
    normalized = model.casefold()
    if normalized == "cloud" or any(
        marker in normalized
        for marker in (":cloud", "-cloud", "cloud:", ".cloud")
    ):
        raise ModelIncompatibleError(
            "Cloud-hosted model variants are not supported."
        )


def _canonical_model_name(model: str) -> str:
    normalized = unicodedata.normalize("NFKC", model).strip().casefold()
    if normalized.endswith(":latest"):
        return normalized.removesuffix(":latest")
    return normalized


def _planning_messages(
    question: str,
    context: NormalizedMarkdownContext,
) -> tuple[dict[str, str], dict[str, str]]:
    context_text = context.normalized_text or "(no Markdown context provided)"
    section_names = ", ".join(context.section_names) or "none"
    entities = ", ".join(context.entities) or "none"
    constraints = ", ".join(context.constraints) or "none"
    return (
        {
            "role": "system",
            "content": (
                "Create a typed retrieval plan from the question and Markdown context. "
                "Return JSON only. Preserve exact document terms, create useful semantic "
                "paraphrases, identify filters and exclusions, and include all evidence "
                "categories needed to answer the question. Never invent facts or an "
                "empty fallback plan."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Normalized Markdown context:\n{context_text}\n\n"
                f"Sections: {section_names}\n"
                f"Entities: {entities}\n"
                f"Constraints: {constraints}"
            ),
        },
    )


def _parse_plan(
    response: str,
    question: str,
    context: NormalizedMarkdownContext,
) -> SearchPlan:
    if not isinstance(response, str):
        raise StructuredOutputError(
            "search plan generation returned invalid structured data."
        )
    if len(response.encode("utf-8")) > MAX_PLAN_RESPONSE_BYTES:
        raise StructuredOutputError(
            "search plan generation returned oversized structured data."
        )
    try:
        payload = json.loads(response)
    except (TypeError, json.JSONDecodeError) as exc:
        raise StructuredOutputError(
            "search plan generation returned malformed JSON."
        ) from exc
    if not isinstance(payload, dict):
        raise StructuredOutputError(
            "search plan generation returned invalid structured data."
        )

    payload["question"] = question
    try:
        plan = SearchPlan.model_validate(payload)
    except ValidationError as exc:
        raise StructuredOutputError(
            "search plan generation returned invalid structured data."
        ) from exc
    try:
        return SearchPlan.model_validate(
            {
                **plan.model_dump(),
                "exact_terms": _merge_terms(
                    context.entities,
                    plan.exact_terms,
                    max_items=MAX_EXACT_TERM_ITEMS,
                    max_length=MAX_TERM_LENGTH,
                ),
                "normalized_constraints": _merge_terms(
                    context.constraints,
                    plan.normalized_constraints,
                    max_items=MAX_AUXILIARY_ITEMS,
                    max_length=MAX_AUXILIARY_LENGTH,
                ),
            }
        )
    except ValidationError as exc:
        raise StructuredOutputError(
            "search plan generation exceeded structured data bounds."
        ) from exc


async def _generation_profile(
    gateway: PlannerGateway,
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


def _enforce_prompt_budget(
    model: str,
    messages: Sequence[Mapping[str, Any]],
    schema: JsonMapping,
    profile: ModelProfile | None,
) -> None:
    context_length = (
        profile.context_length
        if profile is not None
        else DEFAULT_UNPROFILED_CONTEXT_LENGTH
    )
    prompt_text = json.dumps(
        {"messages": messages, "format": schema},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    estimated_prompt_tokens = _estimate_token_count(prompt_text)
    required_context_tokens = estimated_prompt_tokens + OUTPUT_TOKEN_RESERVE
    if required_context_tokens > context_length:
        raise GenerationContextRejectedError(
            f"Search planning prompt for {model} exceeds its context budget "
            f"({required_context_tokens} required, {context_length} available)."
        )


def _estimate_token_count(text: str) -> int:
    ascii_bytes = sum(character.isascii() for character in text)
    non_ascii_bytes = len(text.encode("utf-8")) - ascii_bytes
    return max(
        1,
        (ascii_bytes + ASCII_BYTES_PER_TOKEN - 1) // ASCII_BYTES_PER_TOKEN
        + non_ascii_bytes,
    )


def _merge_terms(
    first: Sequence[str],
    second: Sequence[str],
    *,
    max_items: int | None = None,
    max_length: int | None = None,
) -> tuple[str, ...]:
    checked_first = tuple(
        term
        for term in _unique_nonempty(first)
        if max_length is None or len(term) <= max_length
    )
    first_keys = {term.casefold() for term in checked_first}
    checked_second = tuple(
        term
        for term in _unique_nonempty(second)
        if term.casefold() not in first_keys
        and (max_length is None or len(term) <= max_length)
    )
    if max_items is None:
        return (*checked_first, *checked_second)
    if max_items <= 0:
        return ()
    if not checked_first:
        return checked_second[:max_items]
    if not checked_second:
        return checked_first[:max_items]

    first_quota = min(len(checked_first), (max_items + 1) // 2)
    second_quota = min(len(checked_second), max_items - first_quota)
    unused = max_items - first_quota - second_quota
    if unused:
        extra_second = min(len(checked_second) - second_quota, unused)
        second_quota += extra_second
        unused -= extra_second
    if unused:
        extra_first = min(len(checked_first) - first_quota, unused)
        first_quota += extra_first

    selected_first = checked_first[:first_quota]
    selected_second = checked_second[:second_quota]
    interleaved: list[str] = []
    for index in range(max(first_quota, second_quota)):
        if index < first_quota:
            interleaved.append(selected_first[index])
        if index < second_quota:
            interleaved.append(selected_second[index])
    return tuple(interleaved[:max_items])


def _unique_nonempty(values: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(unicodedata.normalize("NFKC", value).split())
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return tuple(result)


async def _close_gateway(gateway: PlannerGateway) -> None:
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
