import asyncio
from collections.abc import Coroutine, Mapping, Sequence
import json
from typing import Any, Protocol, TypeAlias, TypeVar
import unicodedata

from pydantic import AliasChoices, ConfigDict, Field, ValidationError, field_validator

from ..adapters.ollama.gateway import OllamaGateway
from ..domain.contracts import DomainRecord
from ..domain.errors import ModelIncompatibleError, ModelMissingError
from .context import NormalizedMarkdownContext, normalize_markdown_context


DEFAULT_PLANNER_MODEL = "inkling-2-mini"
DEFAULT_FALLBACK_MODEL = "gemma4:e4b"

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

    question: str = Field(min_length=1)
    intent_names: tuple[str, ...] = Field(
        min_length=1,
        validation_alias=AliasChoices("intent_names", "intents"),
    )
    exact_terms: tuple[str, ...] = Field(
        default=(),
        validation_alias=AliasChoices("exact_terms", "exact_entities"),
    )
    paraphrase_queries: tuple[str, ...] = Field(
        min_length=1,
        validation_alias=AliasChoices("paraphrase_queries", "subqueries"),
    )
    filters: tuple[str, ...] = ()
    exclusions: tuple[str, ...] = ()
    normalized_constraints: tuple[str, ...] = ()
    evidence_categories: tuple[str, ...] = Field(min_length=1)

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
    context = normalize_markdown_context(markdown_context)
    owns_gateway = gateway is None
    selected_gateway: PlannerGateway = gateway or OllamaGateway()
    messages = _planning_messages(checked_question, context)
    models: tuple[str, ...] = (checked_model,)
    if checked_model == DEFAULT_PLANNER_MODEL and checked_fallback is not None:
        if checked_fallback != checked_model:
            models += (checked_fallback,)

    try:
        for index, current_model in enumerate(models):
            try:
                response = await selected_gateway.chat(
                    current_model,
                    messages=messages,
                    schema=SearchPlan.model_json_schema(),
                )
            except ModelMissingError:
                if index + 1 < len(models):
                    continue
                raise
            except ModelIncompatibleError:
                if (
                    checked_model == DEFAULT_PLANNER_MODEL
                    and index + 1 < len(models)
                    and "cloud" not in current_model.casefold()
                ):
                    continue
                raise
            return _parse_plan(response, checked_question, context)
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
        raise ValueError("A local search planning model is required.")
    checked_question = " ".join(unicodedata.normalize("NFKC", question).split())
    checked_model = model.strip()
    if fallback_model is None:
        return checked_question, checked_model, None
    if not isinstance(fallback_model, str) or not fallback_model.strip():
        raise ValueError("The Gemma search planning fallback must be a model name.")
    checked_fallback = fallback_model.strip()
    if not checked_fallback.casefold().startswith("gemma4:"):
        raise ValueError("The search planning fallback must be a Gemma 4 model.")
    return checked_question, checked_model, checked_fallback


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
    try:
        payload = json.loads(response)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("search plan generation returned malformed JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("search plan generation returned invalid structured data.")

    payload["question"] = question
    try:
        plan = SearchPlan.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("search plan generation returned invalid structured data.") from exc
    return plan.model_copy(
        update={
            "exact_terms": _merge_terms(plan.exact_terms, context.entities),
            "normalized_constraints": _merge_terms(
                plan.normalized_constraints,
                context.constraints,
            ),
        }
    )


def _merge_terms(first: Sequence[str], second: Sequence[str]) -> tuple[str, ...]:
    return _unique_nonempty((*first, *second))


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
