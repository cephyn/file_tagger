import asyncio
from collections.abc import Mapping, Sequence
import json
from typing import Any

import pytest

from file_tagger.adapters.ollama.gateway import OllamaGateway
from file_tagger.domain.contracts import ModelProfile
from file_tagger.domain.errors import (
    GenerationContextRejectedError,
    ModelIncompatibleError,
    ModelMissingError,
    OllamaUnavailableError,
)
from file_tagger.rag.context import normalize_markdown_context
from file_tagger.rag.planner import (
    DEFAULT_FALLBACK_MODEL,
    DEFAULT_PLANNER_MODEL,
    SearchPlan,
    build_search_plan,
    build_search_plan_async,
)


class FakePlannerGateway:
    def __init__(
        self,
        response: str = "",
        *,
        missing_models: frozenset[str] = frozenset(),
    ) -> None:
        self.response = response
        self.missing_models = missing_models
        self.chat_calls: list[dict[str, Any]] = []

    async def chat(
        self,
        model: str,
        *,
        messages: Sequence[Mapping[str, Any]],
        schema: Mapping[str, Any] | None = None,
        keep_alive: float | str | None = "5m",
    ) -> str:
        self.chat_calls.append(
            {
                "model": model,
                "messages": messages,
                "schema": schema,
                "keep_alive": keep_alive,
            }
        )
        if model in self.missing_models:
            raise ModelMissingError(f"Missing {model}.")
        return self.response


VALID_PLAN_JSON = """{
  "question": "Find encounters, treasure, and random tables",
  "intent_names": ["encounters", "treasure", "random tables"],
  "exact_terms": ["flooded temple", "encounters"],
  "paraphrase_queries": [
    "temple encounters for a third-level party",
    "treasure suited to an underwater temple"
  ],
  "filters": ["party level: 3", "setting: flooded temple"],
  "exclusions": ["modern weapons"],
  "normalized_constraints": ["level 3"],
  "evidence_categories": ["encounters", "treasure", "random tables"]
}"""


def test_rpg_outline_creates_targeted_search_intents() -> None:
    gateway = FakePlannerGateway(VALID_PLAN_JSON)

    plan = build_search_plan(
        question="Find encounters, treasure, and random tables",
        markdown_context="Level 3 party exploring a flooded temple",
        gateway=gateway,
    )

    assert {"encounters", "treasure", "random tables"} <= set(plan.intent_names)
    assert "level 3" in plan.normalized_constraints
    assert plan.intents == plan.intent_names
    assert plan.exact_entities == plan.exact_terms
    assert plan.subqueries == plan.paraphrase_queries
    assert gateway.chat_calls[0]["model"] == DEFAULT_PLANNER_MODEL
    schema = gateway.chat_calls[0]["schema"]
    assert {
        "intent_names",
        "exact_terms",
        "paraphrase_queries",
        "filters",
        "exclusions",
        "evidence_categories",
    } <= set(schema["properties"])
    prompt = gateway.chat_calls[0]["messages"][1]["content"]
    assert "Question:" in prompt
    assert "Normalized Markdown context:" in prompt


def test_markdown_sections_and_entities_are_normalized() -> None:
    context = normalize_markdown_context(
        """
# Flooded Temple

## Party

- **Level:** 3
- `Anira`

## Complications

The [[Sunken Vault]] is unstable.
"""
    )

    assert context.section_names == ("Flooded Temple", "Party", "Complications")
    assert "Level 3" in context.entities
    assert "Anira" in context.entities
    assert "Sunken Vault" in context.entities
    assert "level 3" in context.constraints
    assert "  " not in context.normalized_text


def test_multiline_html_comments_are_removed_before_markdown_parsing() -> None:
    context = normalize_markdown_context(
        """
# Visible

<!--
## Hidden

`Hidden Entity`
Level 99
-->

## Also Visible

`Actual Entity`
"""
    )

    assert context.section_names == ("Visible", "Also Visible")
    assert "Hidden" not in context.normalized_text
    assert "Hidden Entity" not in context.entities
    assert "level 99" not in context.constraints
    assert "Actual Entity" in context.entities


def test_planner_merges_local_markdown_constraints_into_model_plan() -> None:
    payload = VALID_PLAN_JSON.replace(
        '"normalized_constraints": ["level 3"]',
        '"normalized_constraints": []',
    )

    plan = build_search_plan(
        question="Find encounters, treasure, and random tables",
        markdown_context="- **Level:** 3",
        gateway=FakePlannerGateway(payload),
    )

    assert plan.normalized_constraints == ("level 3",)


def test_planner_caps_markdown_entities_added_to_exact_terms() -> None:
    headings = [f"Section {index}" for index in range(17)]
    payload = VALID_PLAN_JSON.replace(
        '"exact_terms": ["flooded temple", "encounters"]',
        '"exact_terms": []',
    )

    plan = build_search_plan(
        question="Find encounters",
        markdown_context="\n".join(f"# {heading}" for heading in headings),
        gateway=FakePlannerGateway(payload),
    )

    assert len(plan.exact_terms) == 16
    assert plan.exact_terms == tuple(headings[:16])


def test_planner_omits_oversized_markdown_entities_from_exact_terms() -> None:
    oversized = "x" * 129
    payload = VALID_PLAN_JSON.replace(
        '"exact_terms": ["flooded temple", "encounters"]',
        '"exact_terms": []',
    )

    plan = build_search_plan(
        question="Find encounters",
        markdown_context=f"# Valid Heading\n# {oversized}",
        gateway=FakePlannerGateway(payload),
    )

    assert plan.exact_terms == ("Valid Heading",)


def test_planner_deduplicates_model_and_markdown_exact_terms() -> None:
    payload = VALID_PLAN_JSON.replace(
        '"exact_terms": ["flooded temple", "encounters"]',
        '"exact_terms": ["Shared Heading"]',
    )

    plan = build_search_plan(
        question="Find encounters",
        markdown_context="# shared heading\n# Unique Heading",
        gateway=FakePlannerGateway(payload),
    )

    assert plan.exact_terms == ("Shared Heading", "Unique Heading")


@pytest.mark.parametrize(
    "terms",
    [
        [f"term {index}" for index in range(17)],
        ["x" * 129],
    ],
)
def test_model_generated_exact_terms_keep_structured_bounds(
    terms: list[str],
) -> None:
    payload = VALID_PLAN_JSON.replace(
        '"exact_terms": ["flooded temple", "encounters"]',
        f'"exact_terms": {json.dumps(terms)}',
    )

    with pytest.raises(ValueError, match="structured"):
        build_search_plan(
            question="Find encounters",
            gateway=FakePlannerGateway(payload),
        )


def test_plan_canonicalizes_intents_constraints_and_evidence_categories() -> None:
    payload = (
        VALID_PLAN_JSON
        .replace('"encounters", "treasure", "random tables"', '"Encounters", "Treasure", "Random Tables"')
        .replace('"normalized_constraints": ["level 3"]', '"normalized_constraints": ["Level 3"]')
    )

    plan = build_search_plan(
        question="Find encounters, treasure, and random tables",
        markdown_context="Level 3",
        gateway=FakePlannerGateway(payload),
    )

    assert plan.intent_names == ("encounters", "treasure", "random tables")
    assert plan.normalized_constraints == ("level 3",)
    assert plan.evidence_categories == ("encounters", "treasure", "random tables")


def test_planner_uses_selected_gemma_fallback_when_inkling_is_missing() -> None:
    gateway = FakePlannerGateway(
        VALID_PLAN_JSON,
        missing_models=frozenset({DEFAULT_PLANNER_MODEL}),
    )

    plan = asyncio.run(
        build_search_plan_async(
            question="Find encounters",
            markdown_context="Level 3",
            fallback_model="gemma4:e4b",
            gateway=gateway,
        )
    )

    assert plan.intent_names
    assert [call["model"] for call in gateway.chat_calls] == [
        DEFAULT_PLANNER_MODEL,
        "gemma4:e4b",
    ]


@pytest.mark.parametrize("response", ['{"intent_names":', '{"intent_names": []}'])
def test_invalid_or_empty_plans_fail(response: str) -> None:
    with pytest.raises(ValueError, match="search plan|structured"):
        build_search_plan(
            question="Find encounters",
            markdown_context="Level 3",
            gateway=FakePlannerGateway(response),
        )


def test_cloud_planning_model_is_rejected_before_chat() -> None:
    class NoChatClient:
        async def chat(self, **kwargs: Any) -> None:
            raise AssertionError("cloud model reached the Ollama client")

    gateway = OllamaGateway(client=NoChatClient())

    with pytest.raises(ModelIncompatibleError, match="Cloud"):
        build_search_plan(
            question="Find encounters",
            markdown_context="Level 3",
            model="gemma4:cloud",
            gateway=gateway,
        )


def test_default_fallback_is_a_local_gemma_model() -> None:
    assert DEFAULT_FALLBACK_MODEL.startswith("gemma4:")


def test_search_plan_schema_bounds_cardinality_and_each_string() -> None:
    schema = SearchPlan.model_json_schema()

    assert schema["properties"]["question"]["maxLength"] > 0
    for field in (
        "intent_names",
        "exact_terms",
        "paraphrase_queries",
        "filters",
        "exclusions",
        "normalized_constraints",
        "evidence_categories",
    ):
        field_schema = schema["properties"][field]
        assert field_schema["maxItems"] > 0
        assert field_schema["items"]["maxLength"] > 0


def test_oversized_context_is_rejected_before_chat() -> None:
    class SmallContextGateway(FakePlannerGateway):
        async def model_profile(self, role: str, model: str) -> ModelProfile:
            assert role == "generation"
            return ModelProfile(
                role="generation",
                model=model,
                digest="sha256:test",
                context_length=512,
                capabilities=("completion",),
            )

    gateway = SmallContextGateway(VALID_PLAN_JSON)

    with pytest.raises(ModelIncompatibleError, match="context"):
        build_search_plan(
            question="Find encounters",
            markdown_context="source " * 5_000,
            gateway=gateway,
        )

    assert gateway.chat_calls == []


@pytest.mark.parametrize(
    ("model", "fallback_model"),
    [
        ("gemma4:cloud", None),
        (DEFAULT_PLANNER_MODEL, "gemma4:cloud"),
        (DEFAULT_PLANNER_MODEL, "cloud:gemma4"),
    ],
)
def test_cloud_models_are_rejected_by_planner_before_fake_gateway_calls(
    model: str,
    fallback_model: str | None,
) -> None:
    gateway = FakePlannerGateway(VALID_PLAN_JSON)

    with pytest.raises(ModelIncompatibleError, match="Cloud"):
        build_search_plan(
            question="Find encounters",
            markdown_context="Level 3",
            model=model,
            fallback_model=fallback_model,
            gateway=gateway,
        )

    assert gateway.chat_calls == []


def test_unbounded_bad_response_is_rejected() -> None:
    response = VALID_PLAN_JSON.replace(
        '"flooded temple"',
        f'"{"x" * 100_000}"',
    )

    with pytest.raises(ValueError, match="structured"):
        build_search_plan(
            question="Find encounters",
            markdown_context="Level 3",
            gateway=FakePlannerGateway(response),
        )


def test_planner_falls_back_after_wrong_primary_capability() -> None:
    class WrongCapabilityGateway(FakePlannerGateway):
        async def model_profile(self, role: str, model: str) -> ModelProfile:
            return ModelProfile(
                role="generation",
                model=model,
                digest=f"sha256:{model}",
                context_length=32_000,
                capabilities=(
                    ("embedding",)
                    if model == DEFAULT_PLANNER_MODEL
                    else ("completion",)
                ),
            )

    gateway = WrongCapabilityGateway(VALID_PLAN_JSON)

    plan = build_search_plan(
        question="Find encounters",
        markdown_context="Level 3",
        gateway=gateway,
    )

    assert plan.intent_names
    assert [call["model"] for call in gateway.chat_calls] == ["gemma4:e4b"]


def test_planner_falls_back_after_primary_structured_output_failure() -> None:
    class StructuredOutputFallbackGateway(FakePlannerGateway):
        def __init__(self) -> None:
            super().__init__()

        async def chat(
            self,
            model: str,
            *,
            messages: Sequence[Mapping[str, Any]],
            schema: Mapping[str, Any] | None = None,
            keep_alive: float | str | None = "5m",
        ) -> str:
            self.chat_calls.append(
                {
                    "model": model,
                    "messages": messages,
                    "schema": schema,
                    "keep_alive": keep_alive,
                }
            )
            if model == DEFAULT_PLANNER_MODEL:
                return '{"intent_names":'
            return VALID_PLAN_JSON

    gateway = StructuredOutputFallbackGateway()

    plan = build_search_plan(
        question="Find encounters",
        markdown_context="Level 3",
        gateway=gateway,
    )

    assert plan.intent_names
    assert [call["model"] for call in gateway.chat_calls] == [
        DEFAULT_PLANNER_MODEL,
        DEFAULT_FALLBACK_MODEL,
    ]


def test_planner_falls_back_when_only_primary_context_is_too_small() -> None:
    class ProfiledContextGateway(FakePlannerGateway):
        async def model_profile(self, role: str, model: str) -> ModelProfile:
            return ModelProfile(
                role="generation",
                model=model,
                digest=f"sha256:{model}",
                context_length=(
                    512 if model == DEFAULT_PLANNER_MODEL else 128_000
                ),
                capabilities=("completion",),
            )

    gateway = ProfiledContextGateway(VALID_PLAN_JSON)

    plan = build_search_plan(
        question="Find encounters",
        markdown_context="Level 3 " * 1_000,
        gateway=gateway,
    )

    assert plan.intent_names
    assert [call["model"] for call in gateway.chat_calls] == [DEFAULT_FALLBACK_MODEL]


def test_planner_falls_back_after_chat_context_rejection() -> None:
    class ContextRejectingGateway(FakePlannerGateway):
        async def chat(
            self,
            model: str,
            *,
            messages: Sequence[Mapping[str, Any]],
            schema: Mapping[str, Any] | None = None,
            keep_alive: float | str | None = "5m",
        ) -> str:
            self.chat_calls.append(
                {
                    "model": model,
                    "messages": messages,
                    "schema": schema,
                    "keep_alive": keep_alive,
                }
            )
            if model == DEFAULT_PLANNER_MODEL:
                raise GenerationContextRejectedError("context too small")
            return VALID_PLAN_JSON

    gateway = ContextRejectingGateway()

    assert build_search_plan(
        question="Find encounters",
        markdown_context="Level 3",
        gateway=gateway,
    ).intent_names
    assert [call["model"] for call in gateway.chat_calls] == [
        DEFAULT_PLANNER_MODEL,
        DEFAULT_FALLBACK_MODEL,
    ]


def test_planner_does_not_fallback_when_ollama_server_is_unavailable() -> None:
    class UnavailableGateway(FakePlannerGateway):
        async def chat(
            self,
            model: str,
            *,
            messages: Sequence[Mapping[str, Any]],
            schema: Mapping[str, Any] | None = None,
            keep_alive: float | str | None = "5m",
        ) -> str:
            self.chat_calls.append(
                {
                    "model": model,
                    "messages": messages,
                    "schema": schema,
                    "keep_alive": keep_alive,
                }
            )
            raise OllamaUnavailableError("server offline")

    gateway = UnavailableGateway()

    with pytest.raises(OllamaUnavailableError, match="offline"):
        build_search_plan(
            question="Find encounters",
            markdown_context="Level 3",
            gateway=gateway,
        )

    assert [call["model"] for call in gateway.chat_calls] == [DEFAULT_PLANNER_MODEL]
