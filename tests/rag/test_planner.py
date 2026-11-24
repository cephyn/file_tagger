import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from file_tagger.adapters.ollama.gateway import OllamaGateway
from file_tagger.domain.errors import ModelIncompatibleError, ModelMissingError
from file_tagger.rag.context import normalize_markdown_context
from file_tagger.rag.planner import (
    DEFAULT_FALLBACK_MODEL,
    DEFAULT_PLANNER_MODEL,
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
