from collections.abc import Mapping, Sequence
import json
from typing import Any

import pytest

from file_tagger.domain.contracts import ModelProfile
from file_tagger.domain.errors import (
    GenerationContextRejectedError,
    StructuredOutputError,
)
from file_tagger.rag.planner import SearchPlan
from file_tagger.rag.service import (
    NOT_FOUND_ANSWER,
    RagService,
    UncitedAnswerError,
    answer_question,
    execute_search_plan,
)
from file_tagger.search.results import SearchResult, make_result


class FakeModel:
    def __init__(self, response: str) -> None:
        self.response = response
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
        return self.response


class ProfiledFakeModel(FakeModel):
    def __init__(self, response: str, *, context_length: int = 32_768) -> None:
        super().__init__(response)
        self.context_length = context_length

    async def model_profile(self, role: str, model: str) -> ModelProfile:
        assert role == "generation"
        return ModelProfile(
            role="generation",
            model=model,
            digest="sha256:generation",
            context_length=self.context_length,
            capabilities=("completion",),
        )


def _evidence() -> tuple[SearchResult, ...]:
    return (
        make_result(
            {
                "file_path": r"C:\campaigns\flooded-ruin.md",
                "stable_id": "dragon-encounter",
                "page_number": 4,
                "heading": "Temple Encounters",
                "text": "A dragon guards the flooded temple.",
            },
            source="fts",
            rank=1,
            score=0.9,
        ),
    )


def test_answer_without_valid_citation_is_rejected() -> None:
    fake_model = FakeModel(
        "The adventure should include a dragon. [source: invented]"
    )

    with pytest.raises(UncitedAnswerError):
        answer_question("Find suitable encounters", _evidence(), gateway=fake_model)


def test_structured_claim_resolves_to_known_evidence() -> None:
    evidence = _evidence()
    citation = evidence[0].citations[0]
    fake_model = FakeModel(
        json.dumps(
            {
                "answer": "A dragon guards the flooded temple.",
                "claims": [
                    {
                        "text": "A dragon guards the flooded temple.",
                        "chunk_ids": [evidence[0].chunk_id],
                    }
                ],
            }
        )
    )

    answer = answer_question(
        "Find suitable encounters",
        evidence,
        gateway=fake_model,
    )

    assert answer.text == "A dragon guards the flooded temple."
    assert answer.claims[0].chunk_ids == (evidence[0].chunk_id,)
    assert answer.citations == (citation,)
    assert '"chunk_ids"' in json.dumps(fake_model.chat_calls[0]["schema"])
    assert "claims" in fake_model.chat_calls[0]["schema"]["required"]


def test_structured_claim_referencing_unknown_chunk_is_rejected() -> None:
    fake_model = FakeModel(
        json.dumps(
            {
                "answer": "A dragon guards the temple.",
                "claims": [
                    {
                        "text": "A dragon guards the temple.",
                        "chunk_ids": ["invented"],
                    }
                ],
            }
        )
    )

    with pytest.raises(UncitedAnswerError, match="unknown"):
        answer_question("Find suitable encounters", _evidence(), gateway=fake_model)


def test_structured_answer_with_invented_source_marker_is_rejected() -> None:
    evidence = _evidence()
    fake_model = FakeModel(
        json.dumps(
            {
                "answer": "A dragon guards the temple. [source: invented]",
                "claims": [
                    {
                        "text": "A dragon guards the temple.",
                        "chunk_ids": [evidence[0].chunk_id],
                    }
                ],
            }
        )
    )

    with pytest.raises(UncitedAnswerError, match="unknown"):
        answer_question("Find suitable encounters", evidence, gateway=fake_model)


def test_structured_answer_accepts_a_known_chunk_source_marker() -> None:
    evidence = _evidence()
    fake_model = FakeModel(
        json.dumps(
            {
                "answer": (
                    "A dragon guards the temple. "
                    f"[source: {evidence[0].chunk_id}]"
                ),
                "claims": [
                    {
                        "text": "A dragon guards the temple.",
                        "chunk_ids": [evidence[0].chunk_id],
                    }
                ],
            }
        )
    )

    answer = answer_question(
        "Find suitable encounters",
        evidence,
        gateway=fake_model,
    )

    assert answer.citations == evidence[0].citations


def test_empty_evidence_returns_not_found_without_model_call() -> None:
    fake_model = FakeModel("{}")

    answer = answer_question("Where is the treasure?", (), gateway=fake_model)

    assert answer.text == NOT_FOUND_ANSWER
    assert answer.claims == ()
    assert answer.citations == ()
    assert fake_model.chat_calls == []


class FakeSearch:
    def __init__(self, by_query: Mapping[str, Sequence[SearchResult]]) -> None:
        self.by_query = by_query
        self.queries: list[str] = []

    def search(self, query: str, limit: int = 10) -> tuple[SearchResult, ...]:
        self.queries.append(query)
        return tuple(self.by_query.get(query, ()))[:limit]


def _plan(*subqueries: str) -> SearchPlan:
    return SearchPlan(
        question="Find suitable encounters",
        intent_names=("encounters",),
        paraphrase_queries=tuple(subqueries),
        evidence_categories=("encounters",),
    )


def test_search_plan_subqueries_are_merged_and_diversified() -> None:
    first = _result("shared", r"C:\campaign\one.md", 1, 0.9)
    duplicate = _result("shared", r"C:\campaign\one.md", 1, 0.8)
    same_file = _result("same-file", r"C:\campaign\one.md", 2, 0.85)
    second_file = _result("other-file", r"C:\campaign\two.md", 1, 0.8)
    search = FakeSearch(
        {
            "dragon": (first, same_file),
            "temple guardian": (duplicate, second_file),
        }
    )

    evidence = execute_search_plan(
        _plan("dragon", "temple guardian"),
        search,
        per_query_limit=4,
        limit=3,
        max_sections_per_file=1,
    )

    assert search.queries == ["dragon", "temple guardian"]
    assert [result.chunk_id for result in evidence] == ["shared", "other-file"]
    assert len({result.chunk_id for result in evidence}) == len(evidence)


def test_optional_rerank_uses_bounded_structured_chat() -> None:
    evidence = (
        _result("first", r"C:\campaign\first.md", 1, 0.9),
        _result("second", r"C:\campaign\second.md", 1, 0.8),
    )
    fake_model = FakeModel(
        json.dumps(
            {
                "answer": "Second comes first.",
                "claims": [
                    {
                        "text": "Second comes first.",
                        "chunk_ids": ["second"],
                    }
                ],
            }
        )
    )
    responses = iter(
        (
            '{"ranked_chunk_ids":["second","first"]}',
            fake_model.response,
        )
    )

    class RerankingModel(FakeModel):
        async def chat(self, *args: Any, **kwargs: Any) -> str:
            response = next(responses)
            self.response = response
            return await super().chat(*args, **kwargs)

    model = RerankingModel("")

    answer = answer_question(
        "Which is most relevant?",
        evidence,
        gateway=model,
        rerank=True,
        rerank_top_k=2,
    )

    assert answer.claims[0].chunk_ids == ("second",)
    assert len(model.chat_calls) == 2
    assert "ranked_chunk_ids" in json.dumps(model.chat_calls[0]["schema"])
    assert model.chat_calls[1]["messages"][1]["content"].find("second") < (
        model.chat_calls[1]["messages"][1]["content"].find("first")
    )


def test_rerank_rejects_chunk_not_supplied_to_reranker() -> None:
    evidence = (
        _result("first", r"C:\campaign\first.md", 1, 0.9),
        _result("hidden", r"C:\campaign\hidden.md", 1, 0.1),
    )
    model = FakeModel(
        json.dumps(
            {
                "ranked_chunk_ids": ["hidden"],
            }
        )
    )

    with pytest.raises(StructuredOutputError, match="unknown chunk"):
        answer_question(
            "Which is relevant?",
            evidence,
            gateway=model,
            rerank=True,
            rerank_top_k=1,
        )


def test_context_rejection_retries_with_less_evidence() -> None:
    evidence = tuple(
        _result(f"chunk-{index}", rf"C:\campaign\file-{index}.md", 1, 1.0)
        for index in range(6)
    )

    class OverflowThenSucceed(ProfiledFakeModel):
        async def chat(
            self,
            model: str,
            *,
            messages: Sequence[Mapping[str, Any]],
            schema: Mapping[str, Any] | None = None,
            keep_alive: float | str | None = "5m",
        ) -> str:
            response = await super().chat(
                model,
                messages=messages,
                schema=schema,
                keep_alive=keep_alive,
            )
            if len(self.chat_calls) == 1:
                raise GenerationContextRejectedError("context overflow")
            return response

    model = OverflowThenSucceed(
        json.dumps(
            {
                "answer": "The first chunk is relevant.",
                "claims": [
                    {"text": "The first chunk is relevant.", "chunk_ids": ["chunk-0"]}
                ],
            }
        )
    )

    answer_question("Find relevant chunks", evidence, gateway=model)

    assert len(model.chat_calls) == 2
    first_prompt = model.chat_calls[0]["messages"][1]["content"]
    second_prompt = model.chat_calls[1]["messages"][1]["content"]
    assert "chunk-5" in first_prompt
    assert "chunk-5" not in second_prompt
    assert "chunk-0" in model.chat_calls[-1]["messages"][1]["content"]


def test_context_rejection_reduces_already_budgeted_evidence_again() -> None:
    evidence = tuple(
        _result(f"budgeted-{index}", rf"C:\campaign\budget-{index}.md", 1, 1.0)
        for index in range(12)
    )

    class BudgetOverflow(ProfiledFakeModel):
        async def chat(
            self,
            model: str,
            *,
            messages: Sequence[Mapping[str, Any]],
            schema: Mapping[str, Any] | None = None,
            keep_alive: float | str | None = "5m",
        ) -> str:
            response = await super().chat(
                model,
                messages=messages,
                schema=schema,
                keep_alive=keep_alive,
            )
            if len(self.chat_calls) == 1:
                raise GenerationContextRejectedError("context overflow")
            return response

    model = BudgetOverflow(
        json.dumps(
            {
                "answer": "The first budgeted chunk is relevant.",
                "claims": [
                    {
                        "text": "The first budgeted chunk is relevant.",
                        "chunk_ids": ["budgeted-0"],
                    }
                ],
            }
        ),
        context_length=1_800,
    )

    answer_question("Find budgeted chunks", evidence, gateway=model)

    first_prompt = model.chat_calls[0]["messages"][1]["content"]
    second_prompt = model.chat_calls[1]["messages"][1]["content"]
    first_chunk_ids = {
        item["chunk_id"]
        for item in json.loads(first_prompt.partition("Indexed evidence:\n")[2])
    }
    second_chunk_ids = {
        item["chunk_id"]
        for item in json.loads(second_prompt.partition("Indexed evidence:\n")[2])
    }
    assert 1 < len(first_chunk_ids) < len(evidence)
    assert len(second_chunk_ids) < len(first_chunk_ids)


def test_follow_up_reuses_evidence_for_unchanged_question_and_context() -> None:
    evidence = _evidence()
    search = FakeSearch({"initial": evidence})

    class FakePlanner:
        async def __call__(
            self,
            question: str,
            markdown_context: str,
            *,
            gateway: FakeModel,
        ) -> SearchPlan:
            del markdown_context, gateway
            return _plan(question)

    model = FakeModel(
        json.dumps(
            {
                "answer": "A dragon guards the flooded temple.",
                "claims": [
                    {
                        "text": "A dragon guards the flooded temple.",
                        "chunk_ids": ["dragon-encounter"],
                    }
                ],
            }
        )
    )
    service = RagService(search=search, gateway=model, planner=FakePlanner())

    service.answer_question("initial", markdown_context="# Encounter")
    service.answer_question("initial", markdown_context="# Encounter")

    assert search.queries == ["initial"]
    assert len(model.chat_calls) == 2
    assert "A dragon guards the flooded temple." in (
        model.chat_calls[1]["messages"][-2]["content"]
    )
    assert "# Encounter" in model.chat_calls[0]["messages"][-1]["content"]


def test_service_accepts_direct_evidence_as_second_argument() -> None:
    model = FakeModel(
        json.dumps(
            {
                "answer": "A dragon guards the flooded temple.",
                "claims": [
                    {
                        "text": "A dragon guards the flooded temple.",
                        "chunk_ids": ["dragon-encounter"],
                    }
                ],
            }
        )
    )
    service = RagService(gateway=model)

    answer = service.answer_question("Find encounters", _evidence())

    assert answer.text.startswith("A dragon")


def _result(
    chunk_id: str,
    path: str,
    page_number: int,
    score: float,
) -> SearchResult:
    return make_result(
        {
            "file_path": path,
            "stable_id": chunk_id,
            "page_number": page_number,
            "text": f"Evidence from {chunk_id}.",
        },
        source="hybrid",
        rank=1,
        score=score,
    )
