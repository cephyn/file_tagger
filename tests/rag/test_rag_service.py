import asyncio
from collections.abc import Mapping, Sequence
import json
from typing import Any

import pytest

from file_tagger.adapters.gemini.gateway import (
    GEMINI_EMBEDDING_MODEL,
    GEMINI_GENERATION_MODEL,
)
from file_tagger.adapters.ollama.gateway import OllamaGateway
from file_tagger.domain.contracts import ModelProfile
from file_tagger.domain.errors import (
    GenerationContextRejectedError,
    GeminiUnavailableError,
    StructuredOutputError,
)
from file_tagger.rag.planner import SearchPlan
from file_tagger.rag.service import (
    NOT_FOUND_ANSWER,
    RagService,
    UncitedAnswerError,
    answer_question,
    execute_search_plan,
    execute_search_plan_async,
)
from file_tagger.search.hybrid import SearchFilters
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


def test_close_session_removes_rag_conversation_state() -> None:
    service = RagService(search=None, gateway=FakeModel("{}"))
    service._session_states["conversation-1"] = object()  # type: ignore[assignment]

    service.close_session("conversation-1")

    assert "conversation-1" not in service._session_states
    assert "conversation-1" not in service._session_locks


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


def test_gemini_rag_routes_planning_embedding_and_generation_gateways() -> None:
    evidence = _evidence()
    claim = "A dragon guards the flooded temple."
    profile = ModelProfile(
        role="embedding",
        model="gemini-embedding-2",
        digest="gemini-embedding-digest",
        context_length=32_768,
        dimensions=768,
        capabilities=("embedding",),
    )

    class GeminiSearch:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def search(
            self,
            query: str,
            query_embedding: Sequence[float] | None = None,
            **kwargs: Any,
        ) -> tuple[SearchResult, ...]:
            self.calls.append(
                {"query": query, "query_embedding": query_embedding, **kwargs}
            )
            return evidence

    class GeminiEmbeddingGateway:
        provider = "gemini"

        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def embed(
            self,
            model: str,
            text: str | Sequence[str],
            *,
            dimensions: int | None = None,
            truncate: bool = False,
        ) -> tuple[tuple[float, ...], ...]:
            queries = (text,) if isinstance(text, str) else tuple(text)
            self.calls.append(
                {
                    "model": model,
                    "queries": queries,
                    "dimensions": dimensions,
                    "truncate": truncate,
                }
            )
            return tuple((0.0,) * 768 for _ in queries)

    generation_gateway = ProfiledFakeModel(
        json.dumps(
            {
                "answer": claim,
                "claims": [
                    {
                        "text": claim,
                        "evidence_quote": claim,
                        "chunk_ids": [evidence[0].chunk_id],
                    }
                ],
            }
        )
    )
    embedding_gateway = GeminiEmbeddingGateway()
    search = GeminiSearch()
    planner_calls: list[dict[str, Any]] = []

    def gemini_planner(
        question: str,
        markdown_context: str = "",
        *,
        gateway: Any | None = None,
        model: str | None = None,
        fallback_model: str | None = None,
    ) -> SearchPlan:
        planner_calls.append(
            {
                "question": question,
                "markdown_context": markdown_context,
                "gateway": gateway,
                "model": model,
                "fallback_model": fallback_model,
            }
        )
        return SearchPlan(
            question=question,
            intent_names=("encounters",),
            exact_terms=("flooded temple",),
            paraphrase_queries=("temple guardians",),
            evidence_categories=("encounters",),
        )

    service = RagService(
        search=search,
        gateway=generation_gateway,
        embedding_gateway=embedding_gateway,
        planner=gemini_planner,
        model="gemini-4-hydrogen",
        embedding_profile=profile,
    )

    answer = service.answer_question(
        "Find suitable encounters",
        markdown_context="A flooded ruin",
    )

    assert answer.text == claim
    assert planner_calls[0]["gateway"] is generation_gateway
    assert planner_calls[0]["model"] == "gemini-4-hydrogen"
    assert planner_calls[0]["fallback_model"] is None
    assert [call["model"] for call in embedding_gateway.calls] == [
        "gemini-embedding-2"
    ]
    assert [call["model"] for call in generation_gateway.chat_calls] == [
        "gemini-4-hydrogen"
    ]


def test_exact_gemini_generation_model_without_gateway_raises_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import file_tagger.rag.service as rag_service

    def unexpected_ollama_gateway() -> None:
        raise AssertionError("OllamaGateway must not be constructed")

    monkeypatch.setattr(rag_service, "OllamaGateway", unexpected_ollama_gateway)

    with pytest.raises(GeminiUnavailableError):
        answer_question(
            "Find encounters",
            _evidence(),
            model="gemini-4-hydrogen",
        )


def test_exact_gemini_embedding_model_without_gateway_raises_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import file_tagger.rag.service as rag_service

    def unexpected_ollama_gateway() -> None:
        raise AssertionError("OllamaGateway must not be constructed")

    monkeypatch.setattr(rag_service, "OllamaGateway", unexpected_ollama_gateway)
    profile = ModelProfile(
        role="embedding",
        model="gemini-embedding-2",
        digest="gemini-embedding-digest",
        context_length=32_768,
        dimensions=768,
        capabilities=("embedding",),
    )

    with pytest.raises(GeminiUnavailableError):
        RagService(
            search=object(),
            model="gemma4:e4b",
            embedding_profile=profile,
        )


def test_exact_gemini_embedding_profile_rejects_missing_embedding_gateway() -> None:
    generation_gateway = ProfiledFakeModel("{}")
    profile = ModelProfile(
        role="embedding",
        model=GEMINI_EMBEDDING_MODEL,
        digest="gemini-embedding-digest",
        context_length=32_768,
        dimensions=768,
        capabilities=("embedding",),
    )

    with pytest.raises(
        GeminiUnavailableError,
        match="The Gemini embedding gateway is unavailable.",
    ):
        RagService(
            search=object(),
            gateway=generation_gateway,
            embedding_gateway=None,
            model=GEMINI_GENERATION_MODEL,
            embedding_profile=profile,
        )


def test_exact_gemini_embedding_profile_rejects_generation_gateway_as_embedding_gateway() -> None:
    generation_gateway = ProfiledFakeModel("{}")
    profile = ModelProfile(
        role="embedding",
        model=GEMINI_EMBEDDING_MODEL,
        digest="gemini-embedding-digest",
        context_length=32_768,
        dimensions=768,
        capabilities=("embedding",),
    )

    with pytest.raises(
        GeminiUnavailableError,
        match="The Gemini embedding gateway is unavailable.",
    ):
        RagService(
            search=object(),
            gateway=generation_gateway,
            embedding_gateway=generation_gateway,
            model=GEMINI_GENERATION_MODEL,
            embedding_profile=profile,
        )


def test_exact_gemini_embedding_profile_accepts_gemini_embedding_gateway() -> None:
    generation_gateway = ProfiledFakeModel("{}")

    class GeminiEmbeddingGateway:
        provider = "gemini"

        async def embed(
            self,
            model: str,
            text: str | Sequence[str],
            *,
            dimensions: int | None = None,
            truncate: bool = False,
            keep_alive: float | str | None = "5m",
        ) -> tuple[tuple[float, ...], ...]:
            return ((0.0,),)

    embedding_gateway = GeminiEmbeddingGateway()
    profile = ModelProfile(
        role="embedding",
        model=GEMINI_EMBEDDING_MODEL,
        digest="gemini-embedding-digest",
        context_length=32_768,
        dimensions=768,
        capabilities=("embedding",),
    )

    service = RagService(
        search=object(),
        gateway=generation_gateway,
        embedding_gateway=embedding_gateway,
        model=GEMINI_GENERATION_MODEL,
        embedding_profile=profile,
    )

    assert service._embedding_gateway is embedding_gateway


def test_exact_gemini_embedding_profile_rejects_gateway_without_gemini_provider() -> None:
    generation_gateway = ProfiledFakeModel("{}")

    class DuckTypedEmbeddingGateway:
        async def embed(
            self,
            model: str,
            text: str | Sequence[str],
            *,
            dimensions: int | None = None,
            truncate: bool = False,
            keep_alive: float | str | None = "5m",
        ) -> tuple[tuple[float, ...], ...]:
            return ((0.0,),)

    profile = ModelProfile(
        role="embedding",
        model=GEMINI_EMBEDDING_MODEL,
        digest="gemini-embedding-digest",
        context_length=32_768,
        dimensions=768,
        capabilities=("embedding",),
    )

    with pytest.raises(
        GeminiUnavailableError,
        match="The Gemini embedding gateway is unavailable.",
    ):
        RagService(
            search=object(),
            gateway=generation_gateway,
            embedding_gateway=DuckTypedEmbeddingGateway(),
            model=GEMINI_GENERATION_MODEL,
            embedding_profile=profile,
        )


def test_exact_gemini_embedding_profile_rejects_ollama_embedding_gateway() -> None:
    generation_gateway = ProfiledFakeModel("{}")
    profile = ModelProfile(
        role="embedding",
        model=GEMINI_EMBEDDING_MODEL,
        digest="gemini-embedding-digest",
        context_length=32_768,
        dimensions=768,
        capabilities=("embedding",),
    )

    with pytest.raises(
        GeminiUnavailableError,
        match="The Gemini embedding gateway is unavailable.",
    ):
        RagService(
            search=object(),
            gateway=generation_gateway,
            embedding_gateway=OllamaGateway(),
            model=GEMINI_GENERATION_MODEL,
            embedding_profile=profile,
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
                        "evidence_quote": "A dragon guards the flooded temple.",
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
    assert '"evidence_quote"' in json.dumps(fake_model.chat_calls[0]["schema"])
    assert "claims" in fake_model.chat_calls[0]["schema"]["required"]


def test_structured_claim_referencing_unknown_chunk_is_rejected() -> None:
    fake_model = FakeModel(
        json.dumps(
            {
                "answer": "A dragon guards the temple.",
                "claims": [
                    {
                        "text": "A dragon guards the temple.",
                        "evidence_quote": "A dragon guards the flooded temple.",
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
                        "evidence_quote": "A dragon guards the flooded temple.",
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
    claim = "A dragon guards the flooded temple."
    fake_model = FakeModel(
        json.dumps(
            {
                "answer": (
                    f"{claim} "
                    f"[source: {evidence[0].chunk_id}]"
                ),
                "claims": [
                    {
                        "text": claim,
                        "evidence_quote": claim,
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


def test_structured_answer_accepts_contiguous_markers_for_same_claim() -> None:
    claim = "A dragon guards the flooded temple."
    first = make_result(
        {
            "file_path": r"C:\campaign\first.md",
            "stable_id": "a",
            "page_number": 1,
            "text": claim,
        },
        source="hybrid",
    )
    second = make_result(
        {
            "file_path": r"C:\campaign\second.md",
            "stable_id": "b",
            "page_number": 1,
            "text": claim,
        },
        source="hybrid",
    )
    fake_model = FakeModel(
        json.dumps(
            {
                "answer": f"{claim} [source: a] [source: b]",
                "claims": [
                    {
                        "text": claim,
                        "evidence_quote": claim,
                        "chunk_ids": ["a", "b"],
                    }
                ],
            }
        )
    )

    answer = answer_question(
        "Find suitable encounters",
        (first, second),
        gateway=fake_model,
    )

    assert answer.claims[0].chunk_ids == ("a", "b")
    assert answer.citations == (first.citations[0], second.citations[0])


def test_structured_answer_rejects_grouped_marker_outside_claim_chunks() -> None:
    claim = "A dragon guards the flooded temple."
    first = make_result(
        {
            "file_path": r"C:\campaign\first.md",
            "stable_id": "a",
            "page_number": 1,
            "text": claim,
        },
        source="hybrid",
    )
    unrelated = make_result(
        {
            "file_path": r"C:\campaign\unrelated.md",
            "stable_id": "b",
            "page_number": 1,
            "text": claim,
        },
        source="hybrid",
    )
    fake_model = FakeModel(
        json.dumps(
            {
                "answer": f"{claim} [source: a] [source: b]",
                "claims": [
                    {
                        "text": claim,
                        "evidence_quote": claim,
                        "chunk_ids": ["a"],
                    }
                ],
            }
        )
    )

    with pytest.raises(UncitedAnswerError, match="does not match"):
        answer_question(
            "Find suitable encounters",
            (first, unrelated),
            gateway=fake_model,
        )


def test_structured_answer_rejects_grouped_marker_with_different_full_quote() -> None:
    claim = "A dragon guards the flooded temple."
    first = make_result(
        {
            "file_path": r"C:\campaign\first.md",
            "stable_id": "a",
            "page_number": 1,
            "text": claim,
        },
        source="hybrid",
    )
    second = make_result(
        {
            "file_path": r"C:\campaign\second.md",
            "stable_id": "b",
            "page_number": 1,
            "text": "A dragon guards the northern tower.",
        },
        source="hybrid",
    )
    fake_model = FakeModel(
        json.dumps(
            {
                "answer": f"{claim} [source: a] [source: b]",
                "claims": [
                    {
                        "text": claim,
                        "evidence_quote": claim,
                        "chunk_ids": ["a", "b"],
                    }
                ],
            }
        )
    )

    with pytest.raises(UncitedAnswerError, match="complete full cited chunk"):
        answer_question(
            "Find suitable encounters",
            (first, second),
            gateway=fake_model,
        )


def test_structured_claim_accepts_safe_unicode_case_and_whitespace_normalization(
) -> None:
    quote = "Ａ DRAGON\nguards the flooded temple！"
    evidence = (
        make_result(
            {
                "file_path": r"C:\campaign\normalized.md",
                "stable_id": "normalized",
                "page_number": 1,
                "text": quote,
            },
            source="hybrid",
        ),
    )
    claim = "a  dragon guards\tthe flooded temple!"
    fake_model = FakeModel(
        json.dumps(
            {
                "answer": claim,
                "claims": [
                    {
                        "text": claim,
                        "evidence_quote": quote,
                        "chunk_ids": ["normalized"],
                    }
                ],
            }
        )
    )

    answer = answer_question("What guards the temple?", evidence, gateway=fake_model)

    assert answer.text == claim


def test_structured_claim_requires_quote_to_match_every_cited_chunk() -> None:
    dragon = make_result(
        {
            "file_path": r"C:\campaign\dragon.md",
            "stable_id": "dragon",
            "page_number": 1,
            "text": "A dragon guards the flooded temple.",
        },
        source="hybrid",
    )
    knight = make_result(
        {
            "file_path": r"C:\campaign\knight.md",
            "stable_id": "knight",
            "page_number": 1,
            "text": "A knight patrols the western bridge.",
        },
        source="hybrid",
    )
    claim = "A dragon guards the flooded temple."
    fake_model = FakeModel(
        json.dumps(
            {
                "answer": claim,
                "claims": [
                    {
                        "text": claim,
                        "evidence_quote": claim,
                        "chunk_ids": ["dragon", "knight"],
                    }
                ],
            }
        )
    )

    with pytest.raises(UncitedAnswerError, match="complete full cited chunk"):
        answer_question("Describe the guards", (dragon, knight), gateway=fake_model)


@pytest.mark.parametrize(
    ("claim", "evidence_quote"),
    (
        ("The success rate is 50.", "The success rate is 50%."),
        ("The result was 'approved'.", "The result was “approved”."),
        ("The route is east-west.", "The route is east—west."),
    ),
)
def test_structured_claim_preserves_punctuation_and_symbols(
    claim: str,
    evidence_quote: str,
) -> None:
    evidence = (
        make_result(
            {
                "file_path": r"C:\campaign\punctuation.md",
                "stable_id": "punctuation",
                "page_number": 1,
                "text": evidence_quote,
            },
            source="hybrid",
        ),
    )
    fake_model = FakeModel(
        json.dumps(
            {
                "answer": claim,
                "claims": [
                    {
                        "text": claim,
                        "evidence_quote": evidence_quote,
                        "chunk_ids": ["punctuation"],
                    }
                ],
            }
        )
    )

    with pytest.raises(UncitedAnswerError, match="not supported"):
        answer_question("Summarize the result", evidence, gateway=fake_model)


def test_structured_quote_preserves_percent_symbol_from_cited_chunk() -> None:
    evidence = (
        make_result(
            {
                "file_path": r"C:\campaign\percentage.md",
                "stable_id": "percentage",
                "page_number": 1,
                "text": "The success rate is 50%.",
            },
            source="hybrid",
        ),
    )
    fake_model = FakeModel(
        json.dumps(
            {
                "answer": "The success rate is 50.",
                "claims": [
                    {
                        "text": "The success rate is 50.",
                        "evidence_quote": "The success rate is 50.",
                        "chunk_ids": ["percentage"],
                    }
                ],
            }
        )
    )

    with pytest.raises(UncitedAnswerError, match="complete full cited chunk"):
        answer_question("What is the success rate?", evidence, gateway=fake_model)


@pytest.mark.parametrize("visible_markers", ("none", "first", "all"))
def test_structured_answer_matches_mt_claims_without_sentence_segmentation(
    visible_markers: str,
) -> None:
    mountain = make_result(
        {
            "file_path": r"C:\geography\mountain.md",
            "stable_id": "mountain",
            "page_number": 1,
            "text": "Mt. Everest is the highest mountain above sea level.",
        },
        source="hybrid",
    )
    knight = make_result(
        {
            "file_path": r"C:\campaign\knight.md",
            "stable_id": "knight",
            "page_number": 1,
            "text": "The knight crossed the eastern bridge.",
        },
        source="hybrid",
    )
    claims = (
        "Mt. Everest is the highest mountain above sea level.",
        "The knight crossed the eastern bridge.",
    )
    answer = " ".join(claims)
    if visible_markers == "first":
        answer = f"{claims[0]} [source: mountain] {claims[1]}"
    elif visible_markers == "all":
        answer = (
            f"{claims[0]} [source: mountain] "
            f"{claims[1]} [source: knight]"
        )
    fake_model = FakeModel(
        json.dumps(
            {
                "answer": answer,
                "claims": [
                    {
                        "text": claims[0],
                        "evidence_quote": claims[0],
                        "chunk_ids": ["mountain"],
                    },
                    {
                        "text": claims[1],
                        "evidence_quote": claims[1],
                        "chunk_ids": ["knight"],
                    },
                ],
            }
        )
    )

    answer = answer_question(
        "Summarize the mountain and the knight",
        (mountain, knight),
        gateway=fake_model,
    )

    assert [claim.text for claim in answer.claims] == list(claims)


def test_structured_answer_marker_cannot_cover_multiple_claim_spans() -> None:
    mountain = make_result(
        {
            "file_path": r"C:\geography\mountain.md",
            "stable_id": "mountain",
            "page_number": 1,
            "text": "Mt. Everest is above sea level.",
        },
        source="hybrid",
    )
    knight = make_result(
        {
            "file_path": r"C:\campaign\knight.md",
            "stable_id": "knight",
            "page_number": 1,
            "text": "The knight crossed the bridge.",
        },
        source="hybrid",
    )
    mountain_claim = "Mt. Everest is above sea level."
    knight_claim = "The knight crossed the bridge."
    fake_model = FakeModel(
        json.dumps(
            {
                "answer": (
                    f"{mountain_claim} {knight_claim} "
                    "[source: knight]"
                ),
                "claims": [
                    {
                        "text": mountain_claim,
                        "evidence_quote": mountain_claim,
                        "chunk_ids": ["mountain"],
                    },
                    {
                        "text": knight_claim,
                        "evidence_quote": knight_claim,
                        "chunk_ids": ["knight"],
                    },
                ],
            }
        )
    )

    with pytest.raises(UncitedAnswerError, match="not bound"):
        answer_question(
            "Summarize both",
            (mountain, knight),
            gateway=fake_model,
        )


@pytest.mark.parametrize(
    "marker_position",
    (
        "leading",
        "leading_colon",
        "leading_no_space",
        "separate_line",
        "standalone_before",
        "standalone_between",
        "after_claim_colon",
        "after_claim_no_space",
    ),
)
def test_structured_answer_rejects_orphan_source_marker_fragments(
    marker_position: str,
) -> None:
    evidence = _evidence()
    marker = f"[source: {evidence[0].chunk_id}]"
    claim = "A dragon guards the flooded temple."
    answers = {
        "leading": f"{marker} {claim}",
        "leading_colon": f"{marker}: {claim}",
        "leading_no_space": f"{marker}{claim}",
        "separate_line": f"{claim}\n{marker}",
        "standalone_before": f"{marker}\n{claim}",
        "standalone_between": f"{claim}\n{marker}\n{claim}",
        "after_claim_colon": f"{claim} {marker}: {claim}",
        "after_claim_no_space": f"{claim} {marker}{claim}",
    }
    answer = answers[marker_position]
    fake_model = FakeModel(
        json.dumps(
            {
                "answer": answer,
                "claims": [
                    {
                        "text": claim,
                        "evidence_quote": claim,
                        "chunk_ids": [evidence[0].chunk_id],
                    }
                ],
            }
        )
    )

    with pytest.raises(UncitedAnswerError, match="structured claim"):
        answer_question("Find suitable encounters", evidence, gateway=fake_model)


def test_structured_claim_marker_must_resolve_to_its_claim_chunks() -> None:
    first = make_result(
        {
            "file_path": r"C:\campaign\first.md",
            "stable_id": "first",
            "page_number": 1,
            "text": "The first witness arrived early.",
        },
        source="fts",
    )
    second = make_result(
        {
            "file_path": r"C:\campaign\second.md",
            "stable_id": "second",
            "page_number": 1,
            "text": "The second witness arrived late.",
        },
        source="fts",
    )
    fake_model = FakeModel(
        json.dumps(
            {
                "answer": "The first witness arrived early.",
                "claims": [
                    {
                        "text": (
                            "The first witness arrived early. "
                            "[source: second]"
                        ),
                        "evidence_quote": "The first witness arrived early.",
                        "chunk_ids": ["first"],
                    }
                ],
            }
        )
    )

    with pytest.raises(UncitedAnswerError, match="does not match"):
        answer_question("What happened?", (first, second), gateway=fake_model)


def test_structured_answer_rejects_duplicate_visible_source_markers() -> None:
    evidence = _evidence()
    chunk_id = evidence[0].chunk_id
    fake_model = FakeModel(
        json.dumps(
            {
                "answer": (
                    "A dragon guards the flooded temple. "
                    f"[source: {chunk_id}] [source: {chunk_id}]"
                ),
                "claims": [
                    {
                        "text": "A dragon guards the flooded temple.",
                        "evidence_quote": "A dragon guards the flooded temple.",
                        "chunk_ids": [chunk_id],
                    }
                ],
            }
        )
    )

    with pytest.raises(UncitedAnswerError, match="duplicate"):
        answer_question("Find suitable encounters", evidence, gateway=fake_model)


def test_structured_answer_marker_must_match_the_preceding_claim() -> None:
    first = make_result(
        {
            "file_path": r"C:\campaign\first.md",
            "stable_id": "first",
            "page_number": 1,
            "text": "The first witness arrived early.",
        },
        source="fts",
    )
    second = make_result(
        {
            "file_path": r"C:\campaign\second.md",
            "stable_id": "second",
            "page_number": 1,
            "text": "The second witness arrived late.",
        },
        source="fts",
    )
    fake_model = FakeModel(
        json.dumps(
            {
                "answer": (
                    "The first witness arrived early. [source: second]"
                ),
                "claims": [
                    {
                        "text": "The first witness arrived early.",
                        "evidence_quote": "The first witness arrived early.",
                        "chunk_ids": ["first"],
                    }
                ],
            }
        )
    )

    with pytest.raises(UncitedAnswerError, match="does not match"):
        answer_question("What happened?", (first, second), gateway=fake_model)


@pytest.mark.parametrize(
    "quote",
    (
        "The witness likely left before dawn.",
        "The witness conceivably left before dawn.",
        "The witness unlikely left before dawn.",
        "The witness probably left before dawn.",
        "The witness possibly left before dawn.",
        "The witness may have left before dawn.",
        "The witness might have left before dawn.",
        "The witness reportedly left before dawn.",
        "The witness allegedly left before dawn.",
        "The witness purportedly left before dawn.",
        "The witness apparently left before dawn.",
        "The witness supposedly left before dawn.",
        "Perhaps the witness left before dawn.",
        "The witness is believed to have left before dawn.",
    ),
)
def test_structured_claim_cannot_strengthen_qualified_evidence(
    quote: str,
) -> None:
    evidence = (
        make_result(
            {
                "file_path": r"C:\campaign\witness.md",
                "stable_id": "qualified",
                "page_number": 1,
                "text": quote,
            },
            source="hybrid",
        ),
    )
    fake_model = FakeModel(
        json.dumps(
            {
                "answer": "The witness left before dawn.",
                "claims": [
                    {
                        "text": "The witness left before dawn.",
                        "evidence_quote": quote,
                        "chunk_ids": ["qualified"],
                    }
                ],
            }
        )
    )

    with pytest.raises(UncitedAnswerError, match="not supported"):
        answer_question("When did the witness leave?", evidence, gateway=fake_model)


@pytest.mark.parametrize(
    ("excerpt", "claim"),
    (
        (
            "The witness may have left before dawn.",
            "left before dawn",
        ),
        (
            "The witness did not leave before dawn.",
            "leave before dawn",
        ),
    ),
)
def test_structured_claim_rejects_incomplete_evidence_sentence(
    excerpt: str,
    claim: str,
) -> None:
    evidence = (
        make_result(
            {
                "file_path": r"C:\campaign\witness.md",
                "stable_id": "incomplete-evidence",
                "page_number": 1,
                "text": excerpt,
            },
            source="hybrid",
        ),
    )
    fake_model = FakeModel(
        json.dumps(
            {
                "answer": claim,
                "claims": [
                    {
                        "text": claim,
                        "evidence_quote": claim,
                        "chunk_ids": ["incomplete-evidence"],
                    }
                ],
            }
        )
    )

    with pytest.raises(UncitedAnswerError, match="complete"):
        answer_question("When did the witness leave?", evidence, gateway=fake_model)


def test_structured_claim_rejects_complete_middle_evidence_sentence() -> None:
    quote = "The witness may have left before dawn."
    evidence = (
        make_result(
            {
                "file_path": r"C:\campaign\witness.md",
                "stable_id": "neighboring-evidence",
                "page_number": 1,
                "text": (
                    "A watchman arrived at midnight. "
                    f"{quote} "
                    "The eastern gate remained closed."
                ),
            },
            source="hybrid",
        ),
    )
    fake_model = FakeModel(
        json.dumps(
            {
                "answer": quote,
                "claims": [
                    {
                        "text": quote,
                        "evidence_quote": quote,
                        "chunk_ids": ["neighboring-evidence"],
                    }
                ],
            }
        )
    )

    with pytest.raises(UncitedAnswerError, match="complete"):
        answer_question("When did the witness leave?", evidence, gateway=fake_model)


def test_structured_claim_accepts_full_chunk_text_beyond_display_snippet() -> None:
    full_text = (
        f"{'Patrol notes remain unchanged. ' * 20}"
        "The eastern gate remains closed."
    )
    evidence = (
        make_result(
            {
                "file_path": r"C:\campaign\witness.md",
                "stable_id": "full-chunk-evidence",
                "page_number": 1,
                "text": full_text,
            },
            source="hybrid",
            snippet_length=80,
        ),
    )
    fake_model = FakeModel(
        json.dumps(
            {
                "answer": full_text,
                "claims": [
                    {
                        "text": full_text,
                        "evidence_quote": full_text,
                        "chunk_ids": ["full-chunk-evidence"],
                    }
                ],
            }
        )
    )

    answer = answer_question("What happened at the gate?", evidence, gateway=fake_model)

    assert answer.citations[0].excerpt != full_text
    assert answer.citations[0].text == full_text
    assert answer.claims[0].evidence_quote == full_text
    prompt = fake_model.chat_calls[0]["messages"][-1]["content"]
    assert full_text in prompt
    assert "complete full text of every cited chunk" in (
        fake_model.chat_calls[0]["messages"][0]["content"]
    )
    schema = json.dumps(fake_model.chat_calls[0]["schema"])
    assert "complete full text of every chunk" in schema


def test_structured_claim_rejects_quote_omitting_later_qualifier() -> None:
    qualifier = "However, it was not approved for production use."
    full_text = (
        "The release was approved. "
        f"{'The supporting records remain archived. ' * 10}"
        f"{qualifier}"
    )
    quote = "The release was approved."
    evidence = (
        make_result(
            {
                "file_path": r"C:\campaign\release.md",
                "stable_id": "qualified-release",
                "page_number": 1,
                "text": full_text,
            },
            source="hybrid",
        ),
    )
    fake_model = FakeModel(
        json.dumps(
            {
                "answer": quote,
                "claims": [
                    {
                        "text": quote,
                        "evidence_quote": quote,
                        "chunk_ids": ["qualified-release"],
                    }
                ],
            }
        )
    )

    assert qualifier not in evidence[0].citations[0].excerpt
    with pytest.raises(UncitedAnswerError, match="complete"):
        answer_question("Was the release approved?", evidence, gateway=fake_model)


def test_structured_claim_requires_full_chunk_for_abbreviation_split() -> None:
    full_text = "Smith et al. recorded the observation."
    evidence = (
        make_result(
            {
                "file_path": r"C:\campaign\research.md",
                "stable_id": "abbreviation-boundary",
                "page_number": 1,
                "text": full_text,
            },
            source="hybrid",
        ),
    )
    fake_model = FakeModel(
        json.dumps(
            {
                "answer": "Smith et al.",
                "claims": [
                    {
                        "text": "Smith et al.",
                        "evidence_quote": "Smith et al.",
                        "chunk_ids": ["abbreviation-boundary"],
                    }
                ],
            }
        )
    )

    with pytest.raises(UncitedAnswerError, match="complete"):
        answer_question("Who recorded the observation?", evidence, gateway=fake_model)


def test_structured_claim_rejects_unlisted_abbreviation_sentence_fragment() -> None:
    full_text = "Mt. Everest is the highest mountain above sea level."
    evidence = (
        make_result(
            {
                "file_path": r"C:\geography\mountains.md",
                "stable_id": "mount-everest",
                "page_number": 1,
                "text": full_text,
            },
            source="hybrid",
        ),
    )
    fake_model = FakeModel(
        json.dumps(
            {
                "answer": "Mt.",
                "claims": [
                    {
                        "text": "Mt.",
                        "evidence_quote": "Mt.",
                        "chunk_ids": ["mount-everest"],
                    }
                ],
            }
        )
    )

    with pytest.raises(UncitedAnswerError, match="complete"):
        answer_question("What is the highest mountain?", evidence, gateway=fake_model)


def test_structured_claim_rejects_partial_neighboring_sentences() -> None:
    evidence = (
        make_result(
            {
                "file_path": r"C:\campaign\witness.md",
                "stable_id": "partial-neighbors",
                "page_number": 1,
                "text": (
                    "A watchman arrived at midnight. "
                    "The witness may have left before dawn."
                ),
            },
            source="hybrid",
        ),
    )
    quote = "arrived at midnight. The witness may have left before dawn."
    fake_model = FakeModel(
        json.dumps(
            {
                "answer": quote,
                "claims": [
                    {
                        "text": quote,
                        "evidence_quote": quote,
                        "chunk_ids": ["partial-neighbors"],
                    }
                ],
            }
        )
    )

    with pytest.raises(UncitedAnswerError, match="complete"):
        answer_question("When did the witness leave?", evidence, gateway=fake_model)


def test_structured_claim_requires_full_excerpt_when_delimiters_are_uncertain() -> None:
    quote = "A watchman arrived. Gate status pending"
    evidence = (
        make_result(
            {
                "file_path": r"C:\campaign\witness.md",
                "stable_id": "uncertain-boundary",
                "page_number": 1,
                "text": quote,
            },
            source="hybrid",
        ),
    )
    fake_model = FakeModel(
        json.dumps(
            {
                "answer": "Gate status pending",
                "claims": [
                    {
                        "text": "Gate status pending",
                        "evidence_quote": "Gate status pending",
                        "chunk_ids": ["uncertain-boundary"],
                    }
                ],
            }
        )
    )

    with pytest.raises(UncitedAnswerError, match="complete"):
        answer_question("What is the gate status?", evidence, gateway=fake_model)


def test_structured_claim_accepts_full_excerpt_with_abbreviated_delimiter() -> None:
    quote = "Witness status: e.g. may have departed before dawn"
    evidence = (
        make_result(
            {
                "file_path": r"C:\campaign\witness.md",
                "stable_id": "abbreviated-excerpt",
                "page_number": 1,
                "text": quote,
            },
            source="hybrid",
        ),
    )
    fake_model = FakeModel(
        json.dumps(
            {
                "answer": quote,
                "claims": [
                    {
                        "text": quote,
                        "evidence_quote": quote,
                        "chunk_ids": ["abbreviated-excerpt"],
                    }
                ],
            }
        )
    )

    answer = answer_question("What happened?", evidence, gateway=fake_model)

    assert answer.claims[0].evidence_quote == quote


def test_structured_claim_cannot_omit_nontrivial_evidence_words() -> None:
    evidence = _evidence()
    fake_model = FakeModel(
        json.dumps(
            {
                "answer": "A dragon guards the temple.",
                "claims": [
                    {
                        "text": "A dragon guards the temple.",
                        "evidence_quote": "A dragon guards the flooded temple.",
                        "chunk_ids": [evidence[0].chunk_id],
                    }
                ],
            }
        )
    )

    with pytest.raises(UncitedAnswerError, match="not supported"):
        answer_question("What guards the temple?", evidence, gateway=fake_model)


def test_structured_claim_cannot_omit_evidence_symbols() -> None:
    evidence = (
        make_result(
            {
                "file_path": r"C:\campaign\budget.md",
                "stable_id": "budget",
                "page_number": 1,
                "text": "The total is $5.",
            },
            source="hybrid",
        ),
    )
    fake_model = FakeModel(
        json.dumps(
            {
                "answer": "The total is 5.",
                "claims": [
                    {
                        "text": "The total is 5.",
                        "evidence_quote": "The total is $5.",
                        "chunk_ids": ["budget"],
                    }
                ],
            }
        )
    )

    with pytest.raises(UncitedAnswerError, match="not supported"):
        answer_question("What is the total?", evidence, gateway=fake_model)


def test_structured_claim_without_verifiable_quote_is_rejected() -> None:
    evidence = _evidence()
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

    with pytest.raises(UncitedAnswerError, match="evidence quote"):
        answer_question("Find suitable encounters", evidence, gateway=fake_model)


def test_structured_claim_with_quote_not_in_cited_chunk_is_rejected() -> None:
    evidence = _evidence()
    fake_model = FakeModel(
        json.dumps(
            {
                "answer": "A dragon guards the flooded temple.",
                "claims": [
                    {
                        "text": "A dragon guards the flooded temple.",
                        "evidence_quote": "The dragon has three heads.",
                        "chunk_ids": [evidence[0].chunk_id],
                    }
                ],
            }
        )
    )

    with pytest.raises(UncitedAnswerError, match="complete full cited chunk"):
        answer_question("Find suitable encounters", evidence, gateway=fake_model)


def test_structured_claim_unrelated_to_quoted_span_is_rejected() -> None:
    evidence = _evidence()
    fake_model = FakeModel(
        json.dumps(
            {
                "answer": "The moon is a silver coin.",
                "claims": [
                    {
                        "text": "The moon is a silver coin.",
                        "evidence_quote": "A dragon guards the flooded temple.",
                        "chunk_ids": [evidence[0].chunk_id],
                    }
                ],
            }
        )
    )

    with pytest.raises(UncitedAnswerError, match="not supported"):
        answer_question("Describe the moon", evidence, gateway=fake_model)


def test_structured_answer_cannot_append_an_uncited_claim() -> None:
    evidence = _evidence()
    fake_model = FakeModel(
        json.dumps(
            {
                "answer": (
                    "A dragon guards the flooded temple. "
                    "The moon is a silver coin."
                ),
                "claims": [
                    {
                        "text": "A dragon guards the flooded temple.",
                        "evidence_quote": "A dragon guards the flooded temple.",
                        "chunk_ids": [evidence[0].chunk_id],
                    }
                ],
            }
        )
    )

    with pytest.raises(UncitedAnswerError, match="unsupported"):
        answer_question("Summarize the evidence", evidence, gateway=fake_model)


def test_structured_claim_cannot_change_a_fact_in_the_quoted_span() -> None:
    evidence = (
        make_result(
            {
                "file_path": r"C:\campaign\release.md",
                "stable_id": "release-date",
                "page_number": 1,
                "text": "The release is scheduled for July 4, 2026.",
            },
            source="hybrid",
            rank=1,
            score=0.9,
        ),
    )
    fake_model = FakeModel(
        json.dumps(
            {
                "answer": "The release is scheduled for August 2, 2026.",
                "claims": [
                    {
                        "text": "The release is scheduled for August 2, 2026.",
                        "evidence_quote": "The release is scheduled for July 4, 2026.",
                        "chunk_ids": ["release-date"],
                    }
                ],
            }
        )
    )

    with pytest.raises(UncitedAnswerError, match="not supported"):
        answer_question("When is the release?", evidence, gateway=fake_model)


@pytest.mark.parametrize(
    "claim",
    [
        "Alice hired Bob for the role.",
        "The role was given to Alice by Bob.",
    ],
)
def test_structured_claim_cannot_reverse_relationships_from_evidence(
    claim: str,
) -> None:
    evidence = (
        make_result(
            {
                "file_path": r"C:\campaign\appointment.md",
                "stable_id": "appointment",
                "page_number": 1,
                "text": "Bob hired Alice for the role.",
            },
            source="hybrid",
            rank=1,
            score=0.9,
        ),
    )
    fake_model = FakeModel(
        json.dumps(
            {
                "answer": claim,
                "claims": [
                    {
                        "text": claim,
                        "evidence_quote": "Bob hired Alice for the role.",
                        "chunk_ids": ["appointment"],
                    }
                ],
            }
        )
    )

    with pytest.raises(UncitedAnswerError, match="not supported"):
        answer_question("Who was hired?", evidence, gateway=fake_model)


@pytest.mark.parametrize(
    ("claim", "quote"),
    [
        ("Bob was hired by Alice.", "Bob hired Alice."),
        ("Bob hired Alice.", "Bob was hired by Alice."),
    ],
)
def test_structured_claim_cannot_reverse_passive_roles(
    claim: str,
    quote: str,
) -> None:
    evidence = (
        make_result(
            {
                "file_path": r"C:\campaign\appointment.md",
                "stable_id": "passive-appointment",
                "page_number": 1,
                "text": quote,
            },
            source="hybrid",
            rank=1,
            score=0.9,
        ),
    )
    fake_model = FakeModel(
        json.dumps(
            {
                "answer": claim,
                "claims": [
                    {
                        "text": claim,
                        "evidence_quote": quote,
                        "chunk_ids": ["passive-appointment"],
                    }
                ],
            }
        )
    )

    with pytest.raises(UncitedAnswerError, match="not supported"):
        answer_question("Who hired whom?", evidence, gateway=fake_model)


def test_structured_claim_cannot_replace_evidence_with_an_arbitrary_paraphrase() -> None:
    evidence = (
        make_result(
            {
                "file_path": r"C:\campaign\decision.md",
                "stable_id": "decision",
                "page_number": 1,
                "text": "The committee approved the plan.",
            },
            source="hybrid",
            rank=1,
            score=0.9,
        ),
    )
    fake_model = FakeModel(
        json.dumps(
            {
                "answer": "The plan was approved by the committee.",
                "claims": [
                    {
                        "text": "The plan was approved by the committee.",
                        "evidence_quote": "The committee approved the plan.",
                        "chunk_ids": ["decision"],
                    }
                ],
            }
        )
    )

    with pytest.raises(UncitedAnswerError, match="not supported"):
        answer_question("What happened?", evidence, gateway=fake_model)


def test_structured_claim_cannot_drop_evidence_negation() -> None:
    evidence = (
        make_result(
            {
                "file_path": r"C:\campaign\appointment.md",
                "stable_id": "negated-appointment",
                "page_number": 1,
                "text": "Bob was not responsible for the event.",
            },
            source="hybrid",
            rank=1,
            score=0.9,
        ),
    )
    fake_model = FakeModel(
        json.dumps(
            {
                "answer": "Bob was responsible for the event.",
                "claims": [
                    {
                        "text": "Bob was responsible for the event.",
                        "evidence_quote": "Bob was not responsible for the event.",
                        "chunk_ids": ["negated-appointment"],
                    }
                ],
            }
        )
    )

    with pytest.raises(UncitedAnswerError, match="not supported"):
        answer_question("Was Bob responsible?", evidence, gateway=fake_model)


def test_structured_claim_cannot_reduce_evidence_to_unrelated_name_pairs() -> None:
    evidence = (
        make_result(
            {
                "file_path": r"C:\campaign\appointment.md",
                "stable_id": "reduced-appointment",
                "page_number": 1,
                "text": "Bob hired Alice for the senior engineering role.",
            },
            source="hybrid",
            rank=1,
            score=0.9,
        ),
    )
    fake_model = FakeModel(
        json.dumps(
            {
                "answer": "Bob and Alice.",
                "claims": [
                    {
                        "text": "Bob and Alice.",
                        "evidence_quote": (
                            "Bob hired Alice for the senior engineering role."
                        ),
                        "chunk_ids": ["reduced-appointment"],
                    }
                ],
            }
        )
    )

    with pytest.raises(UncitedAnswerError, match="not supported"):
        answer_question("Who is involved?", evidence, gateway=fake_model)


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


class HybridRecordingSearch(FakeSearch):
    def __init__(
        self,
        by_query: Mapping[str, Sequence[SearchResult]],
        profile: ModelProfile,
    ) -> None:
        super().__init__(by_query)
        self.embedding_profile = profile
        self.calls: list[dict[str, Any]] = []

    def search(
        self,
        query: str,
        query_embedding: Sequence[float] | None = None,
        *,
        limit: int = 10,
        model_digest: str | None = None,
        index_version: str | None = None,
    ) -> tuple[SearchResult, ...]:
        self.calls.append(
            {
                "query": query,
                "embedding": tuple(query_embedding or ()),
                "limit": limit,
                "model_digest": model_digest,
                "index_version": index_version,
            }
        )
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


def test_search_plan_executes_exact_terms_and_paraphrases_as_hybrid() -> None:
    profile = ModelProfile(
        role="embedding",
        model="embeddinggemma-2:270m",
        digest="sha256:indexed",
        context_length=32_768,
        dimensions=3,
        capabilities=("embedding",),
    )
    exact = _result("exact", r"C:\campaign\exact.md", 1, 0.9)
    paraphrase = _result("semantic", r"C:\campaign\semantic.md", 1, 0.8)
    search = HybridRecordingSearch(
        {"flooded temple": (exact,), "temple guardians": (paraphrase,)},
        profile,
    )

    class EmbeddingGateway:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[str, ...], int | None]] = []

        async def embed(
            self,
            model: str,
            text: str | Sequence[str],
            *,
            dimensions: int | None = None,
            truncate: bool = False,
            keep_alive: float | str | None = "5m",
        ) -> tuple[tuple[float, ...], ...]:
            del truncate, keep_alive
            queries = (text,) if isinstance(text, str) else tuple(text)
            self.calls.append((model, queries, dimensions))
            return tuple((float(index), 1.0, 0.5) for index, _ in enumerate(queries))

    gateway = EmbeddingGateway()
    plan = SearchPlan(
        question="Find temple guardians",
        intent_names=("encounters",),
        exact_terms=("flooded temple",),
        paraphrase_queries=("temple guardians", "flooded temple"),
        evidence_categories=("encounters",),
    )

    evidence = asyncio.run(
        execute_search_plan_async(
            plan,
            search,
            embedding_gateway=gateway,
            per_query_limit=4,
            limit=3,
        )
    )

    assert [result.chunk_id for result in evidence] == ["exact", "semantic"]
    assert gateway.calls == [
        (
            "embeddinggemma-2:270m",
            ("flooded temple", "temple guardians"),
            3,
        )
    ]
    assert [call["query"] for call in search.calls] == [
        "flooded temple",
        "temple guardians",
    ]
    assert all(call["embedding"] for call in search.calls)
    assert all(call["model_digest"] == "sha256:indexed" for call in search.calls)
    assert all(
        call["index_version"] == profile.default_index_version
        for call in search.calls
    )


def test_search_plan_applies_typed_filters_and_postfilters_exclusions() -> None:
    wanted = make_result(
        {
            "file_path": r"C:\reports\atlas.pdf",
            "stable_id": "wanted",
            "page_number": 1,
            "text": "The public atlas report.",
        },
        source="hybrid",
        rank=1,
        score=0.9,
    )
    excluded = make_result(
        {
            "file_path": r"C:\reports\classified.pdf",
            "stable_id": "excluded",
            "page_number": 2,
            "text": "The classified atlas report.",
        },
        source="hybrid",
        rank=2,
        score=0.8,
    )

    class FilterRecordingSearch(FakeSearch):
        def __init__(self) -> None:
            super().__init__({"atlas": (wanted, excluded)})
            self.filters: list[object] = []

        def search(
            self,
            query: str,
            query_embedding: Sequence[float] | None = None,
            *,
            limit: int = 10,
            filters: object | None = None,
            model_digest: str | None = None,
            index_version: str | None = None,
        ) -> tuple[SearchResult, ...]:
            del query_embedding, model_digest, index_version
            self.filters.append(filters)
            return super().search(query, limit=limit)

    search = FilterRecordingSearch()
    plan = SearchPlan(
        question="Find the atlas report",
        intent_names=("reports",),
        exact_terms=("atlas",),
        paraphrase_queries=("atlas",),
        filters=("path: reports", "tag: work", "file type: pdf"),
        normalized_constraints=("tag: finance",),
        exclusions=("classified",),
        evidence_categories=("reports",),
    )

    evidence = execute_search_plan(plan, search)

    assert [result.chunk_id for result in evidence] == ["wanted"]
    assert len(search.filters) == 1
    assert search.filters[0] == SearchFilters(
        tags=("work", "finance"),
        path="reports",
        file_types=("pdf",),
    )


def test_search_plan_keeps_multiple_allowed_paths_as_filters() -> None:
    result = _result("shared", r"C:\reports\wanted.md", 1, 0.9)
    search_calls: list[tuple[str, object | None]] = []

    class MultiPathRecordingSearch(FakeSearch):
        def __init__(self) -> None:
            super().__init__(
                {
                    "atlas": (result,),
                    "annual report": (result,),
                }
            )

        def search(
            self,
            query: str,
            query_embedding: Sequence[float] | None = None,
            *,
            limit: int = 10,
            filters: object | None = None,
            model_digest: str | None = None,
            index_version: str | None = None,
        ) -> tuple[SearchResult, ...]:
            del query_embedding, model_digest, index_version
            search_calls.append((query, filters))
            return super().search(query, limit=limit)

    search = MultiPathRecordingSearch()
    plan = SearchPlan(
        question="Find reports in both locations",
        intent_names=("reports",),
        exact_terms=("atlas",),
        paraphrase_queries=("annual report",),
        filters=("path: reports/one", "path: reports/two"),
        evidence_categories=("reports",),
    )

    evidence = execute_search_plan(plan, search)

    assert [candidate.chunk_id for candidate in evidence] == ["shared"]
    assert search_calls == [
        (
            "atlas",
            SearchFilters(paths=("reports/one", "reports/two")),
        ),
        (
            "annual report",
            SearchFilters(paths=("reports/one", "reports/two")),
        ),
    ]


def test_optional_rerank_uses_bounded_structured_chat() -> None:
    evidence = (
        _result("first", r"C:\campaign\first.md", 1, 0.9),
        _result("second", r"C:\campaign\second.md", 1, 0.8),
    )
    fake_model = FakeModel(
        json.dumps(
            {
                "answer": "Evidence from second.",
                "claims": [
                    {
                        "text": "Evidence from second.",
                        "evidence_quote": "Evidence from second.",
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
                "answer": "Evidence from chunk-0.",
                "claims": [
                    {
                        "text": "Evidence from chunk-0.",
                        "evidence_quote": "Evidence from chunk-0.",
                        "chunk_ids": ["chunk-0"],
                    }
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
                "answer": "Evidence from budgeted-0.",
                "claims": [
                    {
                        "text": "Evidence from budgeted-0.",
                        "evidence_quote": "Evidence from budgeted-0.",
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


def test_rag_stream_emits_gateway_fragments_before_answer_is_complete() -> None:
    evidence = _evidence()
    claim = "A dragon guards the flooded temple."

    class SlowStreamingGateway:
        def __init__(self) -> None:
            self.fragments: list[str] = []

        async def chat_stream(
            self,
            model: str,
            *,
            messages: Sequence[Mapping[str, Any]],
            schema: Mapping[str, Any] | None = None,
            keep_alive: float | str | None = "5m",
        ):
            del model, messages, schema, keep_alive
            response = json.dumps(
                {
                    "answer": claim,
                    "claims": [
                        {
                            "text": claim,
                            "evidence_quote": claim,
                            "chunk_ids": [evidence[0].chunk_id],
                        }
                    ],
                }
            )
            split = (0, 18, 41, len(response))
            for start, end in zip(split, split[1:]):
                await asyncio.sleep(0.01)
                yield response[start:end]

    gateway = SlowStreamingGateway()
    service = RagService(gateway=gateway)

    async def run() -> tuple[list[str], object]:
        fragments: list[str] = []
        async for fragment in service.answer_question_stream_async(
            "Find suitable encounters",
            evidence,
            session_id="conversation-stream",
        ):
            fragments.append(fragment)
            await asyncio.sleep(0)
        return fragments, service.get_session_answer("conversation-stream")

    fragments, answer = asyncio.run(run())

    assert "".join(fragments) == claim
    assert answer.text == claim
    assert answer.citations == evidence[0].citations


def test_follow_up_reuses_evidence_for_changed_question_and_same_retrieval_context() -> None:
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
                        "evidence_quote": "A dragon guards the flooded temple.",
                        "chunk_ids": ["dragon-encounter"],
                    }
                ],
            }
        )
    )
    service = RagService(search=search, gateway=model, planner=FakePlanner())

    service.answer_question(
        "initial",
        markdown_context="# Encounter",
        session_id="conversation-1",
    )
    service.answer_question(
        "What guards it?",
        markdown_context="# Encounter",
        session_id="conversation-1",
    )

    assert search.queries == ["initial"]
    assert len(model.chat_calls) == 2
    assert "A dragon guards the flooded temple." in (
        model.chat_calls[1]["messages"][-2]["content"]
    )
    assert "# Encounter" in model.chat_calls[0]["messages"][-1]["content"]


def test_session_retrieval_context_change_invalidates_evidence_and_history() -> None:
    first = _result("first", r"C:\campaign\first.md", 1, 0.9)
    second = _result("second", r"C:\campaign\second.md", 1, 0.9)
    search = FakeSearch({"initial": (first,), "later": (second,)})

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

    class ContextModel(FakeModel):
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
            prompt = messages[-1]["content"]
            chunk_id = "first" if '"first"' in prompt else "second"
            return json.dumps(
                {
                    "answer": f"Evidence from {chunk_id}.",
                    "claims": [
                        {
                            "text": f"Evidence from {chunk_id}.",
                            "evidence_quote": f"Evidence from {chunk_id}.",
                            "chunk_ids": [chunk_id],
                        }
                    ],
                }
            )

    model = ContextModel("")
    service = RagService(search=search, gateway=model, planner=FakePlanner())

    service.answer_question(
        "initial",
        markdown_context="# First",
        session_id="conversation-1",
    )
    service.answer_question(
        "later",
        markdown_context="# Second",
        session_id="conversation-1",
    )

    assert search.queries == ["initial", "later"]
    assert "first" not in json.dumps(model.chat_calls[-1]["messages"][:-1])


def test_session_index_profile_change_invalidates_reused_evidence() -> None:
    profile_v1 = ModelProfile(
        role="embedding",
        model="embeddinggemma-2:270m",
        digest="sha256:index-v1",
        context_length=32_768,
        dimensions=3,
        capabilities=("embedding",),
    )
    profile_v2 = profile_v1.model_copy(update={"digest": "sha256:index-v2"})
    first = _result("first", r"C:\campaign\first.md", 1, 0.9)
    second = _result("second", r"C:\campaign\second.md", 1, 0.9)
    search = HybridRecordingSearch(
        {"initial": (first,), "follow-up": (second,)},
        profile_v1,
    )

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

    class IndexedGateway(FakeModel):
        async def embed(
            self,
            model: str,
            text: str | Sequence[str],
            *,
            dimensions: int | None = None,
            truncate: bool = False,
            keep_alive: float | str | None = "5m",
        ) -> tuple[tuple[float, ...], ...]:
            del model, dimensions, truncate, keep_alive
            queries = (text,) if isinstance(text, str) else tuple(text)
            return tuple((0.1, 0.2, 0.3) for _ in queries)

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
            prompt = messages[-1]["content"]
            chunk_id = "first" if '"first"' in prompt else "second"
            return json.dumps(
                {
                    "answer": f"Evidence from {chunk_id}.",
                    "claims": [
                        {
                            "text": f"Evidence from {chunk_id}.",
                            "evidence_quote": f"Evidence from {chunk_id}.",
                            "chunk_ids": [chunk_id],
                        }
                    ],
                }
            )

    model = IndexedGateway("")
    service = RagService(search=search, gateway=model, planner=FakePlanner())

    service.answer_question("initial", session_id="conversation-1")
    search.embedding_profile = profile_v2
    service.answer_question("follow-up", session_id="conversation-1")

    assert search.queries == ["initial", "follow-up"]
    assert search.calls[-1]["model_digest"] == "sha256:index-v2"
    assert search.calls[-1]["index_version"] == profile_v2.default_index_version
    assert "first" not in json.dumps(model.chat_calls[-1]["messages"][:-1])


def test_session_index_epoch_change_invalidates_reused_evidence() -> None:
    first = _result("first", r"C:\campaign\first.md", 1, 0.9)
    second = _result("second", r"C:\campaign\second.md", 1, 0.9)

    class EpochSearch(FakeSearch):
        def __init__(self) -> None:
            super().__init__({"initial": (first,), "follow-up": (second,)})
            self.index_fingerprint = "sha256:corpus-v1"

    search = EpochSearch()

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

    class EpochModel(FakeModel):
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
            prompt = messages[-1]["content"]
            chunk_id = "first" if '"first"' in prompt else "second"
            return json.dumps(
                {
                    "answer": f"Evidence from {chunk_id}.",
                    "claims": [
                        {
                            "text": f"Evidence from {chunk_id}.",
                            "evidence_quote": f"Evidence from {chunk_id}.",
                            "chunk_ids": [chunk_id],
                        }
                    ],
                }
            )

    model = EpochModel("")
    service = RagService(search=search, gateway=model, planner=FakePlanner())

    service.answer_question("initial", session_id="conversation-1")
    search.index_fingerprint = "sha256:corpus-v2"
    service.answer_question("follow-up", session_id="conversation-1")

    assert search.queries == ["initial", "follow-up"]
    assert "first" not in json.dumps(model.chat_calls[-1]["messages"][:-1])


def test_explicit_retrieval_question_change_invalidates_reused_session() -> None:
    first = _result("first", r"C:\campaign\first.md", 1, 0.9)
    second = _result("second", r"C:\campaign\second.md", 1, 0.9)
    search = FakeSearch({"initial": (first,), "replacement": (second,)})

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
                "answer": "Evidence from first.",
                "claims": [
                    {
                        "text": "Evidence from first.",
                        "evidence_quote": "Evidence from first.",
                        "chunk_ids": ["first"],
                    }
                ],
            }
        )
    )
    service = RagService(search=search, gateway=model, planner=FakePlanner())

    service.answer_question(
        "What is relevant?",
        retrieval_question="initial",
        session_id="conversation-1",
    )
    model.response = json.dumps(
        {
            "answer": "Evidence from second.",
            "claims": [
                {
                    "text": "Evidence from second.",
                    "evidence_quote": "Evidence from second.",
                    "chunk_ids": ["second"],
                }
            ],
        }
    )
    service.answer_question(
        "What changed?",
        retrieval_question="replacement",
        session_id="conversation-1",
    )

    assert search.queries == ["initial", "replacement"]
    assert "first" not in json.dumps(model.chat_calls[-1]["messages"][:-1])


def test_concurrent_follow_ups_cannot_mix_session_state() -> None:
    evidence = _result("shared", r"C:\campaign\shared.md", 1, 0.9)
    search = FakeSearch({"initial": (evidence,)})

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

    class SerialModel(FakeModel):
        def __init__(self) -> None:
            super().__init__("")
            self.active = 0
            self.overlapped = False

        async def chat(
            self,
            model: str,
            *,
            messages: Sequence[Mapping[str, Any]],
            schema: Mapping[str, Any] | None = None,
            keep_alive: float | str | None = "5m",
        ) -> str:
            self.active += 1
            self.overlapped = self.overlapped or self.active > 1
            await asyncio.sleep(0.01)
            self.chat_calls.append(
                {
                    "model": model,
                    "messages": messages,
                    "schema": schema,
                    "keep_alive": keep_alive,
                }
            )
            self.active -= 1
            return json.dumps(
                {
                    "answer": "Evidence from shared.",
                    "claims": [
                        {
                            "text": "Evidence from shared.",
                            "evidence_quote": "Evidence from shared.",
                            "chunk_ids": ["shared"],
                        }
                    ],
                }
            )

    model = SerialModel()
    service = RagService(search=search, gateway=model, planner=FakePlanner())

    async def run() -> None:
        await asyncio.gather(
            service.answer_question_async(
                "initial",
                session_id="conversation-1",
            ),
            service.answer_question_async(
                "What follows?",
                session_id="conversation-1",
            ),
        )

    asyncio.run(run())

    assert search.queries == ["initial"]
    assert model.overlapped is False
    assert len(model.chat_calls) == 2
    assert any(
        message.get("content") == "Evidence from shared."
        for message in model.chat_calls[1]["messages"]
    )


def test_service_accepts_direct_evidence_as_second_argument() -> None:
    model = FakeModel(
        json.dumps(
            {
                "answer": "A dragon guards the flooded temple.",
                "claims": [
                    {
                        "text": "A dragon guards the flooded temple.",
                        "evidence_quote": "A dragon guards the flooded temple.",
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
