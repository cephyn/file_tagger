from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from file_tagger.adapters.ollama.gateway import OllamaGateway
from file_tagger.domain.contracts import TagDecision
from file_tagger.domain.errors import GeminiUnavailableError, ModelIncompatibleError
from file_tagger.tagging.cache import CachedTagRecommendations
from file_tagger.tagging.decision_service import decide_tags


class FakeGateway:
    def __init__(self, answers: list[dict[str, Any]]) -> None:
        self.answers = answers
        self.calls: list[dict[str, Any]] = []

    async def system_one(
        self,
        model: str,
        *,
        state: Any,
        questions: Mapping[str, Any],
        keep_alive: float | str | None = "5m",
    ) -> SimpleNamespace:
        self.calls.append(
            {
                "model": model,
                "state": state,
                "questions": questions,
                "keep_alive": keep_alive,
            }
        )
        names = list(questions)
        return SimpleNamespace(
            answers={
                name: answer
                for name, answer in zip(names, self.answers, strict=True)
            }
        )


def choice(decision: str, confidence: float) -> dict[str, Any]:
    remainder = (1.0 - confidence) / 2.0
    probabilities = {"accept": remainder, "reject": remainder, "review": remainder}
    probabilities[decision] = confidence
    return {
        "type": "choice",
        "choice": decision,
        "probabilities": probabilities,
        "confidence": confidence,
    }


def test_tag_decision_uses_one_installed_model_request() -> None:
    gateway = FakeGateway(
        [choice("accept", 0.92), choice("reject", 0.86), choice("reject", 0.9)]
    )

    result = decide_tags(
        document_state="encounter table for a flooded ruin",
        candidates=["dungeon", "rpg", "invoice"],
        model="nimble:9b",
        gateway=gateway,
    )

    assert [decision.tag for decision in result.accepted] == ["dungeon"]
    assert 0.0 <= result.accepted[0].confidence <= 1.0
    assert len(gateway.calls) == 1
    assert gateway.calls[0]["model"] == "nimble:9b"
    assert len(gateway.calls[0]["questions"]) == 3


def test_gemini_decision_model_uses_injected_gateway() -> None:
    gateway = FakeGateway([choice("accept", 0.93)])

    result = decide_tags(
        document_state="encounter table for a flooded ruin",
        candidates=["dungeon"],
        model="gemini-decision-1",
        gateway=gateway,
    )

    assert [decision.tag for decision in result.accepted] == ["dungeon"]
    assert gateway.calls[0]["model"] == "gemini-decision-1"


def test_exact_gemini_decision_model_without_gateway_raises_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import file_tagger.tagging.decision_service as decision_service

    def unexpected_ollama_gateway() -> None:
        raise AssertionError("OllamaGateway must not be constructed")

    monkeypatch.setattr(
        decision_service,
        "OllamaGateway",
        unexpected_ollama_gateway,
    )

    with pytest.raises(GeminiUnavailableError):
        decide_tags(
            "encounter table",
            ["dungeon"],
            model="gemini-decision-1",
        )


@pytest.mark.parametrize("model", ["nimble:9b", "tev1:4b", "tev1:0.8b"])
def test_supported_decision_models_share_the_contract(model: str) -> None:
    gateway = FakeGateway([choice("accept", 0.75)])

    result = decide_tags(
        "encounter table",
        ["dungeon"],
        model=model,
        gateway=gateway,
    )

    assert result.accepted[0].tag == "dungeon"
    assert gateway.calls[0]["model"] == model


def test_choice_noul_and_score_answers_become_tag_decisions() -> None:
    gateway = FakeGateway(
        [
            {
                "type": "choice",
                "choice": "accept",
                "probabilities": {
                    "accept": 0.81,
                    "reject": 0.12,
                    "review": 0.07,
                },
                "confidence": 0.81,
            },
            {"type": "noul", "noul": 0.2},
            {
                "type": "score",
                "score": 1.7,
                "legend": {"0": "irrelevant", "1": "uncertain", "2": "relevant"},
                "probabilities": {"0": 0.05, "1": 0.2, "2": 0.75},
                "confidence": 0.75,
            },
        ]
    )

    result = decide_tags(
        "flooded ruin",
        ["dungeon", "invoice", "rpg"],
        gateway=gateway,
    )

    assert [(item.tag, item.decision) for item in result.decisions] == [
        ("dungeon", "accept"),
        ("invoice", "reject"),
        ("rpg", "accept"),
    ]
    assert all(0.0 <= item.confidence <= 1.0 for item in result.decisions)


def test_candidate_named_answer_keys_are_rejected() -> None:
    class CandidateNamedGateway:
        async def system_one(self, model: str, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(
                answers={
                    "invoice": {"type": "noul", "noul": 0.1},
                    "dungeon": {"type": "noul", "noul": 0.9},
                }
            )

    with pytest.raises(ValueError, match="unknown question"):
        decide_tags(
            "encounter table",
            ["dungeon", "invoice"],
            gateway=CandidateNamedGateway(),
        )


def test_missing_answers_fail_visibly() -> None:
    class MissingGateway:
        async def system_one(self, model: str, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(answers={})

    with pytest.raises(ValueError, match="Missing"):
        decide_tags("document", ["dungeon"], gateway=MissingGateway())


def test_unknown_candidates_fail_visibly() -> None:
    gateway = FakeGateway(
        [
            {
                "type": "choice",
                "choice": "unknown",
                "probabilities": {"unknown": 1.0},
                "confidence": 1.0,
            }
        ]
    )

    with pytest.raises(ValueError, match="unknown"):
        decide_tags("document", ["dungeon"], gateway=gateway)


@pytest.mark.parametrize(
    "answer, message",
    [
        (
            {
                "type": "choice",
                "choice": "accept",
                "probabilities": {"accept": 1.0},
                "confidence": 1.0,
            },
            "keys",
        ),
        (
            {
                "type": "choice",
                "choice": "accept",
                "probabilities": {
                    "accept": 0.8,
                    "reject": 0.2,
                    "review": 0.1,
                },
                "confidence": 0.8,
            },
            "sum",
        ),
        (
            {
                "type": "choice",
                "choice": "reject",
                "probabilities": {
                    "accept": 0.8,
                    "reject": 0.1,
                    "review": 0.1,
                },
                "confidence": 0.1,
            },
            "highest",
        ),
        (
            {
                "type": "choice",
                "choice": "accept",
                "probabilities": {
                    "accept": 0.8,
                    "reject": 0.1,
                    "review": 0.1,
                },
                "confidence": 0.7,
            },
            "confidence",
        ),
        (
            {
                "type": "score",
                "score": 1,
                "legend": {"0": "no", "1": "yes"},
                "probabilities": {"0": 0.2, "unknown": 0.8},
                "confidence": 0.8,
            },
            "keys",
        ),
        (
            {
                "type": "score",
                "score": 1,
                "legend": {"0": "no", "1": "yes"},
                "probabilities": {"0": 0.2, "1": 0.7},
                "confidence": 0.7,
            },
            "sum",
        ),
    ],
)
def test_invalid_system_one_distributions_fail(
    answer: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        decide_tags("document", ["dungeon"], gateway=FakeGateway([answer]))


def test_choice_probability_tolerance_accepts_rounding_noise() -> None:
    answer = {
        "type": "choice",
        "choice": "accept",
        "probabilities": {"accept": 0.8, "reject": 0.1, "review": 0.1},
        "confidence": 0.8000005,
    }

    result = decide_tags("document", ["dungeon"], gateway=FakeGateway([answer]))

    assert result.accepted[0].confidence == pytest.approx(0.8, abs=1e-6)


def test_score_must_match_weighted_level_probabilities() -> None:
    answer = {
        "type": "score",
        "score": 0,
        "legend": {"0": "no", "1": "yes"},
        "probabilities": {"0": 0.2, "1": 0.8},
        "confidence": 0.8,
    }

    with pytest.raises(ValueError, match="weighted"):
        decide_tags("document", ["dungeon"], gateway=FakeGateway([answer]))


def test_score_tolerance_accepts_weighted_probability_rounding() -> None:
    answer = {
        "type": "score",
        "score": 0.8000005,
        "legend": {"0": "no", "1": "yes"},
        "probabilities": {"0": 0.2, "1": 0.8},
        "confidence": 0.8,
    }

    result = decide_tags(
        "document",
        ["dungeon"],
        gateway=FakeGateway([answer]),
    )

    assert result.accepted[0].tag == "dungeon"


def test_original_candidate_identity_is_preserved_after_prompt_normalization() -> None:
    gateway = FakeGateway([choice("accept", 1.0)])

    result = decide_tags(
        "document",
        ["  Dungeon  "],
        gateway=gateway,
    )

    assert result.accepted[0].tag == "  Dungeon  "
    instructions = gateway.calls[0]["questions"]["candidate_000"]["instructions"]
    assert "'Dungeon'" in instructions
    assert "'  Dungeon  '" not in instructions


def test_generated_question_id_does_not_collide_with_candidate_name() -> None:
    gateway = FakeGateway([choice("accept", 1.0), choice("reject", 1.0)])

    result = decide_tags(
        "document",
        ["candidate_001", "candidate_000"],
        gateway=gateway,
    )

    assert [(item.tag, item.decision) for item in result.decisions] == [
        ("candidate_001", "accept"),
        ("candidate_000", "reject"),
    ]


@pytest.mark.parametrize(
    "answer",
    [
        {"type": "noul", "noul": 1.1},
        {
            "type": "choice",
            "choice": "accept",
            "probabilities": {"accept": 1.1},
            "confidence": 1.0,
        },
        {
            "type": "score",
            "score": 0.5,
            "legend": {"0": "no", "1": "yes"},
            "probabilities": {"0": 1.2},
            "confidence": 0.5,
        },
    ],
)
def test_out_of_range_probabilities_fail_visibly(answer: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="probabil|confidence"):
        decide_tags("document", ["dungeon"], gateway=FakeGateway([answer]))


def test_oversized_state_is_rejected_before_model_call() -> None:
    gateway = FakeGateway([choice("accept", 1.0)])

    with pytest.raises(ValueError, match="too large"):
        decide_tags("x" * (64 * 1024), ["dungeon"], gateway=gateway)

    assert gateway.calls == []


def test_more_than_supported_question_count_is_rejected() -> None:
    gateway = FakeGateway([])

    with pytest.raises(ValueError, match="64"):
        decide_tags(
            "document",
            [f"candidate-{index}" for index in range(65)],
            gateway=gateway,
        )

    assert gateway.calls == []


def test_cloud_model_tags_are_rejected_by_gateway() -> None:
    gateway = OllamaGateway(client=object())

    with pytest.raises(ModelIncompatibleError, match="Cloud"):
        decide_tags(
            "document",
            ["dungeon"],
            model="nimble:cloud",
            gateway=gateway,
        )


def test_batch_tag_suggester_routes_through_candidate_service(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from file_tagger import tag_suggestion

    source = tmp_path / "ruin.txt"
    source.write_text("encounter table for a flooded ruin", encoding="utf-8")
    calls: list[dict[str, Any]] = []

    class CandidateService:
        def recommend(
            self,
            document_state: str,
            *,
            file_hash: str,
            extraction_version: str,
            file_path: str,
            force_refresh: bool,
        ) -> CachedTagRecommendations:
            calls.append(
                {
                    "state": document_state,
                    "file_hash": file_hash,
                    "extraction_version": extraction_version,
                    "file_path": file_path,
                    "force_refresh": force_refresh,
                }
            )
            return CachedTagRecommendations(
                file_path=file_path,
                file_hash=file_hash,
                document_state_digest="document-state",
                candidate_model_digest="candidate",
                decision_model_digest="decision",
                decisions=(
                TagDecision(
                    tag="dungeon",
                    decision="accept",
                    confidence=0.9,
                ),
                    TagDecision(
                        tag="invoice",
                        decision="reject",
                        confidence=0.9,
                    ),
                ),
            )

    class CloudConfigMustNotBeRead:
        def __getattr__(self, name: str) -> Any:
            raise AssertionError(f"legacy provider configuration read: {name}")

    suggestions = tag_suggestion.TagSuggester(
        CloudConfigMustNotBeRead(),
        candidate_service=CandidateService(),
    ).suggest_tags_for_file(str(source), force_refresh=True)

    assert suggestions == {"dungeon": 0.9}
    assert calls[0]["force_refresh"] is True
    assert "flooded ruin" in calls[0]["state"]
    assert calls[0]["file_path"] == str(source.resolve())


def test_tag_suggestion_dialog_splits_typed_existing_and_new_decisions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    qtbot: Any,
) -> None:
    from file_tagger import tag_suggestion

    source = tmp_path / "ruin.txt"
    source.write_text("encounter table for a flooded ruin", encoding="utf-8")
    calls: list[bool] = []

    def recommend(
        self: Any,
        file_path: str,
        *,
        force_refresh: bool = False,
    ) -> CachedTagRecommendations:
        calls.append(force_refresh)
        return CachedTagRecommendations(
            file_path=file_path,
            file_hash="hash",
            document_state_digest="document-state",
            candidate_model_digest="candidate",
            decision_model_digest="decision",
            decisions=(
                TagDecision(tag="dungeon", decision="accept", confidence=0.9),
                TagDecision(
                    tag="flooded ruin",
                    decision="review",
                    confidence=0.6,
                    is_new=True,
                ),
                TagDecision(tag="invoice", decision="reject", confidence=0.9),
            ),
        )

    monkeypatch.setattr(
        tag_suggestion.TagSuggester,
        "recommend_tags_for_file",
        recommend,
    )

    dialog = tag_suggestion.TagSuggestionDialog(
        SimpleNamespace(),
        SimpleNamespace(),
        str(source),
    )
    qtbot.addWidget(dialog)

    assert dialog.existing_list.itemWidget(
        dialog.existing_list.item(0)
    ).tag_name == "dungeon"
    assert dialog.new_list.itemWidget(dialog.new_list.item(0)).tag_name == "flooded ruin"
    dialog.analyze_file(force_refresh=True)
    assert calls == [False, True]


def test_production_legacy_cache_persists_typed_recommendations() -> None:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from file_tagger import models, tag_suggestion

    engine = create_engine("sqlite:///:memory:")
    models.Base.metadata.create_all(engine)
    with Session(engine) as session:
        cache = tag_suggestion._LegacyTagRecommendationCache(session)
        stored = CachedTagRecommendations(
            file_path=r"C:\docs\ruin.md",
            file_hash="content-hash",
            document_state_digest="document-state",
            extraction_version="tag-suggestion-state-v1",
            candidate_model_digest="candidate-digest",
            decision_model_digest="decision-digest",
            candidate_set_digest="candidate-set",
            max_candidates=24,
            confirmation_threshold=0.7,
            decisions=(
                TagDecision(tag="flooded ruin", decision="review", confidence=0.6),
            ),
        )

        cache.store(stored)
        session.commit()

    with Session(engine) as session:
        cached = tag_suggestion._LegacyTagRecommendationCache(session).lookup(
            "content-hash",
            file_path=r"C:\docs\ruin.md",
            document_state_digest="document-state",
            extraction_version="tag-suggestion-state-v1",
            candidate_model_digest="candidate-digest",
            decision_model_digest="decision-digest",
            candidate_set_digest="candidate-set",
            max_candidates=24,
            confirmation_threshold=0.7,
        )

    assert cached == stored


def test_document_state_raw_reads_plain_text_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from file_tagger import tag_suggestion

    source = tmp_path / "notes.txt"
    source.write_text("plain text remains useful", encoding="utf-8")

    def unexpected_extraction(file_path: str, **kwargs: Any) -> str:
        del file_path, kwargs
        raise AssertionError("plain text must not be sent to Docling")

    monkeypatch.setattr(
        tag_suggestion,
        "extract_document",
        unexpected_extraction,
    )

    state = tag_suggestion._document_state(str(source))

    assert "plain text remains useful" in state


@pytest.mark.parametrize("suffix", [".csv", ".html", ".json", ".pdf"])
def test_document_state_extracts_structured_and_non_text_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    suffix: str,
) -> None:
    from file_tagger import tag_suggestion

    source = tmp_path / f"report{suffix}"
    source.write_bytes(b"RAW_BINARY_SHOULD_NOT_APPEAR")
    calls: list[str] = []

    def extract(file_path: str, **kwargs: Any):
        del kwargs
        calls.append(file_path)
        return SimpleNamespace(full_markdown="# Extracted document")

    monkeypatch.setattr(tag_suggestion, "extract_document", extract)

    state = tag_suggestion._document_state(str(source))

    assert calls == [str(source)]
    assert "# Extracted document" in state
    assert "RAW_BINARY_SHOULD_NOT_APPEAR" not in state


def test_document_state_is_truncated_safely(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from file_tagger import tag_suggestion

    source = tmp_path / "large.pdf"
    source.write_bytes(b"%PDF")
    extracted = "Filename: large.pdf\n\n" + ("😀" * (32 * 1024))
    monkeypatch.setattr(
        tag_suggestion,
        "extract_document",
        lambda *args, **kwargs: SimpleNamespace(full_markdown=extracted),
    )

    state = tag_suggestion._document_state(str(source))

    assert len(state.encode("utf-8")) <= (
        tag_suggestion.MAX_DOCUMENT_STATE_SAMPLE_BYTES
    )
    assert state.endswith("[truncated]")
    assert "😀" in state
