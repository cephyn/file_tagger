from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from file_tagger.adapters.ollama.gateway import OllamaGateway
from file_tagger.domain.contracts import TagDecision
from file_tagger.domain.errors import ModelIncompatibleError
from file_tagger.tagging.decision_service import TagDecisionResult, decide_tags


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
    probabilities = {"accept": 0.1, "reject": 0.1, "review": 0.1}
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


def test_candidate_named_answers_are_validated_in_candidate_order() -> None:
    class CandidateNamedGateway:
        async def system_one(self, model: str, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(
                answers={
                    "invoice": {"type": "noul", "noul": 0.1},
                    "dungeon": {"type": "noul", "noul": 0.9},
                }
            )

    result = decide_tags(
        "encounter table",
        ["dungeon", "invoice"],
        gateway=CandidateNamedGateway(),
    )

    assert [(item.tag, item.decision) for item in result.decisions] == [
        ("dungeon", "accept"),
        ("invoice", "reject"),
    ]


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


def test_batch_tag_suggester_uses_typed_local_decision_service(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from file_tagger import models, tag_suggestion

    source = tmp_path / "ruin.txt"
    source.write_text("encounter table for a flooded ruin", encoding="utf-8")

    class Query:
        def all(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name="dungeon"), SimpleNamespace(name="invoice")]

    class Session:
        closed = False

        def query(self, model: Any) -> Query:
            assert model is models.Tag
            return Query()

        def close(self) -> None:
            self.closed = True

    session = Session()
    monkeypatch.setattr(models, "init_db", lambda: session)
    calls: list[dict[str, Any]] = []

    def fake_decide_tags(
        document_state: str,
        candidates: list[str],
        **kwargs: Any,
    ) -> TagDecisionResult:
        calls.append(
            {
                "state": document_state,
                "candidates": candidates,
                **kwargs,
            }
        )
        return TagDecisionResult(
            decisions=(
                TagDecision(
                    tag="dungeon",
                    decision="accept",
                    confidence=0.9,
                ),
            )
        )

    monkeypatch.setattr(
        tag_suggestion,
        "decide_tags",
        fake_decide_tags,
        raising=False,
    )

    class CloudConfigMustNotBeRead:
        def __getattr__(self, name: str) -> Any:
            raise AssertionError(f"legacy provider configuration read: {name}")

    suggestions = tag_suggestion.TagSuggester(
        CloudConfigMustNotBeRead()
    ).suggest_tags_for_file(str(source))

    assert suggestions == {"dungeon": 0.9}
    assert calls[0]["candidates"] == ["dungeon", "invoice"]
    assert "flooded ruin" in calls[0]["state"]
    assert session.closed
