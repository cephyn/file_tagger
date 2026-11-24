from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from sqlalchemy import inspect

from file_tagger.domain.contracts import ModelProfile, TagDecision
from file_tagger.infrastructure.database import create_session_factory, upgrade_database
from file_tagger.persistence.models import File
from file_tagger.tagging.cache import (
    CachedTagRecommendations,
    DatabaseTagRecommendationCache,
    InMemoryTagRecommendationCache,
)
from file_tagger.tagging.candidate_service import recommend_tags


class FakeSearch:
    def __init__(self, tags: Sequence[str]) -> None:
        self.tags = tuple(tags)
        self.calls: list[dict[str, Any]] = []

    def search(
        self,
        query: str,
        query_embedding: Sequence[float] | None = None,
        **kwargs: Any,
    ) -> tuple[SimpleNamespace, ...]:
        self.calls.append(
            {
                "query": query,
                "query_embedding": query_embedding,
                **kwargs,
            }
        )
        return (SimpleNamespace(tags=self.tags),)


class FakeGateway:
    def __init__(self, generated: Sequence[str]) -> None:
        self.generated = tuple(generated)
        self.chat_calls: list[dict[str, Any]] = []
        self.decision_calls: list[dict[str, Any]] = []

    async def model_profile(self, role: str, model: str) -> ModelProfile:
        return ModelProfile(
            role=role,  # type: ignore[arg-type]
            model=model,
            digest=f"{role}-digest",
            context_length=32_768,
            capabilities=("completion",) if role == "generation" else ("systemone",),
        )

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
        return json.dumps({"candidates": list(self.generated)})

    async def system_one(
        self,
        model: str,
        *,
        state: Any,
        questions: Mapping[str, Any],
        keep_alive: float | str | None = "5m",
    ) -> SimpleNamespace:
        self.decision_calls.append(
            {
                "model": model,
                "state": state,
                "questions": questions,
                "keep_alive": keep_alive,
            }
        )
        answers = {}
        for index, name in enumerate(questions):
            confidence = 0.9 if index == 0 else 0.55
            answers[name] = {
                "type": "choice",
                "choice": "accept",
                "probabilities": {
                    "accept": confidence,
                    "review": (1.0 - confidence) / 2.0,
                    "reject": (1.0 - confidence) / 2.0,
                },
                "confidence": confidence,
            }
        return SimpleNamespace(answers=answers)


def test_model_digest_change_invalidates_cached_tags() -> None:
    cache = InMemoryTagRecommendationCache()
    stored = CachedTagRecommendations(
        file_hash="abc123",
        extraction_version="docling-v1",
        candidate_model_digest="old",
        decision_model_digest="old",
        decisions=(
            TagDecision(tag="research", decision="accept", confidence=0.9),
        ),
    )

    cache.store(stored)

    assert cache.lookup(stored.file_hash, model_digest="old") == stored
    assert cache.lookup(stored.file_hash, model_digest="new") is None


def test_recommendations_prefer_existing_tags_and_require_low_confidence_review() -> None:
    search = FakeSearch(["Research", "reference"])
    gateway = FakeGateway(["research", "New", " new ", "invoice"])
    cache = InMemoryTagRecommendationCache()

    result = recommend_tags(
        "A new research report with reference material.",
        file_hash="abc123",
        extraction_version="docling-v1",
        search=search,
        gateway=gateway,
        cache=cache,
        candidate_model="inkling-2-mini",
        decision_model="nimble:9b",
        max_candidates=4,
        confirmation_threshold=0.7,
    )

    assert [item.tag for item in result.decisions] == [
        "Research",
        "reference",
        "New",
        "invoice",
    ]
    assert result.decisions[0].decision == "accept"
    assert result.decisions[0].is_new is False
    assert result.decisions[1].decision == "review"
    assert result.decisions[2].decision == "review"
    assert result.decisions[2].is_new is True
    assert result.requires_confirmation == ("reference", "New", "invoice")
    assert len(gateway.decision_calls[0]["questions"]) == 4
    assert search.calls[0]["query"] == "A new research report with reference material."
    assert "Research" in gateway.chat_calls[0]["messages"][1]["content"]
    assert "reference" in gateway.chat_calls[0]["messages"][1]["content"]


def test_recommendations_are_cache_hits_without_repeating_model_requests() -> None:
    search = FakeSearch(["Research"])
    gateway = FakeGateway(["invoice"])
    cache = InMemoryTagRecommendationCache()
    kwargs = {
        "file_hash": "abc123",
        "extraction_version": "docling-v1",
        "search": search,
        "gateway": gateway,
        "cache": cache,
        "candidate_model_digest": "generation-old",
        "decision_model_digest": "decision-old",
    }

    first = recommend_tags("research report", **kwargs)
    second = recommend_tags("research report", **kwargs)

    assert second == first
    assert len(gateway.chat_calls) == 1
    assert len(gateway.decision_calls) == 1
    assert len(search.calls) == 1


def test_exact_extraction_and_model_identities_are_part_of_cache_key() -> None:
    cache = InMemoryTagRecommendationCache()
    stored = CachedTagRecommendations(
        file_hash="abc123",
        extraction_version="docling-v1",
        candidate_model_digest="candidate-v1",
        decision_model_digest="decision-v1",
        decisions=(),
    )
    cache.store(stored)

    assert (
        cache.lookup(
            stored.file_hash,
            extraction_version="docling-v1",
            candidate_model_digest="candidate-v1",
            decision_model_digest="decision-v1",
        )
        == stored
    )
    assert (
        cache.lookup(
            stored.file_hash,
            extraction_version="docling-v2",
            candidate_model_digest="candidate-v1",
            decision_model_digest="decision-v1",
        )
        is None
    )
    assert (
        cache.lookup(
            stored.file_hash,
            extraction_version="docling-v1",
            candidate_model_digest="candidate-v2",
            decision_model_digest="decision-v1",
        )
        is None
    )


def test_database_cache_round_trips_typed_recommendations(tmp_path: Path) -> None:
    database_path = tmp_path / "tag-cache.sqlite3"
    upgrade_database(database_path)
    session_factory = create_session_factory(database_path)
    with session_factory.begin() as session:
        file = File(
            canonical_path=r"C:\docs\report.md",
            content_hash="abc123",
            size_bytes=10,
            mtime_ns=1,
            mime_type="text/markdown",
        )
        session.add(file)
        session.flush()
        DatabaseTagRecommendationCache(session).store(
            CachedTagRecommendations(
                file_id=file.id,
                file_path=file.canonical_path,
                file_hash="abc123",
                extraction_version="docling-v1",
                candidate_model_digest="candidate-v1",
                decision_model_digest="decision-v1",
                decisions=(
                    TagDecision(
                        tag="research",
                        decision="review",
                        confidence=0.55,
                        is_new=True,
                    ),
                ),
            )
        )

    with session_factory() as session:
        cached = DatabaseTagRecommendationCache(session).lookup(
            "abc123",
            extraction_version="docling-v1",
            candidate_model_digest="candidate-v1",
            decision_model_digest="decision-v1",
        )
        columns = {column["name"] for column in inspect(session.get_bind()).get_columns(
            "tag_suggestion_cache"
        )}

    assert cached is not None
    assert cached.decisions == (
        TagDecision(
            tag="research",
            decision="review",
            confidence=0.55,
            is_new=True,
        ),
    )
    assert {
        "extraction_version",
        "candidate_model_digest",
        "decision_model_digest",
    } <= columns


def test_generation_is_bounded_before_decision() -> None:
    gateway = FakeGateway([f"tag-{index}" for index in range(10)])

    result = recommend_tags(
        "report",
        file_hash="bounded",
        extraction_version="v1",
        search=FakeSearch(["existing"]),
        gateway=gateway,
        max_candidates=3,
    )

    assert [item.tag for item in result.decisions] == [
        "existing",
        "tag-0",
        "tag-1",
    ]


def test_low_confidence_rejection_still_requires_confirmation() -> None:
    class RejectingGateway(FakeGateway):
        async def system_one(
            self,
            model: str,
            *,
            state: Any,
            questions: Mapping[str, Any],
            keep_alive: float | str | None = "5m",
        ) -> SimpleNamespace:
            result = await super().system_one(
                model,
                state=state,
                questions=questions,
                keep_alive=keep_alive,
            )
            for answer in result.answers.values():
                answer["choice"] = "reject"
                answer["confidence"] = 0.55
                answer["probabilities"] = {
                    "accept": 0.225,
                    "review": 0.225,
                    "reject": 0.55,
                }
            return result

    result = recommend_tags(
        "report",
        file_hash="review",
        extraction_version="v1",
        search=FakeSearch([]),
        gateway=RejectingGateway(["possibly-relevant"]),
        candidate_model_digest="candidate",
        decision_model_digest="decision",
        confirmation_threshold=0.7,
    )

    assert result.decisions[0].decision == "review"
    assert result.requires_confirmation == ("possibly-relevant",)
