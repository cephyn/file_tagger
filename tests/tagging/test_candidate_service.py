from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import struct
from threading import Barrier
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import event, inspect, text

from file_tagger.domain.contracts import ModelProfile, TagDecision
from file_tagger.ollama.readiness import ReadinessReport, RequiredModels
from file_tagger.infrastructure.database import create_session_factory, upgrade_database
from file_tagger.infrastructure.app_paths import AppPaths
from file_tagger.persistence.models import (
    Chunk,
    Document,
    File,
    FileTag,
    ModelProfileRecord,
    Page,
    Tag,
)
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
    assert len(search.calls) == 2


def test_force_refresh_bypasses_cache_and_replaces_persistent_result() -> None:
    search = FakeSearch(["Research"])
    gateway = FakeGateway(["invoice"])
    cache = InMemoryTagRecommendationCache()
    kwargs = {
        "file_hash": "abc123",
        "file_path": r"C:\docs\report.md",
        "extraction_version": "docling-v1",
        "search": search,
        "gateway": gateway,
        "cache": cache,
        "candidate_model_digest": "generation-old",
        "decision_model_digest": "decision-old",
    }

    first = recommend_tags("research report", **kwargs)
    second = recommend_tags("research report", force_refresh=True, **kwargs)

    assert len(gateway.chat_calls) == 2
    assert len(gateway.decision_calls) == 2
    assert second.cached_at >= first.cached_at
    assert cache.lookup(
        "abc123",
        file_path=r"C:\docs\report.md",
        extraction_version="docling-v1",
        candidate_model_digest="generation-old",
        decision_model_digest="decision-old",
        document_state_digest=second.document_state_digest,
    ) == second


def test_cache_does_not_cross_target_file_boundaries_with_duplicate_content() -> None:
    search = FakeSearch(["Research"])
    gateway = FakeGateway(["invoice"])
    cache = InMemoryTagRecommendationCache()

    first = recommend_tags(
        "Filename: first.md\nreport",
        file_hash="same-content",
        file_path=r"C:\docs\first.md",
        extraction_version="docling-v1",
        search=search,
        gateway=gateway,
        cache=cache,
        candidate_model_digest="generation-old",
        decision_model_digest="decision-old",
    )
    second = recommend_tags(
        "Filename: second.md\nreport",
        file_hash="same-content",
        file_path=r"C:\docs\second.md",
        extraction_version="docling-v1",
        search=search,
        gateway=gateway,
        cache=cache,
        candidate_model_digest="generation-old",
        decision_model_digest="decision-old",
    )

    assert first.file_path == r"C:\docs\first.md"
    assert second.file_path == r"C:\docs\second.md"
    assert first.document_state_digest != second.document_state_digest
    assert len(gateway.chat_calls) == 2
    assert len(gateway.decision_calls) == 2


def test_candidate_digest_preserves_bounded_order_and_refreshes_exact_spelling() -> None:
    search = FakeSearch(["Research", "invoice", "ignored"])
    gateway = FakeGateway([])
    cache = InMemoryTagRecommendationCache()
    kwargs = {
        "file_hash": "abc123",
        "extraction_version": "docling-v1",
        "search": search,
        "gateway": gateway,
        "cache": cache,
        "candidate_model_digest": "generation-v1",
        "decision_model_digest": "decision-v1",
        "max_candidates": 2,
    }

    first = recommend_tags("research invoice", **kwargs)
    search.tags = ("research", " INVOICE ", "ignored-change")
    same_catalog = recommend_tags("research invoice", **kwargs)
    search.tags = ("invoice", "Research", "ignored")
    reordered_catalog = recommend_tags("research invoice", **kwargs)

    assert [item.tag for item in first.decisions] == ["Research", "invoice"]
    assert [item.tag for item in same_catalog.decisions] == ["research", "INVOICE"]
    assert same_catalog.candidate_set_digest == first.candidate_set_digest
    assert reordered_catalog.candidate_set_digest != first.candidate_set_digest
    assert len(search.calls) == 3
    assert len(gateway.chat_calls) == 0
    assert len(gateway.decision_calls) == 2


def test_exact_extraction_and_model_identities_are_part_of_cache_key() -> None:
    cache = InMemoryTagRecommendationCache()
    stored = CachedTagRecommendations(
        file_hash="abc123",
        extraction_version="docling-v1",
        candidate_model_digest="candidate-v1",
        decision_model_digest="decision-v1",
        candidate_set_digest="catalog-v1",
        max_candidates=4,
        confirmation_threshold=0.7,
        decisions=(),
    )
    cache.store(stored)

    assert (
        cache.lookup(
            stored.file_hash,
            extraction_version="docling-v1",
            candidate_model_digest="candidate-v1",
            decision_model_digest="decision-v1",
            candidate_set_digest="catalog-v1",
            max_candidates=4,
            confirmation_threshold=0.7,
        )
        == stored
    )
    assert (
        cache.lookup(
            stored.file_hash,
            extraction_version="docling-v2",
            candidate_model_digest="candidate-v1",
            decision_model_digest="decision-v1",
            candidate_set_digest="catalog-v1",
            max_candidates=4,
            confirmation_threshold=0.7,
        )
        is None
    )
    assert (
        cache.lookup(
            stored.file_hash,
            extraction_version="docling-v1",
            candidate_model_digest="candidate-v2",
            decision_model_digest="decision-v1",
            candidate_set_digest="catalog-v1",
        )
        is None
    )
    assert (
        cache.lookup(
            stored.file_hash,
            extraction_version="docling-v1",
            candidate_model_digest="candidate-v1",
            decision_model_digest="decision-v1",
            candidate_set_digest="catalog-v2",
            max_candidates=4,
            confirmation_threshold=0.7,
        )
        is None
    )
    assert (
        cache.lookup(
            stored.file_hash,
            extraction_version="docling-v1",
            candidate_model_digest="candidate-v1",
            decision_model_digest="decision-v1",
            candidate_set_digest="catalog-v1",
            max_candidates=3,
            confirmation_threshold=0.7,
        )
        is None
    )
    assert (
        cache.lookup(
            stored.file_hash,
            extraction_version="docling-v1",
            candidate_model_digest="candidate-v1",
            decision_model_digest="decision-v1",
            candidate_set_digest="catalog-v1",
            max_candidates=4,
            confirmation_threshold=0.6,
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
                candidate_set_digest="catalog-v1",
                max_candidates=4,
                confirmation_threshold=0.7,
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
            candidate_set_digest="catalog-v1",
            max_candidates=4,
            confirmation_threshold=0.7,
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
        "candidate_set_digest",
        "max_candidates",
        "confirmation_threshold",
        "document_state_digest",
    } <= columns


def test_database_cache_lookup_is_scoped_to_the_target_file(tmp_path: Path) -> None:
    database_path = tmp_path / "duplicate-tag-cache.sqlite3"
    upgrade_database(database_path)
    session_factory = create_session_factory(database_path)
    with session_factory.begin() as session:
        files = [
            File(
                canonical_path=path,
                content_hash="same-content",
                size_bytes=10,
                mtime_ns=index,
                mime_type="text/markdown",
            )
            for index, path in enumerate(
                (r"C:\docs\first.md", r"C:\docs\second.md"),
                start=1,
            )
        ]
        session.add_all(files)
        session.flush()
        for file in files:
            DatabaseTagRecommendationCache(session).store(
                CachedTagRecommendations(
                    file_id=file.id,
                    file_path=file.canonical_path,
                    file_hash="same-content",
                    document_state_digest=f"state-{file.id}",
                    extraction_version="docling-v1",
                    candidate_model_digest="candidate-v1",
                    decision_model_digest="decision-v1",
                    candidate_set_digest="catalog-v1",
                    max_candidates=4,
                    confirmation_threshold=0.7,
                    decisions=(),
                )
            )

    with session_factory() as session:
        first = DatabaseTagRecommendationCache(session).lookup(
            "same-content",
            file_path=r"C:\docs\first.md",
            document_state_digest="state-1",
            extraction_version="docling-v1",
        )
        second = DatabaseTagRecommendationCache(session).lookup(
            "same-content",
            file_path=r"C:\docs\second.md",
            document_state_digest="state-2",
            extraction_version="docling-v1",
        )
        wrong = DatabaseTagRecommendationCache(session).lookup(
            "same-content",
            file_path=r"C:\docs\second.md",
            document_state_digest="state-1",
            extraction_version="docling-v1",
        )

    assert first is not None and first.file_path == r"C:\docs\first.md"
    assert second is not None and second.file_path == r"C:\docs\second.md"
    assert wrong is None


def test_database_cache_store_is_safe_across_separate_sessions(tmp_path: Path) -> None:
    database_path = tmp_path / "concurrent-tag-cache.sqlite3"
    upgrade_database(database_path)
    session_factory = create_session_factory(database_path)
    with session_factory.begin() as session:
        file = File(
            canonical_path=r"C:\docs\concurrent.md",
            content_hash="concurrent-hash",
            size_bytes=10,
            mtime_ns=1,
            mime_type="text/markdown",
        )
        session.add(file)
        session.flush()
        file_id = file.id

    insert_barrier = Barrier(2)
    engine = session_factory.kw["bind"]

    def synchronize_cache_inserts(
        connection: Any,
        cursor: Any,
        statement: str,
        parameters: Any,
        context: Any,
        executemany: bool,
    ) -> None:
        del connection, cursor, parameters, context, executemany
        if "insert into tag_suggestion_cache" in statement.lower():
            insert_barrier.wait(timeout=5)

    event.listen(engine, "before_cursor_execute", synchronize_cache_inserts)

    def store(tag: str) -> None:
        with session_factory.begin() as session:
            DatabaseTagRecommendationCache(session).store(
                CachedTagRecommendations(
                    file_id=file_id,
                    file_path=r"C:\docs\concurrent.md",
                    file_hash="concurrent-hash",
                    extraction_version="docling-v1",
                    candidate_model_digest="candidate-v1",
                    decision_model_digest="decision-v1",
                    candidate_set_digest="catalog-v1",
                    max_candidates=4,
                    confirmation_threshold=0.7,
                    decisions=(
                        TagDecision(
                            tag=tag,
                            decision="accept",
                            confidence=0.9,
                        ),
                    ),
                )
            )

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(store, tag) for tag in ("first", "second")]
            for future in futures:
                future.result(timeout=10)
    finally:
        event.remove(engine, "before_cursor_execute", synchronize_cache_inserts)

    with session_factory() as session:
        cached = DatabaseTagRecommendationCache(session).lookup(
            "concurrent-hash",
            extraction_version="docling-v1",
            candidate_model_digest="candidate-v1",
            decision_model_digest="decision-v1",
            candidate_set_digest="catalog-v1",
            max_candidates=4,
            confirmation_threshold=0.7,
        )

    assert cached is not None
    assert cached.decisions[0].tag in {"first", "second"}


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


def _seed_production_chunk(
    session: Any,
    *,
    path: Path,
    stable_id: str,
    text_value: str,
    vector: tuple[float, ...],
    tags: Sequence[str],
    model_digest: str,
    index_version: str,
) -> None:
    file = File(
        canonical_path=str(path.resolve()),
        mime_type="text/markdown",
        indexing_state="indexed",
    )
    session.add(file)
    session.flush()
    document = Document(
        file_id=file.id,
        revision=1,
        file_hash=f"hash-{stable_id}",
        extraction_version="test",
        parser="test",
        page_count=1,
        extraction_status="complete",
        diagnostics=[],
    )
    session.add(document)
    session.flush()
    page = Page(
        document_id=document.id,
        page_number=1,
        ordinal=0,
        text=text_value,
        markdown=text_value,
        headings=[],
        quality="good",
    )
    session.add(page)
    session.flush()
    chunk = Chunk(
        stable_id=stable_id,
        document_id=document.id,
        page_id=page.id,
        ordinal=0,
        text=text_value,
        content_hash=f"content-{stable_id}",
        model_digest=model_digest,
        index_version=index_version,
    )
    session.add(chunk)
    session.flush()
    session.execute(
        text(
            "INSERT INTO chunk_embeddings_256 "
            "(chunk_id, embedding) VALUES (:chunk_id, :embedding)"
        ),
        {
            "chunk_id": chunk.id,
            "embedding": struct.pack("<256f", *vector),
        },
    )
    for tag_name in tags:
        tag = Tag(name=tag_name)
        session.add(tag)
        session.flush()
        session.add(FileTag(file_id=file.id, tag_id=tag.id))


def test_production_tag_suggester_uses_semantic_hybrid_search_and_new_quota(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from file_tagger.tag_suggestion import TagSuggester

    data_dir = tmp_path / "app-data"
    monkeypatch.setenv("FILE_TAGGER_DATA_DIR", str(data_dir))
    paths = AppPaths.discover()
    upgrade_database(paths.database)
    profile = ModelProfile(
        role="embedding",
        model="embeddinggemma-2:270m",
        digest="sha256:embedding",
        context_length=32_768,
        dimensions=256,
        capabilities=("embedding",),
    )
    assert profile.default_index_version is not None
    relevant_vector = (1.0, 0.0, *((0.0,) * 254))
    irrelevant_vector = (0.0, 1.0, *((0.0,) * 254))
    factory = create_session_factory(paths.database)
    with factory.begin() as session:
        session.add(
            ModelProfileRecord(
                role=profile.role,
                model=profile.model,
                digest=profile.digest,
                dimensions=profile.dimensions,
                context_length=profile.context_length,
                index_version=profile.default_index_version,
                capabilities=list(profile.capabilities),
                benchmark_data={},
            )
        )
        _seed_production_chunk(
            session,
            path=tmp_path / "semantic-map.md",
            stable_id="semantic-map",
            text_value="A catalog of subterranean cartography.",
            vector=relevant_vector,
            tags=tuple(f"existing-{index:02d}" for index in range(26)),
            model_digest=profile.digest,
            index_version=profile.default_index_version,
        )
        _seed_production_chunk(
            session,
            path=tmp_path / "unrelated-ledger.md",
            stable_id="unrelated-ledger",
            text_value="Quarterly invoices and account balances.",
            vector=irrelevant_vector,
            tags=("invoice-only",),
            model_digest=profile.digest,
            index_version=profile.default_index_version,
        )

    source = tmp_path / "friendly-woodland-animal.txt"
    source.write_text("friendly woodland animal habitat", encoding="utf-8")

    class EmbeddingGateway(FakeGateway):
        def __init__(self) -> None:
            super().__init__([f"new-{index:02d}" for index in range(12)])
            self.embedding_calls: list[dict[str, Any]] = []

        async def embed(
            self,
            model: str,
            text: str | Sequence[str],
            *,
            dimensions: int | None = None,
            truncate: bool = False,
            keep_alive: float | str | None = "5m",
        ) -> tuple[tuple[float, ...], ...]:
            self.embedding_calls.append(
                {
                    "model": model,
                    "text": text,
                    "dimensions": dimensions,
                    "truncate": truncate,
                    "keep_alive": keep_alive,
                }
            )
            return (relevant_vector,)

    gateway = EmbeddingGateway()
    result = TagSuggester(
        SimpleNamespace(get_pdf_extractor=lambda: "accurate"),
        gateway=gateway,
        cache=InMemoryTagRecommendationCache(),
        modern_runtime=SimpleNamespace(
            session_factory=factory,
            embedding_profile=profile,
            candidate_model="inkling-2-mini",
            decision_model="nimble:9b",
        ),
    ).recommend_tags_for_file(str(source))

    assert len(result.decisions) == 24
    assert sum(not item.is_new for item in result.decisions) == 12
    assert sum(item.is_new for item in result.decisions) == 12
    assert "invoice-only" not in {item.tag for item in result.decisions}
    assert {item.tag for item in result.decisions if item.is_new} == {
        f"new-{index:02d}" for index in range(12)
    }
    assert len(gateway.embedding_calls) == 1
    assert "friendly woodland animal habitat" in str(gateway.embedding_calls[0]["text"])


def test_production_tag_suggester_uses_runtime_selected_embedding_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from file_tagger.tag_suggestion import TagSuggester

    data_dir = tmp_path / "app-data"
    monkeypatch.setenv("FILE_TAGGER_DATA_DIR", str(data_dir))
    paths = AppPaths.discover()
    upgrade_database(paths.database)
    profile_a = ModelProfile(
        role="embedding",
        model="embedding-a",
        digest="sha256:embedding-a",
        context_length=32_768,
        dimensions=256,
        capabilities=("embedding",),
    )
    profile_b = ModelProfile(
        role="embedding",
        model="embedding-b",
        digest="sha256:embedding-b",
        context_length=32_768,
        dimensions=512,
        capabilities=("embedding",),
    )
    decision_profile = ModelProfile(
        role="decision",
        model="decision-a",
        digest="sha256:decision-a",
        context_length=32_768,
        capabilities=("systemone",),
    )
    generation_profile = ModelProfile(
        role="generation",
        model="candidate-a",
        digest="sha256:candidate-a",
        context_length=32_768,
        capabilities=("completion",),
    )
    factory = create_session_factory(paths.database)
    with factory.begin() as session:
        for profile in (profile_a, profile_b):
            session.add(
                ModelProfileRecord(
                    role=profile.role,
                    model=profile.model,
                    digest=profile.digest,
                    dimensions=profile.dimensions,
                    context_length=profile.context_length,
                    index_version=profile.default_index_version,
                    capabilities=list(profile.capabilities),
                    benchmark_data={},
                )
            )

    source = tmp_path / "selected-profile.txt"
    source.write_text("selected profile", encoding="utf-8")

    class EmbeddingGateway(FakeGateway):
        def __init__(self) -> None:
            super().__init__(["generated"])
            self.embedding_calls: list[str] = []

        async def embed(
            self,
            model: str,
            text: str | Sequence[str],
            *,
            dimensions: int | None = None,
            truncate: bool = False,
            keep_alive: float | str | None = "5m",
        ) -> tuple[tuple[float, ...], ...]:
            del text, truncate, keep_alive
            self.embedding_calls.append(model)
            assert dimensions == profile_a.dimensions
            return (tuple(0.0 for _ in range(profile_a.dimensions or 0)),)

    runtime = SimpleNamespace(
        session_factory=factory,
        readiness=ReadinessReport(
            ready=True,
            profiles=(
                profile_a,
                profile_b,
                decision_profile,
                generation_profile,
            ),
            selected_models=RequiredModels(
                embedding=profile_a.model,
                decision="decision-a",
                generation="candidate-a",
            ),
        ),
    )
    gateway = EmbeddingGateway()

    TagSuggester(
        SimpleNamespace(get_pdf_extractor=lambda: "accurate"),
        gateway=gateway,
        cache=InMemoryTagRecommendationCache(),
        modern_runtime=runtime,
    ).recommend_tags_for_file(str(source))

    assert gateway.embedding_calls == ["embedding-a"]
    assert [call["model"] for call in gateway.chat_calls] == ["candidate-a"]
    assert [call["model"] for call in gateway.decision_calls] == ["decision-a"]


def test_production_tag_suggester_rejects_incomplete_runtime_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from file_tagger.tag_suggestion import TagSuggester

    data_dir = tmp_path / "app-data"
    monkeypatch.setenv("FILE_TAGGER_DATA_DIR", str(data_dir))
    paths = AppPaths.discover()
    upgrade_database(paths.database)
    factory = create_session_factory(paths.database)
    source = tmp_path / "incomplete-selection.txt"
    source.write_text("incomplete selection", encoding="utf-8")
    runtime = SimpleNamespace(
        session_factory=factory,
        readiness=ReadinessReport(
            ready=False,
            selected_models=RequiredModels(
                embedding="embedding-a",
                decision="decision-a",
                generation="candidate-a",
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="Modern runtime model selection is incomplete",
    ):
        TagSuggester(
            SimpleNamespace(get_pdf_extractor=lambda: "accurate"),
            gateway=FakeGateway(["generated"]),
            cache=InMemoryTagRecommendationCache(),
            modern_runtime=runtime,
        ).recommend_tags_for_file(str(source))


def test_runtime_helpers_do_not_fall_back_to_stale_readiness_report() -> None:
    from file_tagger.tag_suggestion import (
        _runtime_embedding_profile,
        _runtime_selected_model,
    )

    profile = ModelProfile(
        role="embedding",
        model="embedding-a",
        digest="sha256:embedding-a",
        context_length=32_768,
        dimensions=256,
        capabilities=("embedding",),
    )
    runtime = SimpleNamespace(
        embedding_profile=None,
        candidate_model=None,
        decision_model=None,
        readiness=ReadinessReport(
            ready=True,
            profiles=(profile,),
            selected_models=RequiredModels(
                embedding=profile.model,
                decision="decision-a",
                generation="candidate-a",
            ),
        ),
    )

    assert _runtime_embedding_profile(runtime) is None
    assert _runtime_selected_model(runtime, "generation") is None
    assert _runtime_selected_model(runtime, "decision") is None


def test_production_tag_suggester_uses_query_aware_fts_without_embedding_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from file_tagger.tag_suggestion import TagSuggester

    data_dir = tmp_path / "app-data"
    monkeypatch.setenv("FILE_TAGGER_DATA_DIR", str(data_dir))
    paths = AppPaths.discover()
    upgrade_database(paths.database)
    model_digest = "sha256:index-only"
    index_version = f"embeddinggemma-2:270m@{model_digest}:256"
    relevant_vector = (1.0, 0.0, *((0.0,) * 254))
    factory = create_session_factory(paths.database)
    with factory.begin() as session:
        _seed_production_chunk(
            session,
            path=tmp_path / "cathedral-budget.md",
            stable_id="cathedral-budget",
            text_value="Cathedral restoration project budget and masonry.",
            vector=relevant_vector,
            tags=tuple(f"architecture-{index:02d}" for index in range(26)),
            model_digest=model_digest,
            index_version=index_version,
        )
        _seed_production_chunk(
            session,
            path=tmp_path / "garden-notes.md",
            stable_id="garden-notes",
            text_value="Seasonal planting notes and compost.",
            vector=relevant_vector,
            tags=("gardening-only",),
            model_digest=model_digest,
            index_version=index_version,
        )

    source = tmp_path / "cathedral-budget.txt"
    source.write_text("Cathedral restoration budget", encoding="utf-8")
    gateway = FakeGateway([f"new-{index:02d}" for index in range(12)])

    result = TagSuggester(
        SimpleNamespace(get_pdf_extractor=lambda: "accurate"),
        gateway=gateway,
        cache=InMemoryTagRecommendationCache(),
    ).recommend_tags_for_file(str(source))

    selected = {item.tag for item in result.decisions}
    assert len(selected) == 24
    assert "gardening-only" not in selected
    assert sum(not item.is_new for item in result.decisions) == 12
    assert sum(item.is_new for item in result.decisions) == 12
