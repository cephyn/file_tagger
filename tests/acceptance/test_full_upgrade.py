import asyncio
from collections.abc import AsyncIterator
import json
import shutil
import sqlite3
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest
from sqlalchemy import func, select

from file_tagger.domain.contracts import ModelProfile
from file_tagger.extraction.docling_adapter import ExtractedDocument, ExtractedPage
from file_tagger.indexing.repository import IndexingRepository
from file_tagger.indexing.service import IndexingService
from file_tagger.infrastructure.app_paths import AppPaths
from file_tagger.modern_context import ModernRuntime, initialize_modern_runtime
from file_tagger.ollama.readiness import ReadinessReport, RequiredModels
from file_tagger.persistence.models import File
from file_tagger.rag.planner import SearchPlan, build_search_plan_async
from file_tagger.rag.service import RagService
from file_tagger.search.hybrid import HybridSearch


class ReloadableConfig:
    def __init__(self, paths: AppPaths) -> None:
        self.paths = paths
        self.reloads = 0

    def reload_after_migration(self) -> None:
        self.reloads += 1


class FixtureExtractor:
    """Deterministic local extractor used to keep the acceptance test offline."""

    def extract(self, file_path: Path) -> ExtractedDocument:
        path = Path(file_path)
        content = path.read_bytes()
        if path.suffix.lower() == ".md":
            text = path.read_text(encoding="utf-8")
            markdown = text
            headings: tuple[str, ...] = ("Encounters", "Treasure")
        else:
            text = (
                "# Encounters\n"
                "- Goblin Scouts guard the Moonlit Bridge.\n"
                "## Treasure\n"
                "- A silver dragon statuette is hidden in the shrine."
            )
            markdown = f"# Encounters\n\n{text}"
            headings = ("Encounters",)
        return ExtractedDocument(
            file_path=str(path),
            file_hash=sha256(content).hexdigest(),
            parser="acceptance-fixture",
            parser_version="1",
            status="complete",
            pages=(
                ExtractedPage(
                    number=1,
                    ordinal=0,
                    text=text,
                    markdown=markdown,
                    headings=headings,
                ),
            ),
            full_markdown=markdown,
        )


class FakeOllama:
    """Local-only in-process stand-in for the Ollama chat and embedding APIs."""

    def __init__(self) -> None:
        self.chat_calls: list[dict[str, Any]] = []
        self.embed_calls: list[str] = []
        self.plans: list[Any] = []

    async def embed(
        self,
        model: str,
        text: str | Sequence[str],
        *,
        dimensions: int | None = None,
        truncate: bool = False,
        keep_alive: float | str | None = "5m",
    ) -> tuple[tuple[float, ...], ...]:
        del model, truncate, keep_alive
        if isinstance(text, str):
            texts: tuple[str, ...] = (text,)
        else:
            texts = tuple(text)
        self.embed_calls.extend(texts)
        width = dimensions or 256
        vectors: tuple[tuple[float, ...], ...] = tuple(
            tuple(
                (1.0 if str(index) in item.lower() else 0.25)
                for index in range(width)
            )
            for item in texts
        )
        return vectors

    async def chat(
        self,
        model: str,
        *,
        messages: Sequence[Mapping[str, Any]],
        schema: Mapping[str, Any] | None = None,
        keep_alive: float | str | None = "5m",
    ) -> str:
        del keep_alive
        assert ":cloud" not in model and "-cloud" not in model.lower()
        self.chat_calls.append({"model": model, "messages": messages, "schema": schema})
        properties = set(schema.get("properties", {})) if schema else set()

        if "intent_names" in properties or "intents" in properties:
            return json.dumps(
                {
                    "intent_names": ["locate"],
                    "exact_entities": ["encounters", "treasure"],
                    "paraphrase_queries": [
                        "Find encounters",
                        "Find treasure",
                    ],
                    "evidence_categories": ["encounters", "treasure"],
                }
            )

        if "claims" in properties:
            user_content = next(
                (
                    str(message["content"])
                    for message in reversed(messages)
                    if message.get("role") == "user"
                ),
                "",
            )
            evidence_start = user_content.find("Indexed evidence:")
            evidence_payload: list[dict[str, Any]] = []
            if evidence_start >= 0:
                evidence_text = user_content[evidence_start:].split(":", 1)[1].strip()
                try:
                    evidence_payload = json.loads(evidence_text)
                except json.JSONDecodeError:
                    evidence_payload = []

            claims: list[dict[str, Any]] = []
            answer_parts: list[str] = []
            for item in evidence_payload:
                chunk_id = str(item["chunk_id"])
                quote = str(item["text"])
                answer_parts.append(quote)
                claims.append(
                    {
                        "text": quote,
                        "evidence_quote": quote,
                        "chunk_ids": [chunk_id],
                    }
                )

            if not claims:
                return json.dumps({"answer": "not found in indexed files", "claims": []})
            return json.dumps(
                {
                    "answer": " ".join(answer_parts),
                    "claims": claims,
                }
            )

        return json.dumps({"intents": ["locate"]})

    async def chat_stream(
        self,
        model: str,
        *,
        messages: Sequence[Mapping[str, Any]],
        schema: Mapping[str, Any] | None = None,
        keep_alive: float | str | None = "5m",
    ) -> AsyncIterator[str]:
        del keep_alive
        response = await self.chat(
            model,
            messages=messages,
            schema=schema,
        )
        yield response


class AcceptanceApp:
    def __init__(self, runtime: ModernRuntime, fake: FakeOllama) -> None:
        self.runtime = runtime
        self.fake = fake
        self.config = ReloadableConfig(runtime.paths)
        self.plan: SearchPlan | None = None
        assert runtime.readiness is not None
        embedding_profile = runtime.embedding_profile
        generation_profile = runtime.readiness.selected_profile("generation")
        assert embedding_profile is not None
        assert generation_profile is not None
        self.embedding_profile: ModelProfile = embedding_profile
        self.generation_profile: ModelProfile = generation_profile

    def upgrade(self) -> ModernRuntime:
        return self.runtime

    def reindex_all(self) -> tuple[Any, ...]:
        return self.runtime.reindex_all()

    def ask(self, question: str, *, context: str = "") -> Any:
        async def collect() -> Any:
            factory = self.runtime.session_factory
            assert factory is not None
            session = factory()
            search = HybridSearch(
                session,
                model_digest=self.embedding_profile.digest,
                index_version=self.embedding_profile.default_index_version,
                embedding_profile=self.embedding_profile,
            )
            rag = RagService(
                search=search,
                gateway=self.fake,
                embedding_gateway=self.fake,
                model=self.generation_profile.model,
                model_digest=self.embedding_profile.digest,
                index_version=self.embedding_profile.default_index_version,
                embedding_profile=self.embedding_profile,
            )
            try:
                plan = await build_search_plan_async(
                    question,
                    context,
                    model=self.generation_profile.model,
                    fallback_model=None,
                    gateway=self.fake,
                )
                self.plan = plan
                return await rag.answer_plan_async(
                    plan,
                    context,
                    session_id="acceptance",
                )
            finally:
                session.close()

        return asyncio.run(collect())


@pytest.fixture
def legacy_install(tmp_path: Path) -> Path:
    fixture = Path(__file__).parents[1] / "fixtures" / "legacy"
    destination = tmp_path / "legacy"
    shutil.copytree(fixture, destination)

    library_source = Path(__file__).parents[1] / "fixtures" / "library"
    library = tmp_path / "library"
    shutil.copytree(library_source, library)

    with sqlite3.connect(destination / "file_tags.sqlite3") as connection:
        connection.execute(
            "UPDATE files SET path = ? WHERE id = 1",
            (str(library / "encounters.pdf"),),
        )
        connection.execute(
            "UPDATE files SET path = ? WHERE id = 2",
            (str(library / "adventure-outline.md"),),
        )

    return destination


@pytest.fixture
def fake_ollama() -> FakeOllama:
    return FakeOllama()


def test_upgrade_indexes_and_answers_from_local_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legacy_install: Path,
    fake_ollama: FakeOllama,
) -> None:
    data_dir = tmp_path / "per-user"
    monkeypatch.setenv("FILE_TAGGER_DATA_DIR", str(data_dir))
    paths = AppPaths.discover()
    paths = replace(
        paths,
        legacy_database=legacy_install / "file_tags.sqlite3",
        legacy_chroma=legacy_install / ".chroma",
    )

    runtime = initialize_modern_runtime(
        paths,
        ReloadableConfig(paths),
        "migration-password",
        start_readiness=False,
    )
    assert runtime.sqlite_migration is not None
    assert runtime.sqlite_migration.files_migrated == 2
    assert runtime.sqlite_migration.tags_migrated == 2
    assert runtime.sqlite_migration.associations_migrated == 2
    assert runtime.config_migration is not None
    assert runtime.config_migration.cloud_fields_removed == 3
    assert runtime.config_migration.backup_dir is not None
    assert runtime.config_migration.backup_dir.is_dir()

    sqlite_backups = list(
        (paths.data_dir / "migration-backups" / "sqlite").glob("file_tags-*.sqlite3")
    )
    assert len(sqlite_backups) == 1
    with sqlite3.connect(sqlite_backups[0]) as backup:
        assert backup.execute("SELECT count(*) FROM files").fetchone() == (2,)
        assert backup.execute("SELECT count(*) FROM tags").fetchone() == (2,)

    embedding_profile = ModelProfile(
        role="embedding",
        model="embeddinggemma-2:270m",
        digest="sha256:acceptance-embedding",
        context_length=8_192,
        dimensions=256,
        capabilities=("embedding",),
    )
    generation_profile = ModelProfile(
        role="generation",
        model="gemma4:e4b",
        digest="sha256:acceptance-generation",
        context_length=32_768,
        capabilities=("completion",),
    )
    decision_profile = ModelProfile(
        role="decision",
        model="nimble:9b",
        digest="sha256:acceptance-decision",
        context_length=8_192,
        capabilities=("completion",),
    )
    selected = RequiredModels(
        embedding=embedding_profile.model,
        decision=decision_profile.model,
        generation=generation_profile.model,
    )
    runtime.required_models = selected
    runtime.readiness = ReadinessReport(
        ready=True,
        profiles=(embedding_profile, generation_profile, decision_profile),
        selected_models=selected,
    )
    runtime._embedding_gateway = fake_ollama
    assert runtime.session_factory is not None
    runtime._indexing_service = IndexingService(
        IndexingRepository(runtime.session_factory),
        FixtureExtractor(),
        embedding_gateway=fake_ollama,
        embedding_profile=embedding_profile,
        recover_on_startup=False,
    )

    app = AcceptanceApp(runtime, fake_ollama)
    assert app.upgrade() is runtime
    results = app.reindex_all()
    assert len(results) == 2
    assert all(result.status == "completed" for result in results)

    answer = app.ask(
        "Find encounters and treasure",
        context=(
            (tmp_path / "library" / "adventure-outline.md").read_text(encoding="utf-8")
        ),
    )

    assert answer.citations
    assert "# Encounters" in answer.text
    assert "## Treasure" in answer.text
    assert app.plan is not None
    assert set(app.plan.evidence_categories) >= {"encounters", "treasure"}
    assert all(":cloud" not in call["model"] for call in fake_ollama.chat_calls)
    assert all("-cloud" not in call["model"].lower() for call in fake_ollama.chat_calls)

    with runtime.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(File)) == 2
