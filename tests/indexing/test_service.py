from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from concurrent.futures import TimeoutError
from hashlib import sha256
import os
from pathlib import Path
import struct
import time

import pytest
from sqlalchemy import select, text

from file_tagger.domain.contracts import ModelProfile
from file_tagger.domain.errors import (
    EmbeddingContextRejectedError,
    IndexTransactionError,
)
from file_tagger.extraction.docling_adapter import ExtractedDocument, ExtractedPage
from file_tagger.indexing.chunker import DocumentChunk
from file_tagger.indexing.repository import IndexingRepository
from file_tagger.indexing.service import IndexingService
from file_tagger.infrastructure.app_paths import AppPaths
from file_tagger.infrastructure.database import create_session_factory, upgrade_database
from file_tagger.modern_context import ModernRuntime
from file_tagger.persistence.models import Chunk, File
from file_tagger.search.hybrid import HybridSearch


@dataclass
class MutableExtractor:
    document: ExtractedDocument
    calls: int = 0

    def extract(self, file_path: str | Path) -> ExtractedDocument:
        self.calls += 1
        return self.document


class FixedGateway:
    async def embed(
        self,
        model: str,
        text: str,
        *,
        dimensions: int | None = None,
        truncate: bool = False,
        keep_alive: str = "5m",
    ) -> tuple[tuple[float, ...],]:
        del model, text, truncate, keep_alive
        assert dimensions is not None
        return (tuple(float(index) for index in range(dimensions)),)


@dataclass
class CountingGateway(FixedGateway):
    texts: list[str]

    async def embed(
        self,
        model: str,
        text: str,
        *,
        dimensions: int | None = None,
        truncate: bool = False,
        keep_alive: str = "5m",
    ) -> tuple[tuple[float, ...],]:
        self.texts.append(text)
        return await super().embed(
            model,
            text,
            dimensions=dimensions,
            truncate=truncate,
            keep_alive=keep_alive,
        )


class SlowGateway(FixedGateway):
    async def embed(
        self,
        model: str,
        text: str,
        *,
        dimensions: int | None = None,
        truncate: bool = False,
        keep_alive: str = "5m",
    ) -> tuple[tuple[float, ...],]:
        import asyncio

        await asyncio.sleep(0.35)
        return await super().embed(
            model,
            text,
            dimensions=dimensions,
            truncate=truncate,
            keep_alive=keep_alive,
        )


class SplittingGateway(FixedGateway):
    async def embed(
        self,
        model: str,
        text: str,
        *,
        dimensions: int | None = None,
        truncate: bool = False,
        keep_alive: str = "5m",
    ) -> tuple[tuple[float, ...],]:
        if len(text) > 25:
            raise EmbeddingContextRejectedError("private model detail")
        return await super().embed(
            model,
            text,
            dimensions=dimensions,
            truncate=truncate,
            keep_alive=keep_alive,
        )


class FailingExtractor:
    def extract(self, file_path: str | Path) -> ExtractedDocument:
        del file_path
        raise RuntimeError("credential=do-not-store-this-secret")


class CrashingEmbedder:
    def embed_document_chunks(self, chunks: tuple[DocumentChunk, ...]) -> None:
        del chunks
        raise KeyboardInterrupt


class IdenticalExtractor:
    def extract(self, file_path: str | Path) -> ExtractedDocument:
        path = Path(file_path)
        return extraction(path, marker="identical content", file_hash="unused")


def extraction(path: Path, *, marker: str, file_hash: str) -> ExtractedDocument:
    del file_hash
    return ExtractedDocument(
        file_path=str(path),
        file_hash=sha256(path.read_bytes()).hexdigest(),
        parser="test",
        parser_version="1",
        status="complete",
        pages=(
            ExtractedPage(
                number=1,
                ordinal=0,
                text=marker,
                markdown=f"# Revision\n\n{marker}",
                headings=("# Revision",),
            ),
        ),
    )


def test_identical_files_receive_distinct_stable_chunk_ids(tmp_path: Path) -> None:
    database = tmp_path / "identical-files.sqlite3"
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text("same source", encoding="utf-8")
    second.write_text("same source", encoding="utf-8")
    upgrade_database(database)
    session_factory = create_session_factory(database)
    service = IndexingService(
        IndexingRepository(session_factory),
        extractor=IdenticalExtractor(),
        embedding_gateway=FixedGateway(),
        embedding_profile=ModelProfile(
            role="embedding",
            model="embeddinggemma-2:270m",
            digest="sha256:embedding",
            context_length=1_000,
            dimensions=256,
        ),
    )

    assert service.reindex(first).status == "completed"
    assert service.reindex(second).status == "completed"

    with session_factory() as session:
        stable_ids = list(session.scalars(select(Chunk.stable_id)))
    assert len(stable_ids) == 2
    assert len(set(stable_ids)) == 2


@pytest.mark.parametrize("deletion_method", ("delete_file", "sync_legacy_changes"))
def test_runtime_file_deletion_removes_vectors_before_chunk_id_reuse(
    tmp_path: Path,
    deletion_method: str,
) -> None:
    database = tmp_path / "live-deletion.sqlite3"
    source = tmp_path / "guide.md"
    source.write_text("source", encoding="utf-8")
    upgrade_database(database)
    session_factory = create_session_factory(database)
    service = IndexingService(
        IndexingRepository(session_factory),
        extractor=MutableExtractor(
            extraction(source, marker="reusable chunk", file_hash="unused")
        ),
        embedding_gateway=FixedGateway(),
        embedding_profile=ModelProfile(
            role="embedding",
            model="embeddinggemma-2:270m",
            digest="sha256:embedding",
            context_length=1_000,
            dimensions=256,
        ),
    )
    assert service.reindex(source).status == "completed"
    with session_factory.begin() as session:
        original_chunk_id = session.scalar(select(Chunk.id))
        assert original_chunk_id is not None
        for dimensions in (512, 768):
            vector = tuple(float(index) for index in range(dimensions))
            session.execute(
                text(
                    f"INSERT INTO chunk_embeddings_{dimensions} "
                    "(chunk_id, embedding) VALUES (:chunk_id, :embedding)"
                ),
                {
                    "chunk_id": original_chunk_id,
                    "embedding": struct.pack(f"<{dimensions}f", *vector),
                },
            )

    runtime = ModernRuntime(AppPaths.discover(), session_factory)
    if deletion_method == "delete_file":
        runtime.delete_file(str(source))
    else:
        runtime.sync_legacy_changes(file_deletes=(str(source),))

    with session_factory() as session:
        assert session.scalar(select(File)) is None
        assert session.scalar(select(Chunk)) is None
        for dimensions in (256, 512, 768):
            assert (
                session.scalar(
                    text(f"SELECT count(*) FROM chunk_embeddings_{dimensions}")
                )
                == 0
            )

    assert service.reindex(source).status == "completed"
    with session_factory() as session:
        replacement_chunk_id = session.scalar(select(Chunk.id))
    assert replacement_chunk_id == original_chunk_id


def test_indexed_file_mime_type_supports_filtered_hybrid_search(
    tmp_path: Path,
) -> None:
    database = tmp_path / "mime-search.sqlite3"
    source = tmp_path / "guide.md"
    source.write_text("source", encoding="utf-8")
    upgrade_database(database)
    session_factory = create_session_factory(database)
    repository = IndexingRepository(session_factory)
    profile = ModelProfile(
        role="embedding",
        model="embeddinggemma-2:270m",
        digest="sha256:embedding",
        context_length=1_000,
        dimensions=256,
    )
    service = IndexingService(
        repository,
        extractor=MutableExtractor(
            extraction(source, marker="unicorn habitat", file_hash="unused")
        ),
        embedding_gateway=FixedGateway(),
        embedding_profile=profile,
    )

    service.queue(source)
    with session_factory.begin() as session:
        file = session.scalar(select(File))
        assert file is not None
        assert file.mime_type == "text/markdown"
        file.mime_type = "application/octet-stream"

    indexed = service.reindex(source)

    assert indexed.status == "completed"
    embedding = tuple(float(index) for index in range(256))
    with session_factory() as session:
        file = session.scalar(select(File))
        assert file is not None
        assert file.mime_type == "text/markdown"
        assert file.content_hash == sha256(source.read_bytes()).hexdigest()
        assert file.size_bytes == source.stat().st_size
        assert file.mtime_ns == source.stat().st_mtime_ns
        results = HybridSearch(session).search(
            "friendly woodland creature",
            embedding,
            embedding_profile=profile,
            file_type="text/markdown",
        )
    assert len(results) == 1
    assert results[0].file_path == str(source.resolve())
    assert results[0].source == "vector"


def test_job_claim_is_exclusive_and_stale_owner_cannot_mutate_completion(
    tmp_path: Path,
) -> None:
    database = tmp_path / "claim.sqlite3"
    source = tmp_path / "guide.md"
    source.write_text("source", encoding="utf-8")
    upgrade_database(database)
    repository = IndexingRepository(create_session_factory(database))
    service = IndexingService(
        repository,
        extractor=MutableExtractor(
            extraction(source, marker="owned", file_hash="unused")
        ),
        embedding_gateway=FixedGateway(),
        embedding_profile=ModelProfile(
            role="embedding",
            model="embeddinggemma-2:270m",
            digest="sha256:embedding",
            context_length=1_000,
            dimensions=256,
        ),
    )
    job_id = service.queue(source)

    owner = repository.begin(job_id)

    with pytest.raises(IndexTransactionError, match="already claimed"):
        repository.begin(job_id)

    assert owner.lease_token is not None
    repository.pause(job_id, lease_token=owner.lease_token)
    completed = service.resume(job_id)

    with pytest.raises(IndexTransactionError, match="does not own"):
        repository.checkpoint(
            job_id,
            lease_token=owner.lease_token,
            processed_count=0,
            total_count=1,
        )
    with pytest.raises(IndexTransactionError, match="does not own"):
        repository.fail(
            job_id,
            {"code": "stale", "summary": "stale worker"},
            lease_token=owner.lease_token,
        )

    assert completed.status == "completed"
    assert repository.state(job_id).status == "completed"


def test_restarted_service_recovers_expired_job_without_cooperative_pause(
    tmp_path: Path,
) -> None:
    database = tmp_path / "crash-restart.sqlite3"
    source = tmp_path / "guide.md"
    source.write_text("source", encoding="utf-8")
    upgrade_database(database)
    session_factory = create_session_factory(database)
    now = [datetime(2026, 11, 24, 12, tzinfo=UTC)]
    profile = ModelProfile(
        role="embedding",
        model="embeddinggemma-2:270m",
        digest="sha256:embedding",
        context_length=1_000,
        dimensions=256,
    )
    crashed_repository = IndexingRepository(
        session_factory,
        lease_duration=timedelta(seconds=30),
        clock=lambda: now[0],
    )
    extractor = MutableExtractor(
        extraction(source, marker="recover me", file_hash="unused")
    )
    crashed_service = IndexingService(
        crashed_repository,
        extractor=extractor,
        embedder=CrashingEmbedder(),
        embedding_profile=profile,
    )
    job_id = crashed_service.queue(source)

    with pytest.raises(KeyboardInterrupt):
        crashed_service.resume(job_id)
    abandoned = crashed_repository.state(job_id)
    assert abandoned.status == "running"
    assert abandoned.lease_token is not None

    now[0] += timedelta(seconds=31)
    restarted_extractor = MutableExtractor(
        extraction(source, marker="recover me", file_hash="unused")
    )
    restarted_repository = IndexingRepository(
        session_factory,
        lease_duration=timedelta(seconds=30),
        clock=lambda: now[0],
    )
    IndexingService(
        restarted_repository,
        extractor=restarted_extractor,
        embedding_gateway=FixedGateway(),
        embedding_profile=profile,
    )

    recovered = restarted_repository.state(job_id)
    assert recovered.status == "completed"
    assert recovered.processed_count == recovered.total_count == 1
    assert restarted_extractor.calls == 0
    assert restarted_repository.current_revision(source) == 1
    with pytest.raises(IndexTransactionError, match="does not own"):
        restarted_repository.checkpoint(
            job_id,
            lease_token=abandoned.lease_token,
            processed_count=0,
            total_count=1,
        )


def test_restarted_service_does_not_take_over_live_owner(tmp_path: Path) -> None:
    database = tmp_path / "live-owner.sqlite3"
    source = tmp_path / "guide.md"
    source.write_text("source", encoding="utf-8")
    upgrade_database(database)
    session_factory = create_session_factory(database)
    now = [datetime(2026, 11, 24, 12, tzinfo=UTC)]
    repository = IndexingRepository(
        session_factory,
        lease_duration=timedelta(seconds=30),
        clock=lambda: now[0],
    )
    extractor = MutableExtractor(
        extraction(source, marker="still working", file_hash="unused")
    )
    profile = ModelProfile(
        role="embedding",
        model="embeddinggemma-2:270m",
        digest="sha256:embedding",
        context_length=1_000,
        dimensions=256,
    )
    job_id = repository.queue(source)
    owner = repository.begin(job_id)

    IndexingService(
        IndexingRepository(
            session_factory,
            lease_duration=timedelta(seconds=30),
            clock=lambda: now[0],
        ),
        extractor=extractor,
        embedding_gateway=FixedGateway(),
        embedding_profile=profile,
    )

    live = repository.state(job_id)
    assert live.status == "running"
    assert live.lease_token == owner.lease_token
    assert extractor.calls == 0


def test_live_worker_heartbeats_during_slow_embedding(tmp_path: Path) -> None:
    database = tmp_path / "heartbeat.sqlite3"
    source = tmp_path / "guide.md"
    source.write_text("source", encoding="utf-8")
    upgrade_database(database)
    repository = IndexingRepository(
        create_session_factory(database),
        lease_duration=timedelta(seconds=0.2),
    )
    service = IndexingService(
        repository,
        extractor=MutableExtractor(
            extraction(source, marker="long request", file_hash="unused")
        ),
        embedding_gateway=SlowGateway(),
        embedding_profile=ModelProfile(
            role="embedding",
            model="embeddinggemma-2:270m",
            digest="sha256:embedding",
            context_length=1_000,
            dimensions=256,
        ),
    )

    result = service.reindex(source)

    assert result.status == "completed"


def test_lease_is_fenced_before_a_slow_synchronous_operation(
    tmp_path: Path,
) -> None:
    database = tmp_path / "fenced-heartbeat.sqlite3"
    source = tmp_path / "guide.md"
    source.write_text("source", encoding="utf-8")
    upgrade_database(database)
    repository = IndexingRepository(
        create_session_factory(database),
        lease_duration=timedelta(seconds=0.2),
    )
    service = IndexingService(
        repository,
        extractor=MutableExtractor(
            extraction(source, marker="fenced", file_hash="unused")
        ),
        embedding_profile=ModelProfile(
            role="embedding",
            model="embeddinggemma-2:270m",
            digest="sha256:embedding",
            context_length=1_000,
            dimensions=256,
        ),
        embedder=lambda chunks: chunks,
        recover_on_startup=False,
    )
    job_id = service.queue(source)
    owner = repository.begin(job_id)
    assert owner.lease_token is not None

    heartbeat_calls = 0
    original_heartbeat = repository.heartbeat

    def tracked_heartbeat(job: int, *, lease_token: str):
        nonlocal heartbeat_calls
        heartbeat_calls += 1
        return original_heartbeat(job, lease_token=lease_token)

    repository.heartbeat = tracked_heartbeat  # type: ignore[method-assign]

    def operation():
        assert heartbeat_calls > 0
        time.sleep(0.35)
        return repository.state(job_id)

    result = service._with_lease_heartbeat(job_id, owner.lease_token, operation)

    assert result.lease_token == owner.lease_token
    assert heartbeat_calls >= 1


def test_worker_timeout_error_does_not_become_heartbeat_polling(
    tmp_path: Path,
) -> None:
    database = tmp_path / "worker-timeout.sqlite3"
    source = tmp_path / "guide.md"
    source.write_text("source", encoding="utf-8")
    upgrade_database(database)
    repository = IndexingRepository(
        create_session_factory(database),
        lease_duration=timedelta(seconds=0.2),
    )
    service = IndexingService(
        repository,
        extractor=MutableExtractor(
            extraction(source, marker="fenced", file_hash="unused")
        ),
        embedding_profile=ModelProfile(
            role="embedding",
            model="embeddinggemma-2:270m",
            digest="sha256:embedding",
            context_length=1_000,
            dimensions=256,
        ),
        embedder=lambda chunks: chunks,
        recover_on_startup=False,
    )
    job_id = service.queue(source)
    owner = repository.begin(job_id)
    assert owner.lease_token is not None
    original_heartbeat = repository.heartbeat
    heartbeat_calls = 0

    def guarded_heartbeat(job: int, *, lease_token: str):
        nonlocal heartbeat_calls
        heartbeat_calls += 1
        if heartbeat_calls > 2:
            raise AssertionError("worker TimeoutError was treated as a poll timeout")
        return original_heartbeat(job, lease_token=lease_token)

    repository.heartbeat = guarded_heartbeat  # type: ignore[method-assign]

    def operation():
        raise TimeoutError("embedding timed out")

    with pytest.raises(TimeoutError, match="embedding timed out"):
        service._with_lease_heartbeat(job_id, owner.lease_token, operation)

    assert heartbeat_calls == 1


def test_changed_source_near_commit_preserves_previous_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "changed-at-commit.sqlite3"
    source = tmp_path / "guide.md"
    source.write_text("source", encoding="utf-8")
    upgrade_database(database)
    session_factory = create_session_factory(database)
    repository = IndexingRepository(session_factory)
    extractor = MutableExtractor(
        extraction(source, marker="previous", file_hash="unused")
    )
    service = IndexingService(
        repository,
        extractor=extractor,
        embedding_gateway=FixedGateway(),
        embedding_profile=ModelProfile(
            role="embedding",
            model="embeddinggemma-2:270m",
            digest="sha256:embedding",
            context_length=1_000,
            dimensions=256,
        ),
    )
    assert service.reindex(source).status == "completed"
    extractor.document = extraction(
        source,
        marker="replacement",
        file_hash="unused",
    )
    replace_revision = repository.replace_revision

    def change_source_before_replace(*args, **kwargs):
        source.write_text("changed at commit", encoding="utf-8")
        return replace_revision(*args, **kwargs)

    monkeypatch.setattr(repository, "replace_revision", change_source_before_replace)

    result = service.reindex(source)

    assert result.status == "failed"
    assert repository.current_revision(source) == 1
    with session_factory() as session:
        assert session.scalar(
            text("SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'previous'")
        ) == 1
        assert session.scalar(
            text("SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'replacement'")
        ) == 0


def test_vector_insert_failure_keeps_previous_revision_searchable(
    tmp_path: Path,
) -> None:
    database = tmp_path / "index.sqlite3"
    source = tmp_path / "guide.md"
    source.write_text("source", encoding="utf-8")
    upgrade_database(database)
    session_factory = create_session_factory(database)
    repository = IndexingRepository(session_factory)
    extractor = MutableExtractor(extraction(source, marker="previous", file_hash="one"))
    profile = ModelProfile(
        role="embedding",
        model="embeddinggemma-2:270m",
        digest="sha256:embedding",
        context_length=1_000,
        dimensions=256,
        capabilities=("embedding",),
    )
    service = IndexingService(
        repository,
        extractor=extractor,
        embedding_gateway=FixedGateway(),
        embedding_profile=profile,
    )

    assert service.reindex(source).status == "completed"
    extractor.document = extraction(source, marker="replacement", file_hash="two")
    repository.fail_next_vector_insert()

    result = service.reindex(source)

    assert result.status == "failed"
    assert repository.current_revision(source) == 1
    with session_factory() as session:
        assert session.scalar(
            text("SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'previous'")
        ) == 1
        assert session.scalar(
            text(
                "SELECT count(*) FROM chunk_embeddings_256 "
                "WHERE chunk_id IN (SELECT id FROM chunks)"
            )
        ) == 1


def test_completed_job_progress_counts_source_chunks_not_vector_splits(
    tmp_path: Path,
) -> None:
    database = tmp_path / "progress.sqlite3"
    source = tmp_path / "guide.md"
    source.write_text("source", encoding="utf-8")
    upgrade_database(database)
    repository = IndexingRepository(create_session_factory(database))
    extractor = MutableExtractor(
        extraction(
            source,
            marker="a long paragraph with enough text to split safely " * 4,
            file_hash="one",
        )
    )
    service = IndexingService(
        repository,
        extractor=extractor,
        embedding_gateway=SplittingGateway(),
        embedding_profile=ModelProfile(
            role="embedding",
            model="embeddinggemma-2:270m",
            digest="sha256:embedding",
            context_length=1_000,
            dimensions=256,
        ),
    )

    result = service.reindex(source)

    assert result.status == "completed"
    assert result.total_count == 1
    assert result.processed_count == result.total_count


def test_cancellation_checkpoint_resumes_with_retry_count(tmp_path: Path) -> None:
    database = tmp_path / "cancel.sqlite3"
    source = tmp_path / "guide.md"
    source.write_text("source", encoding="utf-8")
    upgrade_database(database)
    repository = IndexingRepository(create_session_factory(database))
    document = extraction(
        source,
        marker="first source second source",
        file_hash="one",
    )
    chunker_calls = 0

    def two_chunks(
        pages: tuple[ExtractedPage, ...],
    ) -> list[DocumentChunk]:
        nonlocal chunker_calls
        chunker_calls += 1
        del pages
        return [
            DocumentChunk(
                stable_id=f"chunk-{ordinal}",
                ordinal=ordinal,
                page_number=1,
                text=text,
                source_span=(start, start + len(text)),
                content_hash=sha256(text.encode()).hexdigest(),
            )
            for ordinal, (text, start) in enumerate(
                (("first source", 0), ("second source", 13))
            )
        ]

    extractor = MutableExtractor(document)
    gateway = CountingGateway([])
    service = IndexingService(
        repository,
        extractor=extractor,
        embedding_gateway=gateway,
        embedding_profile=ModelProfile(
            role="embedding",
            model="embeddinggemma-2:270m",
            digest="sha256:embedding",
            context_length=1_000,
            dimensions=256,
        ),
        chunker=two_chunks,
    )
    job_id = service.queue(source)
    checks = 0

    def cancel_after_first_chunk() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 4

    paused = service.resume(
        job_id,
        cancellation_check=cancel_after_first_chunk,
    )

    assert paused.status == "queued"
    assert paused.total_count == 2
    assert paused.processed_count == 1
    assert repository.current_revision(source) is None

    completed = service.resume(job_id)

    assert completed.status == "completed"
    assert completed.retry_count == 1
    assert completed.processed_count == 2
    assert repository.current_revision(source) == 1
    assert extractor.calls == 1
    assert chunker_calls == 1
    assert gateway.texts == ["first source", "second source"]


def test_resume_invalidates_checkpoint_when_source_content_changes(
    tmp_path: Path,
) -> None:
    database = tmp_path / "changed-source.sqlite3"
    source = tmp_path / "guide.md"
    source.write_text("source", encoding="utf-8")
    initial_stat = source.stat()
    upgrade_database(database)
    repository = IndexingRepository(create_session_factory(database))
    extractor = MutableExtractor(
        extraction(source, marker="first second", file_hash="one")
    )
    gateway = CountingGateway([])

    def two_chunks(
        pages: tuple[ExtractedPage, ...],
    ) -> list[DocumentChunk]:
        del pages
        return [
            DocumentChunk(
                stable_id=f"chunk-{ordinal}",
                ordinal=ordinal,
                page_number=1,
                text=text,
                source_span=(start, start + len(text)),
                content_hash=sha256(text.encode()).hexdigest(),
            )
            for ordinal, (text, start) in enumerate(
                (("first", 0), ("second", 6))
            )
        ]

    service = IndexingService(
        repository,
        extractor=extractor,
        embedding_gateway=gateway,
        embedding_profile=ModelProfile(
            role="embedding",
            model="embeddinggemma-2:270m",
            digest="sha256:embedding",
            context_length=1_000,
            dimensions=256,
        ),
        chunker=two_chunks,
    )
    job_id = service.queue(source)
    checks = 0

    def cancel_after_first_chunk() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 4

    paused = service.resume(job_id, cancellation_check=cancel_after_first_chunk)
    source.write_text("update", encoding="utf-8")
    os.utime(source, ns=(initial_stat.st_atime_ns, initial_stat.st_mtime_ns))
    extractor.document = extraction(
        source,
        marker="first second updated",
        file_hash="unused",
    )

    completed = service.resume(job_id)

    assert paused.processed_count == 1
    assert completed.status == "completed"
    assert extractor.calls == 2
    assert gateway.texts == ["first", "first", "second"]


def test_cancelled_queued_job_pauses_before_extraction_and_can_resume(
    tmp_path: Path,
) -> None:
    database = tmp_path / "queued-cancel.sqlite3"
    source = tmp_path / "guide.md"
    source.write_text("source", encoding="utf-8")
    upgrade_database(database)
    repository = IndexingRepository(create_session_factory(database))
    extractor = MutableExtractor(
        extraction(source, marker="eventual", file_hash="one")
    )
    service = IndexingService(
        repository,
        extractor=extractor,
        embedding_gateway=FixedGateway(),
        embedding_profile=ModelProfile(
            role="embedding",
            model="embeddinggemma-2:270m",
            digest="sha256:embedding",
            context_length=1_000,
            dimensions=256,
        ),
    )
    job_id = service.queue(source)

    service.cancel(job_id)
    paused = service.resume(job_id)
    completed = service.resume(job_id)

    assert paused.status == "queued"
    assert paused.processed_count == 0
    assert completed.status == "completed"
    assert completed.retry_count == 1


def test_failure_summary_does_not_persist_exception_message(tmp_path: Path) -> None:
    database = tmp_path / "safe-error.sqlite3"
    source = tmp_path / "guide.md"
    source.write_text("source", encoding="utf-8")
    upgrade_database(database)
    repository = IndexingRepository(create_session_factory(database))
    service = IndexingService(
        repository,
        extractor=FailingExtractor(),
        embedding_gateway=FixedGateway(),
        embedding_profile=ModelProfile(
            role="embedding",
            model="embeddinggemma-2:270m",
            digest="sha256:embedding",
            context_length=1_000,
            dimensions=256,
        ),
    )

    result = service.reindex(source)

    assert result.status == "failed"
    assert result.failure == {
        "code": "indexing_failed",
        "summary": "Indexing failed before the file revision could be replaced.",
    }
    assert "do-not-store" not in str(repository.state(result.job_id).failure)


def test_failed_transaction_can_resume_without_losing_retry_state(
    tmp_path: Path,
) -> None:
    database = tmp_path / "retry.sqlite3"
    source = tmp_path / "guide.md"
    source.write_text("source", encoding="utf-8")
    upgrade_database(database)
    repository = IndexingRepository(create_session_factory(database))
    service = IndexingService(
        repository,
        extractor=MutableExtractor(
            extraction(source, marker="retryable", file_hash="one")
        ),
        embedding_gateway=FixedGateway(),
        embedding_profile=ModelProfile(
            role="embedding",
            model="embeddinggemma-2:270m",
            digest="sha256:embedding",
            context_length=1_000,
            dimensions=256,
        ),
    )
    repository.fail_next_vector_insert()

    failed = service.reindex(source)
    resumed = service.resume(failed.job_id)

    assert failed.status == "failed"
    assert failed.failed_count == 1
    assert resumed.status == "completed"
    assert resumed.retry_count == 1
    assert resumed.failed_count == 1
    assert repository.current_revision(source) == 1


def test_successful_revision_replaces_fts_and_vector_rows_together(
    tmp_path: Path,
) -> None:
    database = tmp_path / "replacement.sqlite3"
    source = tmp_path / "guide.md"
    source.write_text("source", encoding="utf-8")
    upgrade_database(database)
    session_factory = create_session_factory(database)
    repository = IndexingRepository(session_factory)
    extractor = MutableExtractor(
        extraction(source, marker="previousword", file_hash="one")
    )
    service = IndexingService(
        repository,
        extractor=extractor,
        embedding_gateway=FixedGateway(),
        embedding_profile=ModelProfile(
            role="embedding",
            model="embeddinggemma-2:270m",
            digest="sha256:embedding",
            context_length=1_000,
            dimensions=256,
        ),
    )
    assert service.reindex(source).status == "completed"
    extractor.document = extraction(
        source,
        marker="replacementword",
        file_hash="two",
    )

    result = service.reindex(source)

    assert result.status == "completed"
    assert result.revision == 2
    with session_factory() as session:
        assert session.scalar(
            text(
                "SELECT count(*) FROM chunks_fts "
                "WHERE chunks_fts MATCH 'previousword'"
            )
        ) == 0
        assert session.scalar(
            text(
                "SELECT count(*) FROM chunks_fts "
                "WHERE chunks_fts MATCH 'replacementword'"
            )
        ) == 1
        assert session.scalar(
            text("SELECT count(*) FROM chunk_embeddings_256")
        ) == 1
