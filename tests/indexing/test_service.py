from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import text

from file_tagger.domain.contracts import ModelProfile
from file_tagger.domain.errors import EmbeddingContextRejectedError
from file_tagger.extraction.docling_adapter import ExtractedDocument, ExtractedPage
from file_tagger.indexing.chunker import DocumentChunk
from file_tagger.indexing.repository import IndexingRepository
from file_tagger.indexing.service import IndexingService
from file_tagger.infrastructure.database import create_session_factory, upgrade_database


@dataclass
class MutableExtractor:
    document: ExtractedDocument

    def extract(self, file_path: str | Path) -> ExtractedDocument:
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


def extraction(path: Path, *, marker: str, file_hash: str) -> ExtractedDocument:
    return ExtractedDocument(
        file_path=str(path),
        file_hash=file_hash,
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
                content_hash=f"hash-{ordinal}",
            )
            for ordinal, (text, start) in enumerate(
                (("first source", 0), ("second source", 13))
            )
        ]

    service = IndexingService(
        repository,
        extractor=MutableExtractor(document),
        embedding_gateway=FixedGateway(),
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
        return checks >= 3

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
