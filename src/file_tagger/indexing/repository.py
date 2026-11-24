from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import struct
from typing import Any

from sqlalchemy import delete, func, select, text
from sqlalchemy.orm import Session, sessionmaker

from ..domain.errors import IndexTransactionError
from ..extraction.docling_adapter import ExtractedDocument
from ..persistence.models import (
    Chunk,
    Document,
    File,
    IndexJob,
    IndexJobCheckpointChunk,
    Page,
)
from .chunker import DocumentChunk
from .embedder import EmbeddedChunk

VECTOR_DIMENSIONS = (256, 512, 768)


@dataclass(frozen=True)
class JobState:
    id: int
    file_path: str
    status: str
    total_count: int
    processed_count: int
    failed_count: int
    retry_count: int
    revision: int | None
    failure: dict[str, Any] | None


@dataclass(frozen=True)
class JobCheckpoint:
    extracted: ExtractedDocument
    chunks: tuple[DocumentChunk, ...]
    embedded_chunks: tuple[EmbeddedChunk, ...]
    processed_count: int
    total_count: int
    extraction_version: str
    model_digest: str
    index_version: str
    dimensions: int
    source_size: int
    source_mtime_ns: int
    source_hash: str


class IndexingRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory
        self._fail_next_vector = False

    def queue(self, file_path: str | Path, *, job_type: str = "index") -> int:
        canonical_path = self._canonical_path(file_path)
        with self._session_factory.begin() as session:
            file = session.scalar(
                select(File).where(File.canonical_path == canonical_path)
            )
            if file is None:
                file = File(
                    canonical_path=canonical_path,
                    indexing_state="queued",
                )
                session.add(file)
                session.flush()
            else:
                file.indexing_state = "queued"
            job = IndexJob(
                file_id=file.id,
                job_type=job_type,
                status="queued",
                total_count=0,
                processed_count=0,
                failed_count=0,
                retry_count=0,
            )
            session.add(job)
            session.flush()
            return int(job.id)

    def begin(self, job_id: int) -> JobState:
        with self._session_factory.begin() as session:
            job = self._job(session, job_id)
            if job.status == "completed":
                raise IndexTransactionError(
                    "A completed indexing job cannot be resumed."
                )
            is_retry = job.started_at is not None
            if is_retry:
                job.retry_count += 1
            job.status = "running"
            if not (
                job.failure and job.failure.get("code") == "cancel_requested"
            ):
                job.failure = None
            job.started_at = job.started_at or datetime.now(UTC)
            job.completed_at = None
            if job.file is not None:
                job.file.indexing_state = "running"
            session.flush()
            return self._state(session, job)

    def checkpoint(
        self,
        job_id: int,
        *,
        processed_count: int,
        total_count: int,
        extracted: ExtractedDocument | None = None,
        chunks: Sequence[DocumentChunk] = (),
        embedded_chunks: Sequence[EmbeddedChunk] = (),
        extraction_version: str | None = None,
        model_digest: str | None = None,
        index_version: str | None = None,
        dimensions: int | None = None,
        source_size: int | None = None,
        source_mtime_ns: int | None = None,
        source_hash: str | None = None,
        source_ordinal: int | None = None,
    ) -> JobState:
        with self._session_factory.begin() as session:
            job = self._job(session, job_id)
            job.processed_count = max(0, processed_count)
            job.total_count = max(0, total_count)
            if extracted is not None:
                metadata = (
                    extraction_version,
                    model_digest,
                    index_version,
                    dimensions,
                    source_size,
                    source_mtime_ns,
                    source_hash,
                )
                if any(value is None for value in metadata):
                    raise IndexTransactionError(
                        "Incomplete indexing checkpoint metadata."
                    )
                session.execute(
                    delete(IndexJobCheckpointChunk).where(
                        IndexJobCheckpointChunk.job_id == job.id
                    )
                )
                checkpoint_data = {
                    "schema_version": 1,
                    "extracted": extracted.model_dump(mode="json"),
                    "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
                    "extraction_version": extraction_version,
                    "model_digest": model_digest,
                    "index_version": index_version,
                    "dimensions": dimensions,
                    "source_size": source_size,
                    "source_mtime_ns": source_mtime_ns,
                    "source_hash": source_hash,
                }
                job.checkpoint = {
                    **checkpoint_data,
                    "checksum": self._checkpoint_checksum(checkpoint_data),
                }
            if embedded_chunks:
                if source_ordinal is None:
                    raise IndexTransactionError(
                        "Embedded checkpoint chunks need a source ordinal."
                    )
                session.add(
                    IndexJobCheckpointChunk(
                        job_id=job.id,
                        source_ordinal=source_ordinal,
                        embedded_chunks=[
                            chunk.model_dump(mode="json")
                            for chunk in embedded_chunks
                        ],
                    )
                )
            session.flush()
            return self._state(session, job)

    def load_checkpoint(self, job_id: int) -> JobCheckpoint | None:
        with self._session_factory() as session:
            payload = self._job(session, job_id).checkpoint
            if not isinstance(payload, dict) or payload.get("schema_version") != 1:
                return None
            checkpoint_data = dict(payload)
            embedded_rows = list(
                session.execute(
                    select(
                        IndexJobCheckpointChunk.source_ordinal,
                        IndexJobCheckpointChunk.embedded_chunks,
                    )
                    .where(IndexJobCheckpointChunk.job_id == job_id)
                    .order_by(IndexJobCheckpointChunk.source_ordinal)
                )
            )
            embedded_payloads = [row[1] for row in embedded_rows]
            try:
                checksum = checkpoint_data.pop("checksum")
                if checksum != self._checkpoint_checksum(checkpoint_data):
                    return None
                checkpoint = JobCheckpoint(
                    extracted=ExtractedDocument.model_validate(
                        checkpoint_data["extracted"]
                    ),
                    chunks=tuple(
                        DocumentChunk.model_validate(item)
                        for item in checkpoint_data["chunks"]
                    ),
                    embedded_chunks=tuple(
                        EmbeddedChunk.model_validate(item)
                        for embedded_group in embedded_payloads
                        for item in embedded_group
                    ),
                    processed_count=len(embedded_payloads),
                    total_count=len(checkpoint_data["chunks"]),
                    extraction_version=str(checkpoint_data["extraction_version"]),
                    model_digest=str(checkpoint_data["model_digest"]),
                    index_version=str(checkpoint_data["index_version"]),
                    dimensions=int(checkpoint_data["dimensions"]),
                    source_size=int(checkpoint_data["source_size"]),
                    source_mtime_ns=int(checkpoint_data["source_mtime_ns"]),
                    source_hash=str(checkpoint_data["source_hash"]),
                )
            except (KeyError, TypeError, ValueError):
                return None
            if (
                checkpoint_data.get("schema_version") != 1
                or [row[0] for row in embedded_rows]
                != list(range(len(embedded_rows)))
                or len(embedded_payloads) > len(checkpoint_data["chunks"])
                or [chunk.ordinal for chunk in checkpoint.chunks]
                != list(range(checkpoint.total_count))
                or any(
                    chunk.content_hash != self._content_hash(chunk.text)
                    for chunk in checkpoint.chunks
                )
                or [chunk.chunk.ordinal for chunk in checkpoint.embedded_chunks]
                != list(range(len(checkpoint.embedded_chunks)))
                or any(
                    len(chunk.vector) != checkpoint.dimensions
                    for chunk in checkpoint.embedded_chunks
                )
                or _merged_spans(chunk.source_span for chunk in checkpoint.chunks[
                    : checkpoint.processed_count
                ])
                != _merged_spans(
                    chunk.chunk.source_span for chunk in checkpoint.embedded_chunks
                )
                or any(
                    embedded.chunk.content_hash != self._content_hash(
                        embedded.chunk.text
                    )
                    for embedded in checkpoint.embedded_chunks
                )
            ):
                return None
            return checkpoint

    def request_cancellation(self, job_id: int) -> JobState:
        with self._session_factory.begin() as session:
            job = self._job(session, job_id)
            if job.status == "completed":
                raise IndexTransactionError(
                    "A completed indexing job cannot be cancelled."
                )
            job.failure = {
                "code": "cancel_requested",
                "summary": "Cancellation was requested; resume to continue this job.",
            }
            session.flush()
            return self._state(session, job)

    def cancellation_requested(self, job_id: int) -> bool:
        with self._session_factory() as session:
            failure = self._job(session, job_id).failure
            return bool(failure and failure.get("code") == "cancel_requested")

    def pause(self, job_id: int) -> JobState:
        with self._session_factory.begin() as session:
            job = self._job(session, job_id)
            job.status = "queued"
            job.failure = None
            job.completed_at = None
            if job.file is not None:
                job.file.indexing_state = "queued"
            session.flush()
            return self._state(session, job)

    def fail(
        self,
        job_id: int,
        failure: dict[str, str],
    ) -> JobState:
        with self._session_factory.begin() as session:
            job = self._job(session, job_id)
            job.status = "failed"
            job.failure = dict(failure)
            job.failed_count += 1
            job.completed_at = datetime.now(UTC)
            if job.file is not None:
                job.file.indexing_state = "failed"
            session.flush()
            return self._state(session, job)

    def state(self, job_id: int) -> JobState:
        with self._session_factory() as session:
            return self._state(session, self._job(session, job_id))

    def current_revision(self, file_path: str | Path) -> int | None:
        canonical_path = self._canonical_path(file_path)
        with self._session_factory() as session:
            return session.scalar(
                select(func.max(Document.revision))
                .join(File, Document.file_id == File.id)
                .where(File.canonical_path == canonical_path)
            )

    def fail_next_vector_insert(self) -> None:
        self._fail_next_vector = True

    def replace_revision(
        self,
        job_id: int,
        extracted: ExtractedDocument,
        embedded_chunks: Sequence[EmbeddedChunk],
        *,
        extraction_version: str,
        model_digest: str,
        index_version: str,
        dimensions: int,
    ) -> JobState:
        if dimensions not in VECTOR_DIMENSIONS:
            raise IndexTransactionError(
                f"Unsupported vector dimension: {dimensions}."
            )

        with self._session_factory.begin() as session:
            job = self._job(session, job_id)
            if job.file_id is None:
                raise IndexTransactionError("The indexing job has no file.")
            file = session.get(File, job.file_id)
            if file is None:
                raise IndexTransactionError("The indexing job file no longer exists.")

            canonical_path = self._canonical_path(extracted.file_path)
            if file.canonical_path != canonical_path:
                raise IndexTransactionError(
                    "The extracted path does not match the queued file."
                )

            previous_revision = session.scalar(
                select(func.max(Document.revision)).where(
                    Document.file_id == file.id
                )
            )
            revision = int(previous_revision or 0) + 1

            old_document_ids = list(
                session.scalars(
                    select(Document.id).where(Document.file_id == file.id)
                )
            )
            for old_document_id in old_document_ids:
                for old_dimensions in VECTOR_DIMENSIONS:
                    session.execute(
                        text(
                            f"DELETE FROM chunk_embeddings_{old_dimensions} "
                            "WHERE chunk_id IN "
                            "(SELECT id FROM chunks WHERE document_id = :document_id)"
                        ),
                        {"document_id": old_document_id},
                    )
            if old_document_ids:
                session.execute(
                    delete(Document).where(Document.file_id == file.id)
                )

            document = Document(
                file_id=file.id,
                revision=revision,
                file_hash=extracted.file_hash,
                extraction_version=extraction_version,
                parser=extracted.parser,
                parser_version=extracted.parser_version,
                page_count=extracted.page_count,
                extraction_status=extracted.status,
                diagnostics=[
                    diagnostic.model_dump(mode="json")
                    for diagnostic in extracted.diagnostics
                ],
            )
            session.add(document)
            session.flush()

            page_ids: dict[int, int] = {}
            for extracted_page in extracted.pages:
                page = Page(
                    document_id=document.id,
                    page_number=extracted_page.number,
                    ordinal=extracted_page.ordinal,
                    text=extracted_page.text,
                    markdown=extracted_page.markdown,
                    headings=list(extracted_page.headings),
                    quality=extracted_page.quality,
                )
                session.add(page)
                session.flush()
                page_ids[extracted_page.number] = int(page.id)

            vector_rows: list[tuple[int, bytes]] = []
            for embedded in embedded_chunks:
                page_id = page_ids.get(embedded.chunk.page_number)
                if page_id is None:
                    raise IndexTransactionError(
                        "A chunk refers to an extracted page that is unavailable."
                    )
                chunk = Chunk(
                    stable_id=embedded.chunk.stable_id,
                    document_id=document.id,
                    page_id=page_id,
                    ordinal=embedded.chunk.ordinal,
                    text=embedded.chunk.text,
                    heading=embedded.chunk.heading,
                    section=embedded.chunk.section,
                    content_hash=embedded.chunk.content_hash,
                    model_digest=model_digest,
                    index_version=index_version,
                )
                session.add(chunk)
                session.flush()
                vector_rows.append((int(chunk.id), self._vector_bytes(embedded.vector)))

            for chunk_id, vector in vector_rows:
                if self._fail_next_vector:
                    self._fail_next_vector = False
                    raise IndexTransactionError("Simulated vector insert failure.")
                session.execute(
                    text(
                        f"INSERT INTO chunk_embeddings_{dimensions} "
                        "(chunk_id, embedding) VALUES (:chunk_id, :embedding)"
                    ),
                    {"chunk_id": chunk_id, "embedding": vector},
                )

            file.content_hash = extracted.file_hash
            file.indexing_state = "indexed"
            try:
                source = Path(extracted.file_path)
                stat = source.stat()
            except OSError:
                pass
            else:
                file.size_bytes = stat.st_size
                file.mtime_ns = stat.st_mtime_ns

            job.status = "completed"
            job.failure = None
            job.processed_count = job.total_count
            job.checkpoint = None
            session.execute(
                delete(IndexJobCheckpointChunk).where(
                    IndexJobCheckpointChunk.job_id == job.id
                )
            )
            job.completed_at = datetime.now(UTC)
            session.flush()
            return self._state(session, job)

    @staticmethod
    def _canonical_path(file_path: str | Path) -> str:
        return str(Path(file_path).expanduser().resolve())

    @staticmethod
    def _vector_bytes(vector: Sequence[float]) -> bytes:
        return struct.pack(f"<{len(vector)}f", *vector)

    @staticmethod
    def _job(session: Session, job_id: int) -> IndexJob:
        job = session.get(IndexJob, job_id)
        if job is None:
            raise IndexTransactionError("The requested indexing job was not found.")
        return job

    def _state(self, session: Session, job: IndexJob) -> JobState:
        revision = None
        if job.file_id is not None:
            revision = session.scalar(
                select(func.max(Document.revision)).where(
                    Document.file_id == job.file_id
                )
            )
        file_path = str(job.file.canonical_path) if job.file is not None else ""
        return JobState(
            id=int(job.id),
            file_path=file_path,
            status=job.status,
            total_count=job.total_count,
            processed_count=job.processed_count,
            failed_count=job.failed_count,
            retry_count=job.retry_count,
            revision=revision,
            failure=dict(job.failure) if job.failure else None,
        )

    @staticmethod
    def _checkpoint_checksum(payload: dict[str, Any]) -> str:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
        return sha256(encoded).hexdigest()

    @staticmethod
    def _content_hash(text: str) -> str:
        return sha256(text.encode()).hexdigest()


IndexRepository = IndexingRepository


def _merged_spans(
    spans: Iterable[tuple[int, int]],
) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]
