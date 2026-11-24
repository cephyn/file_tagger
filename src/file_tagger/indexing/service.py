import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
import inspect
from pathlib import Path
from threading import Event, Thread
from typing import Any, TypeVar, cast

from ..domain.contracts import ModelProfile
from ..domain.errors import (
    EmbeddingContextRejectedError,
    IndexTransactionError,
    ModelIncompatibleError,
)
from ..extraction.docling_adapter import ExtractedDocument
from .chunker import DocumentChunk, chunk_document
from .embedder import EmbeddedChunk, EmbeddingBatchResult, embed_document_chunks
from .repository import IndexingRepository, JobCheckpoint, JobState

CancellationCheck = Callable[[], bool]
T = TypeVar("T")


@dataclass(frozen=True)
class IndexingResult:
    job_id: int
    status: str
    file_path: str
    revision: int | None
    total_count: int
    processed_count: int
    failed_count: int
    retry_count: int
    failure: dict[str, Any] | None = None

    @classmethod
    def from_state(cls, state: JobState) -> "IndexingResult":
        return cls(
            job_id=state.id,
            status=state.status,
            file_path=state.file_path,
            revision=state.revision,
            total_count=state.total_count,
            processed_count=state.processed_count,
            failed_count=state.failed_count,
            retry_count=state.retry_count,
            failure=state.failure,
        )


class IndexingService:
    def __init__(
        self,
        repository: IndexingRepository,
        extractor: Any,
        embedding_gateway: Any | None = None,
        embedding_profile: ModelProfile | None = None,
        *,
        embedder: Any | None = None,
        profile: ModelProfile | None = None,
        extraction_version: str = "docling-v1",
        index_version: str | None = None,
        chunker: Callable[..., Sequence[DocumentChunk]] = chunk_document,
        recover_on_startup: bool = True,
    ) -> None:
        selected_profile = embedding_profile or profile
        if selected_profile is None:
            raise ValueError("An embedding profile is required.")
        if embedding_gateway is None and embedder is None:
            raise ValueError("An embedding gateway or embedder is required.")
        if selected_profile.dimensions not in {256, 512, 768}:
            raise ValueError("The embedding profile must use 256, 512, or 768 dimensions.")
        self._repository = repository
        self._extractor = extractor
        self._embedding_gateway = embedding_gateway
        self._embedder = embedder
        self._profile = selected_profile
        self._extraction_version = extraction_version
        selected_index_version = index_version or selected_profile.default_index_version
        if selected_index_version is None:
            raise ValueError("The embedding profile must have dimensions.")
        self._index_version = selected_index_version
        self._chunker = chunker
        if recover_on_startup:
            self.recover_abandoned()

    @property
    def repository(self) -> IndexingRepository:
        return self._repository

    def queue(self, file_path: str | Path) -> int:
        return self._repository.queue(file_path)

    def reindex(
        self,
        file_path: str | Path,
        *,
        cancellation_check: CancellationCheck | None = None,
        cancel_requested: CancellationCheck | None = None,
        job_id: int | None = None,
    ) -> IndexingResult:
        requested_job = job_id or self.queue(file_path)
        check = cancellation_check or cancel_requested
        return self.resume(requested_job, cancellation_check=check)

    def resume(
        self,
        job_id: int,
        *,
        cancellation_check: CancellationCheck | None = None,
    ) -> IndexingResult:
        try:
            state = self._repository.begin(job_id)
        except IndexTransactionError:
            current = self._repository.state(job_id)
            if current.status in {"completed", "running"}:
                return IndexingResult.from_state(current)
            raise
        lease_token = state.lease_token
        if lease_token is None:
            raise IndexTransactionError("The indexing job claim has no lease token.")

        embedded: list[EmbeddedChunk] = []
        total_count = 0
        try:
            if self._cancelled(job_id, cancellation_check):
                return IndexingResult.from_state(
                    self._pause_owned(job_id, lease_token)
                )

            checkpoint = self._with_lease_heartbeat(
                job_id,
                lease_token,
                lambda: self._repository.load_checkpoint(job_id),
            )
            if self._checkpoint_reusable(state, checkpoint):
                assert checkpoint is not None
                extracted = checkpoint.extracted
                chunks = list(checkpoint.chunks)
                embedded = list(checkpoint.embedded_chunks)
                total_count = checkpoint.total_count
                source_size = checkpoint.source_size
                source_mtime_ns = checkpoint.source_mtime_ns
                source_hash = checkpoint.source_hash
                first_unprocessed = checkpoint.processed_count
            else:
                source_signature = self._with_lease_heartbeat(
                    job_id,
                    lease_token,
                    lambda: self._source_signature(state.file_path),
                )
                extracted = self._with_lease_heartbeat(
                    job_id,
                    lease_token,
                    lambda: self._extract(state.file_path),
                )
                if not isinstance(extracted, ExtractedDocument):
                    raise IndexTransactionError(
                        "The extractor did not return an extracted document."
                    )
                if extracted.status == "failed":
                    raise IndexTransactionError(
                        "Document extraction failed before indexing could continue."
                    )
                after_extraction = self._with_lease_heartbeat(
                    job_id,
                    lease_token,
                    lambda: self._source_signature(state.file_path),
                )
                if after_extraction != source_signature:
                    raise IndexTransactionError(
                        "The source file changed during document extraction."
                    )
                if extracted.file_hash != source_signature[2]:
                    raise IndexTransactionError(
                        "The extracted source hash does not match the checkpoint source."
                    )

                chunks = list(
                    self._with_lease_heartbeat(
                        job_id,
                        lease_token,
                        lambda: self._chunker(extracted.pages),
                    )
                )
                total_count = len(chunks)
                source_size, source_mtime_ns, source_hash = after_extraction
                first_unprocessed = 0
                self._repository.checkpoint(
                    job_id,
                    lease_token=lease_token,
                    processed_count=0,
                    total_count=total_count,
                    extracted=extracted,
                    chunks=chunks,
                    extraction_version=self._extraction_version,
                    model_digest=self._profile.digest,
                    index_version=self._index_version,
                    dimensions=self._profile.dimensions,
                    source_size=source_size,
                    source_mtime_ns=source_mtime_ns,
                    source_hash=source_hash,
                )

            for processed_count, chunk in enumerate(
                chunks[first_unprocessed:],
                start=first_unprocessed + 1,
            ):
                if self._cancelled(job_id, cancellation_check):
                    self._repository.checkpoint(
                        job_id,
                        lease_token=lease_token,
                        processed_count=processed_count - 1,
                        total_count=total_count,
                    )
                    return IndexingResult.from_state(
                        self._pause_owned(job_id, lease_token)
                    )
                result = self._with_lease_heartbeat(
                    job_id,
                    lease_token,
                    lambda: self._run_embedding((chunk,)),
                )
                if not result.complete_source_coverage:
                    raise ModelIncompatibleError(
                        "Embedding output did not cover the complete source chunk."
                    )
                for vector in result.vectors:
                    ordinal = len(embedded)
                    embedded.append(
                        vector.model_copy(
                            update={
                                "chunk": vector.chunk.model_copy(
                                    update={"ordinal": ordinal}
                                )
                            }
                        )
                    )
                checkpoint_vectors = embedded[-len(result.vectors) :] if result.vectors else ()
                self._repository.checkpoint(
                    job_id,
                    lease_token=lease_token,
                    processed_count=processed_count,
                    total_count=total_count,
                    embedded_chunks=checkpoint_vectors,
                    source_ordinal=processed_count - 1,
                )

            if self._cancelled(job_id, cancellation_check):
                return IndexingResult.from_state(
                    self._pause_owned(job_id, lease_token)
                )

            dimensions = self._profile.dimensions
            assert dimensions is not None
            current_signature = self._with_lease_heartbeat(
                job_id,
                lease_token,
                lambda: self._source_signature(state.file_path),
            )
            if current_signature != (
                source_size,
                source_mtime_ns,
                source_hash,
            ):
                raise IndexTransactionError(
                    "The source file changed before the revision swap."
                )
            completed = self._repository.replace_revision(
                job_id,
                extracted,
                embedded,
                lease_token=lease_token,
                extraction_version=self._extraction_version,
                model_digest=self._profile.digest,
                index_version=self._index_version,
                dimensions=dimensions,
                source_size=source_size,
                source_mtime_ns=source_mtime_ns,
                source_hash=source_hash,
            )
            return IndexingResult.from_state(completed)
        except asyncio.CancelledError:
            return IndexingResult.from_state(
                self._pause_owned(job_id, lease_token)
            )
        except Exception as exc:
            return IndexingResult.from_state(
                self._fail_owned(job_id, lease_token, self._safe_failure(exc))
            )

    def cancel(self, job_id: int) -> IndexingResult:
        return IndexingResult.from_state(
            self._repository.request_cancellation(job_id)
        )

    def state(self, job_id: int) -> IndexingResult:
        return IndexingResult.from_state(self._repository.state(job_id))

    def recover_abandoned(self) -> tuple[IndexingResult, ...]:
        return tuple(self.resume(job_id) for job_id in self._repository.expired_job_ids())

    def _with_lease_heartbeat(
        self,
        job_id: int,
        lease_token: str,
        operation: Callable[[], T],
    ) -> T:
        stop = Event()

        def renew_lease() -> None:
            while not stop.wait(self._repository.heartbeat_interval):
                try:
                    self._repository.heartbeat(job_id, lease_token=lease_token)
                except IndexTransactionError:
                    return

        worker = Thread(target=renew_lease, name="indexing-lease-heartbeat", daemon=True)
        worker.start()
        try:
            return operation()
        finally:
            stop.set()
            worker.join()

    def _extract(self, file_path: str) -> ExtractedDocument:
        extract = getattr(self._extractor, "extract", self._extractor)
        value = extract(file_path)
        if inspect.isawaitable(value):
            value = self._run_awaitable(value)
        return cast(ExtractedDocument, value)

    def _run_embedding(
        self,
        chunks: Sequence[DocumentChunk],
    ) -> EmbeddingBatchResult:
        if self._embedder is None:
            result = embed_document_chunks(
                chunks,
                self._embedding_gateway,
                profile=self._profile,
            )
        else:
            embed = getattr(
                self._embedder,
                "embed_document_chunks",
                self._embedder,
            )
            try:
                result = embed(chunks, profile=self._profile)
            except TypeError:
                result = embed(chunks)
        if inspect.isawaitable(result):
            result = self._run_awaitable(result)
        return cast(EmbeddingBatchResult, result)

    @staticmethod
    def _run_awaitable(awaitable: Awaitable[Any]) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            async def resolve() -> Any:
                return await awaitable

            return asyncio.run(resolve())
        raise IndexTransactionError(
            "Indexing cannot synchronously run inside an active asyncio loop."
        )

    def _cancelled(
        self,
        job_id: int,
        cancellation_check: CancellationCheck | None,
    ) -> bool:
        return self._repository.cancellation_requested(job_id) or bool(
            cancellation_check and cancellation_check()
        )

    def _pause_owned(self, job_id: int, lease_token: str) -> JobState:
        try:
            return self._repository.pause(job_id, lease_token=lease_token)
        except IndexTransactionError:
            return self._repository.state(job_id)

    def _fail_owned(
        self,
        job_id: int,
        lease_token: str,
        failure: dict[str, str],
    ) -> JobState:
        try:
            return self._repository.fail(
                job_id,
                failure,
                lease_token=lease_token,
            )
        except IndexTransactionError:
            return self._repository.state(job_id)

    def _checkpoint_reusable(
        self,
        state: JobState,
        checkpoint: JobCheckpoint | None,
    ) -> bool:
        if checkpoint is None:
            return False
        dimensions = self._profile.dimensions
        if (
            checkpoint.extraction_version != self._extraction_version
            or checkpoint.model_digest != self._profile.digest
            or checkpoint.index_version != self._index_version
            or checkpoint.dimensions != dimensions
            or checkpoint.processed_count != state.processed_count
            or checkpoint.total_count != state.total_count
            or checkpoint.total_count != len(checkpoint.chunks)
        ):
            return False
        source_hash = checkpoint.source_hash
        try:
            source = Path(checkpoint.extracted.file_path).expanduser().resolve()
            current_signature = self._source_signature(str(source))
        except (OSError, IndexTransactionError):
            return False
        return (
            str(source) == state.file_path
            and current_signature
            == (checkpoint.source_size, checkpoint.source_mtime_ns, source_hash)
        )

    @staticmethod
    def _source_signature(file_path: str) -> tuple[int, int, str]:
        source = Path(file_path)
        before = source.stat()
        digest = sha256()
        with source.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        after = source.stat()
        signature = (after.st_size, after.st_mtime_ns)
        if signature != (before.st_size, before.st_mtime_ns):
            raise IndexTransactionError("The source file changed while being read.")
        return *signature, digest.hexdigest()

    @staticmethod
    def _safe_failure(exc: Exception) -> dict[str, str]:
        if isinstance(exc, EmbeddingContextRejectedError):
            return {
                "code": "embedding_context_rejected",
                "summary": "A chunk could not fit the embedding model context.",
            }
        if isinstance(exc, ModelIncompatibleError):
            return {
                "code": "model_incompatible",
                "summary": "The embedding model returned unusable index data.",
            }
        if isinstance(exc, IndexTransactionError):
            return {
                "code": "index_transaction_failed",
                "summary": "The file index transaction could not be completed.",
            }
        return {
            "code": "indexing_failed",
            "summary": "Indexing failed before the file revision could be replaced.",
        }
