import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any, cast

from ..domain.contracts import ModelProfile
from ..domain.errors import (
    EmbeddingContextRejectedError,
    IndexTransactionError,
    ModelIncompatibleError,
)
from ..extraction.docling_adapter import ExtractedDocument
from .chunker import DocumentChunk, chunk_document
from .embedder import EmbeddedChunk, EmbeddingBatchResult, embed_document_chunks
from .repository import IndexingRepository, JobState

CancellationCheck = Callable[[], bool]


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
        self._index_version = index_version or (
            f"{selected_profile.model}@{selected_profile.digest}:"
            f"{selected_profile.dimensions}"
        )
        self._chunker = chunker

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
        except Exception as exc:
            current = self._repository.state(job_id)
            if current.status == "completed":
                return IndexingResult.from_state(current)
            failure = self._safe_failure(exc)
            return IndexingResult.from_state(self._repository.fail(current.id, failure))

        embedded: list[EmbeddedChunk] = []
        total_count = 0
        try:
            if self._cancelled(job_id, cancellation_check):
                return IndexingResult.from_state(self._repository.pause(job_id))

            extracted = self._extract(state.file_path)
            if not isinstance(extracted, ExtractedDocument):
                raise IndexTransactionError(
                    "The extractor did not return an extracted document."
                )
            if extracted.status == "failed":
                raise IndexTransactionError(
                    "Document extraction failed before indexing could continue."
                )

            chunks = list(self._chunker(extracted.pages))
            total_count = len(chunks)
            self._repository.checkpoint(
                job_id,
                processed_count=0,
                total_count=total_count,
            )

            for processed_count, chunk in enumerate(chunks, start=1):
                if self._cancelled(job_id, cancellation_check):
                    self._repository.checkpoint(
                        job_id,
                        processed_count=processed_count - 1,
                        total_count=total_count,
                    )
                    return IndexingResult.from_state(
                        self._repository.pause(job_id)
                    )
                result = self._run_embedding((chunk,))
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
                self._repository.checkpoint(
                    job_id,
                    processed_count=processed_count,
                    total_count=total_count,
                )

            if self._cancelled(job_id, cancellation_check):
                return IndexingResult.from_state(self._repository.pause(job_id))

            dimensions = self._profile.dimensions
            assert dimensions is not None
            completed = self._repository.replace_revision(
                job_id,
                extracted,
                embedded,
                extraction_version=self._extraction_version,
                model_digest=self._profile.digest,
                index_version=self._index_version,
                dimensions=dimensions,
            )
            return IndexingResult.from_state(completed)
        except asyncio.CancelledError:
            return IndexingResult.from_state(self._repository.pause(job_id))
        except Exception as exc:
            return IndexingResult.from_state(
                self._repository.fail(job_id, self._safe_failure(exc))
            )

    def cancel(self, job_id: int) -> IndexingResult:
        return IndexingResult.from_state(
            self._repository.request_cancellation(job_id)
        )

    def state(self, job_id: int) -> IndexingResult:
        return IndexingResult.from_state(self._repository.state(job_id))

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
