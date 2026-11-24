from collections.abc import Sequence
from hashlib import sha256
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..adapters.ollama.gateway import OllamaGateway
from ..domain.contracts import ModelProfile
from ..domain.errors import EmbeddingContextRejectedError, ModelIncompatibleError
from .chunker import DocumentChunk


class EmbeddedChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk: DocumentChunk
    vector: tuple[float, ...] = Field(min_length=1)


class EmbeddingBatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    vectors: tuple[EmbeddedChunk, ...] = ()
    complete_source_coverage: bool
    model: str
    model_digest: str
    dimensions: int


def _select_profile(
    profile: ModelProfile | Sequence[ModelProfile],
) -> ModelProfile:
    profiles = (profile,) if isinstance(profile, ModelProfile) else tuple(profile)
    if not profiles:
        raise ModelIncompatibleError("An embedding model profile is required.")
    identities = {
        (candidate.model, candidate.digest, candidate.dimensions)
        for candidate in profiles
    }
    if len(identities) != 1:
        raise ModelIncompatibleError(
            "Mixed embedding models, digests, or dimensions are not allowed."
        )
    selected = profiles[0]
    if selected.role != "embedding" or selected.dimensions is None:
        raise ModelIncompatibleError(
            "The selected profile is not a dimensioned embedding model."
        )
    return selected


def _merge_spans(spans: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _complete_coverage(
    originals: Sequence[DocumentChunk],
    embedded: Sequence[EmbeddedChunk],
) -> bool:
    if not originals:
        return not embedded
    return _merge_spans([chunk.source_span for chunk in originals]) == _merge_spans(
        [item.chunk.source_span for item in embedded]
    )


def _split_offset(text: str) -> int:
    if len(text) <= 1:
        return 0
    midpoint = len(text) // 2
    windows = (
        (r"\n\s*\n", 0),
        (r"\n", 1),
        (r"(?<=[.!?])\s+", 2),
        (r"\s+", 3),
    )
    for pattern, _ in windows:
        boundaries = [
            match.end()
            for match in re.finditer(pattern, text)
            if 0 < match.end() < len(text)
        ]
        if boundaries:
            return min(boundaries, key=lambda boundary: abs(boundary - midpoint))
    return midpoint


def _split_chunk(chunk: DocumentChunk) -> tuple[DocumentChunk, DocumentChunk] | None:
    offset = _split_offset(chunk.text)
    if offset <= 0 or offset >= len(chunk.text):
        return None
    left_text = chunk.text[:offset]
    right_text = chunk.text[offset:]
    start, end = chunk.source_span
    if start + offset >= end:
        return None

    def part(
        text: str,
        source_span: tuple[int, int],
    ) -> DocumentChunk:
        content_hash = sha256(text.encode()).hexdigest()
        return chunk.model_copy(
            update={
                "stable_id": sha256(
                    f"{chunk.stable_id}:{source_span[0]}:{source_span[1]}:{content_hash}".encode()
                ).hexdigest(),
                "text": text,
                "source_span": source_span,
                "content_hash": content_hash,
            }
        )

    return (
        part(left_text, (start, start + offset)),
        part(right_text, (start + offset, end)),
    )


async def _embed_chunk(
    chunk: DocumentChunk,
    gateway: Any,
    profile: ModelProfile,
) -> list[EmbeddedChunk]:
    dimensions = profile.dimensions
    assert dimensions is not None
    try:
        vectors = await gateway.embed(
            profile.model,
            chunk.text,
            dimensions=dimensions,
            truncate=False,
        )
    except EmbeddingContextRejectedError as exc:
        parts = _split_chunk(chunk)
        if parts is None:
            raise EmbeddingContextRejectedError(
                f"Chunk at {chunk.source_span} is unsplittable after context rejection."
            ) from exc
        left = await _embed_chunk(parts[0], gateway, profile)
        right = await _embed_chunk(parts[1], gateway, profile)
        return [*left, *right]

    if len(vectors) != 1:
        raise ModelIncompatibleError(
            f"Expected one embedding for one chunk, received {len(vectors)}."
        )
    vector = tuple(float(value) for value in vectors[0])
    if len(vector) != dimensions:
        raise ModelIncompatibleError(
            f"Embedding dimension mismatch: expected {dimensions}, received {len(vector)}."
        )
    return [EmbeddedChunk(chunk=chunk, vector=vector)]


async def embed_document_chunks(
    chunks: Sequence[DocumentChunk],
    gateway: OllamaGateway | Any,
    *,
    profile: ModelProfile | Sequence[ModelProfile],
) -> EmbeddingBatchResult:
    selected = _select_profile(profile)
    dimensions = selected.dimensions
    assert dimensions is not None
    embedded: list[EmbeddedChunk] = []
    for chunk in chunks:
        embedded.extend(await _embed_chunk(chunk, gateway, selected))

    embedded.sort(key=lambda item: item.chunk.source_span)
    vectors = tuple(
        item.model_copy(
            update={
                "chunk": item.chunk.model_copy(update={"ordinal": ordinal})
            }
        )
        for ordinal, item in enumerate(embedded)
    )
    return EmbeddingBatchResult(
        vectors=vectors,
        complete_source_coverage=_complete_coverage(chunks, vectors),
        model=selected.model,
        model_digest=selected.digest,
        dimensions=dimensions,
    )
