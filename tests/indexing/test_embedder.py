import asyncio
from dataclasses import dataclass

import pytest

from file_tagger.domain.contracts import ModelProfile
from file_tagger.domain.errors import (
    EmbeddingContextRejectedError,
    ModelIncompatibleError,
)
from file_tagger.indexing.chunker import DocumentChunk
from file_tagger.indexing.embedder import embed_document_chunks


def profile(
    *,
    digest: str = "sha256:embedding",
    dimensions: int = 3,
) -> ModelProfile:
    return ModelProfile(
        role="embedding",
        model="embeddinggemma-2:270m",
        digest=digest,
        context_length=128_000,
        dimensions=dimensions,
        capabilities=("embedding",),
    )


def oversized_chunk() -> DocumentChunk:
    text = "\n\n".join(
        f"Paragraph {index}: " + ("context " * 20)
        for index in range(8)
    )
    return DocumentChunk(
        stable_id="original",
        ordinal=0,
        page_number=1,
        heading="Context",
        section="Context",
        text=text,
        source_span=(0, len(text)),
        content_hash="source-hash",
    )


@dataclass
class RejectingGateway:
    limit: int
    calls: list[dict] = None

    def __post_init__(self) -> None:
        self.calls = []

    async def embed(
        self,
        model,
        text,
        *,
        dimensions=None,
        truncate=False,
        keep_alive="5m",
    ):
        self.calls.append(
            {
                "model": model,
                "text": text,
                "dimensions": dimensions,
                "truncate": truncate,
                "keep_alive": keep_alive,
            }
        )
        if len(text) > self.limit:
            raise EmbeddingContextRejectedError("context rejected")
        return (tuple(float(index) for index in range(dimensions or 3)),)


class WrongDimensionGateway(RejectingGateway):
    async def embed(self, *args, **kwargs):
        await super().embed(*args, **kwargs)
        return ((0.1, 0.2, 0.3),)


def test_context_rejection_splits_until_every_span_is_embedded() -> None:
    gateway = RejectingGateway(limit=180)

    result = asyncio.run(
        embed_document_chunks(
            [oversized_chunk()],
            gateway,
            profile=profile(),
        )
    )

    assert len(result.vectors) > 1
    assert result.complete_source_coverage is True
    assert all(call["truncate"] is False for call in gateway.calls)
    assert [item.chunk.ordinal for item in result.vectors] == list(
        range(len(result.vectors))
    )


def test_unsplittable_context_rejection_is_explicit() -> None:
    chunk = oversized_chunk().model_copy(update={"text": "x", "source_span": (0, 1)})

    with pytest.raises(EmbeddingContextRejectedError, match="unsplittable"):
        asyncio.run(
            embed_document_chunks(
                [chunk],
                RejectingGateway(limit=0),
                profile=profile(),
            )
        )


def test_wrong_embedding_dimension_is_rejected() -> None:
    short = oversized_chunk().model_copy(
        update={"text": "short", "source_span": (0, 5)}
    )
    short_gateway = RejectingGateway(limit=1_000)

    result = asyncio.run(
        embed_document_chunks(
            [short],
            short_gateway,
            profile=profile(dimensions=5),
        )
    )

    assert len(result.vectors[0].vector) == 5
    with pytest.raises(ModelIncompatibleError, match="dimension"):
        asyncio.run(
            embed_document_chunks(
                [short],
                WrongDimensionGateway(limit=1_000),
                profile=profile(dimensions=2),
            )
        )


def test_mixed_model_profiles_are_rejected() -> None:
    with pytest.raises(ModelIncompatibleError, match="Mixed"):
        asyncio.run(
            embed_document_chunks(
                [oversized_chunk()],
                RejectingGateway(limit=1_000),
                profile=[
                    profile(digest="sha256:first"),
                    profile(digest="sha256:second"),
                ],
            )
        )
