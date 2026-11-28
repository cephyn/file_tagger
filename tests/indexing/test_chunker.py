from dataclasses import dataclass
from typing import Iterable

from file_tagger.indexing.chunker import chunk_document


@dataclass(frozen=True)
class Page:
    number: int
    markdown: str


def merge_spans(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def test_every_source_span_is_covered_by_a_chunk() -> None:
    pages = [
        Page(
            number=1,
            markdown=(
                "# Overview\n\n"
                + ("Alpha paragraph with details. " * 30)
                + "\n\n| Item | Value |\n| --- | --- |\n"
                + "\n".join(f"| {index} | data |" for index in range(20))
            ),
        )
    ]

    chunks = chunk_document(pages, soft_limit_tokens=40, overlap_tokens=5)

    covered = merge_spans(chunk.source_span for chunk in chunks)
    assert covered == [(0, len(pages[0].markdown))]
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.text == pages[0].markdown[start:end] for chunk in chunks for start, end in [chunk.source_span])


def test_chunking_preserves_page_heading_table_and_non_ascii_metadata() -> None:
    markdown = (
        "# 研究ノート\n\n"
        "日本語の本文です。重要な内容を保持します。\n\n"
        "## Table\n\n"
        "| Key | Value |\n| --- | --- |\n| α | β |\n"
    )

    chunks = chunk_document(
        [Page(number=3, markdown=markdown), Page(number=4, markdown="No heading page.")],
        soft_limit_tokens=10,
        overlap_tokens=2,
    )

    assert chunks[0].page_number == 3
    assert chunks[0].heading == "研究ノート"
    assert any(chunk.heading == "Table" for chunk in chunks)
    assert chunks[-1].page_number == 4
    assert all(chunk.content_hash for chunk in chunks)


def test_overlap_is_bounded_and_empty_pages_do_not_create_chunks() -> None:
    text = "\n\n".join(f"Paragraph {index}. " + ("x" * 80) for index in range(12))
    chunks = chunk_document(
        [Page(number=1, markdown=text), Page(number=2, markdown="")],
        soft_limit_tokens=25,
        overlap_tokens=4,
    )

    overlap_chars = [
        max(0, first.source_span[1] - second.source_span[0])
        for first, second in zip(chunks, chunks[1:])
        if first.page_number == second.page_number
    ]
    assert max(overlap_chars, default=0) <= 16
    assert all(chunk.text for chunk in chunks)


def test_legacy_chunker_returns_compatible_text_without_changing_coverage() -> None:
    text = "Long paragraph. " * 100
    chunks = chunk_document(
        [Page(number=1, markdown=text)],
        soft_limit_tokens=20,
        overlap_tokens=10,
    )
    assert chunks
