from collections.abc import Sequence
from hashlib import sha256
import math
from typing import Protocol
import re

from pydantic import BaseModel, ConfigDict, Field


class PageLike(Protocol):
    @property
    def number(self) -> int: ...

    @property
    def markdown(self) -> str: ...


class DocumentChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stable_id: str = Field(min_length=1)
    ordinal: int = Field(ge=0)
    page_number: int = Field(gt=0)
    heading: str | None = None
    section: str | None = None
    text: str = Field(min_length=1)
    source_span: tuple[int, int]
    content_hash: str = Field(min_length=1)


def _estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _boundary_candidates(text: str, start: int, soft_end: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    patterns = (
        (5, r"\n\s*\n"),
        (4, r"(?m)^#{1,6}\s+"),
        (4, r"(?m)^\|.*\|\s*$"),
        (3, r"(?<=[.!?])\s+"),
        (2, r"\n"),
        (1, r"\s+"),
    )
    for priority, pattern in patterns:
        for match in re.finditer(pattern, text[start:soft_end]):
            boundary = start + match.end()
            if start < boundary < soft_end:
                candidates.append((boundary, priority))
    return candidates


def _choose_end(text: str, start: int, soft_limit_chars: int) -> int:
    if start >= len(text):
        return len(text)
    soft_end = min(len(text), start + max(1, soft_limit_chars))
    if soft_end == len(text):
        return soft_end
    candidates = _boundary_candidates(text, start, soft_end)
    if not candidates:
        return soft_end
    maximum_priority = max(priority for _, priority in candidates)
    best_priority = max(
        candidate for candidate in candidates if candidate[1] == maximum_priority
    )
    return best_priority[0]


def _overlap_start(
    text: str,
    start: int,
    end: int,
    overlap_chars: int,
) -> int:
    if overlap_chars <= 0:
        return end
    earliest = max(start + 1, end - overlap_chars)
    candidates = [
        boundary
        for boundary, _ in _boundary_candidates(text, earliest, end)
        if boundary > start
    ]
    if not candidates:
        return end
    return min(candidates, key=lambda boundary: end - boundary)


def _heading_at(text: str, offset: int) -> str | None:
    heading: str | None = None
    for match in re.finditer(r"(?m)^(#{1,6})[ \t]+(.+?)\s*$", text[:offset]):
        heading = match.group(2).strip()
    return heading


def _chunk_page(
    page: PageLike,
    page_offset: int,
    *,
    soft_limit_tokens: int,
    overlap_tokens: int,
) -> list[DocumentChunk]:
    markdown = page.markdown
    if not markdown:
        return []

    soft_limit_chars = max(1, soft_limit_tokens * 4)
    overlap_chars = max(0, overlap_tokens * 4)
    chunks: list[DocumentChunk] = []
    start = 0
    while start < len(markdown):
        end = _choose_end(markdown, start, soft_limit_chars)
        if end <= start:
            end = min(len(markdown), start + 1)
        chunk_text = markdown[start:end]
        absolute_start = page_offset + start
        absolute_end = page_offset + end
        heading_offset = len(markdown) if len(markdown) == 0 else start + 1
        if start == 0:
            first_line_end = markdown.find("\n")
            heading_offset = len(markdown) if first_line_end < 0 else first_line_end
        heading = _heading_at(markdown, heading_offset)
        content_hash = sha256(chunk_text.encode()).hexdigest()
        chunks.append(
            DocumentChunk(
                stable_id=sha256(
                    (
                        f"{page.number}:{absolute_start}:{absolute_end}:{content_hash}"
                    ).encode()
                ).hexdigest(),
                ordinal=0,
                page_number=page.number,
                heading=heading,
                section=heading,
                text=chunk_text,
                source_span=(absolute_start, absolute_end),
                content_hash=content_hash,
            )
        )
        if end >= len(markdown):
            break
        next_start = _overlap_start(markdown, start, end, overlap_chars)
        start = end if next_start <= start else next_start
    return chunks


def chunk_document(
    pages: Sequence[PageLike],
    *,
    soft_limit_tokens: int = 400,
    overlap_tokens: int = 24,
) -> list[DocumentChunk]:
    if soft_limit_tokens <= 0:
        raise ValueError("soft_limit_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= soft_limit_tokens:
        raise ValueError("overlap_tokens must be non-negative and below the soft limit")

    chunks: list[DocumentChunk] = []
    page_offset = 0
    for page in pages:
        page_chunks = _chunk_page(
            page,
            page_offset,
            soft_limit_tokens=soft_limit_tokens,
            overlap_tokens=overlap_tokens,
        )
        chunks.extend(page_chunks)
        page_offset += len(page.markdown)

    return [
        chunk.model_copy(update={"ordinal": ordinal})
        for ordinal, chunk in enumerate(chunks)
    ]
