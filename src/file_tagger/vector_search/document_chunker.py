"""Compatibility adapter for provenance-aware document chunking."""

from dataclasses import dataclass
import re

from ..indexing.chunker import chunk_document


@dataclass(frozen=True)
class _Page:
    number: int
    markdown: str


class DocumentChunker:
    @staticmethod
    def chunk_document(
        text: str,
        min_chunk_size: int = 200,
        max_chunk_size: int = 1000,
        overlap: int = 50,
    ) -> list[str]:
        del min_chunk_size
        soft_limit_tokens = max(1, max_chunk_size // 4)
        overlap_tokens = min(max(0, overlap // 4), soft_limit_tokens - 1)
        return [
            chunk.text
            for chunk in chunk_document(
                [_Page(number=1, markdown=text)],
                soft_limit_tokens=soft_limit_tokens,
                overlap_tokens=overlap_tokens,
            )
        ]

    @staticmethod
    def extract_chunk_title(chunk: str) -> str:
        header_match = re.search(r"^(#+)\s+(.+)$", chunk, re.MULTILINE)
        if header_match:
            return header_match.group(2).strip()
        for line in chunk.splitlines():
            if line.strip():
                title = line.strip()
                return title[:50] + "..." if len(title) > 50 else title
        return "Untitled chunk"
