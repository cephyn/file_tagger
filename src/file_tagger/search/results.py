from collections.abc import Iterable, Mapping
from base64 import urlsafe_b64encode
import json
from typing import Literal, cast

from pydantic import Field

from ..domain.contracts import DomainRecord, SearchCandidate

DEFAULT_SNIPPET_LENGTH = 240
SearchSource = Literal["fts", "vector", "hybrid"]


class MatchedSection(DomainRecord):
    chunk_id: str = Field(min_length=1)
    page_number: int = Field(gt=0)
    heading: str | None = None
    section: str | None = None
    snippet: str = Field(min_length=1)


class SearchCitation(DomainRecord):
    citation_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    page_number: int = Field(gt=0)
    chunk_id: str = Field(min_length=1)
    heading: str | None = None
    section: str | None = None
    excerpt: str = Field(min_length=1)

    @property
    def file_path(self) -> str:
        return self.path


Citation = SearchCitation


class SearchResult(SearchCandidate):
    tags: tuple[str, ...] = ()
    mime_type: str | None = None
    summary: str = Field(min_length=1)
    matched_sections: tuple[MatchedSection, ...] = Field(min_length=1)
    citations: tuple[SearchCitation, ...] = Field(min_length=1)
    stale: bool = False

    @property
    def path(self) -> str:
        return self.file_path

    @property
    def snippets(self) -> tuple[str, ...]:
        return tuple(section.snippet for section in self.matched_sections)


def make_result(
    chunks: object | Iterable[object],
    *,
    source: SearchSource = "fts",
    rank: int = 1,
    score: float = 0.0,
    snippet_length: int = DEFAULT_SNIPPET_LENGTH,
) -> SearchResult:
    if snippet_length <= 0:
        raise ValueError("The snippet length must be positive.")

    inputs = _chunk_inputs(chunks)
    if not inputs:
        raise ValueError("At least one indexed chunk is required.")

    path = _required_path(inputs[0])
    sections: list[MatchedSection] = []
    tags: set[str] = set()
    mime_types: set[str] = set()
    stale = False
    seen_chunks: set[str] = set()
    for item in inputs:
        item_path = _required_path(item)
        if item_path != path:
            continue
        tags.update(_tags(item))
        mime_type = _mime_type(item)
        if mime_type is not None:
            mime_types.add(mime_type)
        stale = stale or _is_stale(item)
        chunk_id = _required_text(item, "stable_id", "chunk_id")
        if chunk_id in seen_chunks:
            continue
        seen_chunks.add(chunk_id)
        sections.append(
            MatchedSection(
                chunk_id=chunk_id,
                page_number=_required_int(item, "page_number"),
                heading=_optional_text(item, "heading"),
                section=_optional_text(item, "section"),
                snippet=bounded_snippet(
                    _required_text(item, "text", "snippet", "excerpt"),
                    snippet_length,
                ),
            )
        )

    if not sections:
        raise ValueError("The indexed chunks did not contain usable evidence.")
    return _result_from_sections(
        path=path,
        tags=tags,
        mime_type=next(iter(sorted(mime_types)), None),
        sections=sections,
        stale=stale,
        source=source,
        rank=rank,
        score=score,
    )


def merge_results(
    results: Iterable[SearchResult],
    *,
    limit: int | None = None,
) -> tuple[SearchResult, ...]:
    if limit is not None and limit < 0:
        raise ValueError("The result limit cannot be negative.")
    if limit == 0:
        return ()
    ordered_results = sorted(
        results,
        key=_result_order_key,
    )
    grouped: dict[str, list[SearchResult]] = {}
    for result in ordered_results:
        grouped.setdefault(result.path, []).append(result)

    merged: list[SearchResult] = []
    for path, file_results in grouped.items():
        sections: list[MatchedSection] = []
        tags: set[str] = set()
        mime_types: set[str] = set()
        seen_chunks: set[str] = set()
        sources: set[SearchSource] = set()
        for result in file_results:
            tags.update(result.tags)
            if result.mime_type is not None:
                mime_types.add(result.mime_type)
            sources.add(result.source)
            for section in result.matched_sections:
                if section.chunk_id in seen_chunks:
                    continue
                seen_chunks.add(section.chunk_id)
                sections.append(section)
        source = _merge_sources(sources)
        score = max(item.score for item in file_results)
        merged.append(
            _result_from_sections(
                path=path,
                tags=tags,
                mime_type=next(iter(sorted(mime_types, key=str.casefold)), None),
                sections=sections,
                stale=any(item.stale for item in file_results),
                source=source,
                rank=len(merged) + 1,
                score=score if score <= 1.0 else 1.0,
            )
        )
        if limit is not None and len(merged) >= limit:
            break
    return tuple(merged)


def bounded_snippet(
    text: str,
    limit: int = DEFAULT_SNIPPET_LENGTH,
) -> str:
    if limit <= 0:
        raise ValueError("The snippet length must be positive.")
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    if limit <= 3:
        return "." * limit
    prefix = compact[: limit - 3].rstrip()
    boundary = prefix.rfind(" ")
    if boundary >= (limit - 3) // 2:
        prefix = prefix[:boundary]
    return f"{prefix}..."


def _result_from_sections(
    *,
    path: str,
    tags: set[str],
    mime_type: str | None,
    sections: list[MatchedSection],
    stale: bool,
    source: SearchSource,
    rank: int,
    score: float,
) -> SearchResult:
    primary = sections[0]
    citations = tuple(
        SearchCitation(
            citation_id=_citation_id(path, section.chunk_id),
            path=path,
            page_number=section.page_number,
            chunk_id=section.chunk_id,
            heading=section.heading,
            section=section.section,
            excerpt=section.snippet,
        )
        for section in sections
    )
    return SearchResult(
        chunk_id=primary.chunk_id,
        file_path=path,
        page_number=primary.page_number,
        heading=primary.heading,
        text=primary.snippet,
        source=source,
        rank=rank,
        score=score,
        tags=tuple(sorted(tags, key=lambda tag: (tag.casefold(), tag))),
        mime_type=mime_type,
        summary=primary.snippet,
        matched_sections=tuple(sections),
        citations=citations,
        stale=stale,
    )


def _chunk_inputs(chunks: object | Iterable[object]) -> tuple[object, ...]:
    if isinstance(chunks, (str, bytes, Mapping)):
        return (chunks,)
    if isinstance(chunks, SearchCandidate):
        return (chunks,)
    if isinstance(chunks, Iterable):
        return tuple(chunks)
    return (chunks,)


def _required_path(item: object) -> str:
    value = _first_value(item, "path", "file_path", "canonical_path")
    if value is None:
        page = _first_value(item, "page")
        document = _first_value(page, "document") if page is not None else None
        file = _first_value(document, "file") if document is not None else None
        value = _first_value(file, "canonical_path", "path", "file_path")
    if value is None or not str(value).strip():
        raise ValueError("The indexed chunk has no file path.")
    return str(value)


def _required_text(item: object, *names: str) -> str:
    value = _first_value(item, *names)
    if value is None:
        raise ValueError(f"The indexed chunk has no {' or '.join(names)}.")
    text = str(value)
    if not text.strip():
        raise ValueError(f"The indexed chunk has an empty {' or '.join(names)}.")
    return text


def _optional_text(item: object, name: str) -> str | None:
    value = _first_value(item, name)
    return str(value) if value is not None else None


def _required_int(item: object, name: str) -> int:
    value = _first_value(item, name)
    if value is None:
        page = _first_value(item, "page")
        value = _first_value(page, name)
    if value is None:
        raise ValueError(f"The indexed chunk has no {name}.")
    return int(str(value))


def _tags(item: object) -> tuple[str, ...]:
    tags = _first_value(item, "tags")
    if tags is None:
        page = _first_value(item, "page")
        document = _first_value(page, "document") if page is not None else None
        file = _first_value(document, "file") if document is not None else None
        tags = _first_value(file, "tags")
    if tags is None:
        return ()
    if isinstance(tags, str):
        try:
            decoded = json.loads(tags)
        except json.JSONDecodeError:
            return (tags,) if tags.strip() else ()
        tags = decoded
    if not isinstance(tags, Iterable) or isinstance(tags, (str, bytes, Mapping)):
        return ()
    values: list[str] = []
    for tag in tags:
        name = _first_value(tag, "name")
        if name is None:
            related = _first_value(tag, "tag")
            name = _first_value(related, "name") if related is not None else tag
        if name is not None and str(name).strip():
            values.append(str(name))
    return tuple(values)


def _is_stale(item: object) -> bool:
    explicit = _first_value(item, "stale")
    if explicit is not None:
        return bool(explicit)
    state = _first_value(item, "indexing_state")
    if state is None:
        page = _first_value(item, "page")
        document = _first_value(page, "document") if page is not None else None
        file = _first_value(document, "file") if document is not None else None
        state = _first_value(file, "indexing_state")
    return state is not None and str(state) != "indexed"


def _mime_type(item: object) -> str | None:
    value = _first_value(item, "mime_type")
    if value is None:
        page = _first_value(item, "page")
        document = _first_value(page, "document") if page is not None else None
        file = _first_value(document, "file") if document is not None else None
        value = _first_value(file, "mime_type")
    if value is None or not str(value).strip():
        return None
    return str(value)


def _first_value(item: object | None, *names: str) -> object | None:
    if item is None:
        return None
    for name in names:
        if isinstance(item, Mapping):
            value = cast(Mapping[object, object], item).get(name)
        else:
            value = getattr(item, name, None)
        if value is not None:
            return value
    return None


def _result_order_key(
    result: SearchResult,
) -> tuple[
    int,
    float,
    str,
    str,
    int,
    tuple[bool, str],
    str,
    str,
    str,
    tuple[str, ...],
    tuple[bool, str],
    bool,
    tuple[tuple[str, int, tuple[bool, str], tuple[bool, str], str], ...],
]:
    return (
        result.rank,
        -result.score,
        result.path,
        result.chunk_id,
        result.page_number,
        _optional_order_key(result.heading),
        result.text,
        result.source,
        result.summary,
        result.tags,
        _optional_order_key(result.mime_type),
        result.stale,
        tuple(_section_order_key(section) for section in result.matched_sections),
    )


def _section_order_key(
    section: MatchedSection,
) -> tuple[str, int, tuple[bool, str], tuple[bool, str], str]:
    return (
        section.chunk_id,
        section.page_number,
        _optional_order_key(section.heading),
        _optional_order_key(section.section),
        section.snippet,
    )


def _optional_order_key(value: str | None) -> tuple[bool, str]:
    return value is not None, value or ""


def _citation_id(path: str, chunk_id: str) -> str:
    encoded_path = urlsafe_b64encode(path.encode("utf-8")).rstrip(b"=").decode()
    encoded_chunk = urlsafe_b64encode(chunk_id.encode("utf-8")).rstrip(b"=").decode()
    return f"source-{encoded_path}.{encoded_chunk}"


def _merge_sources(sources: set[SearchSource]) -> SearchSource:
    if len(sources) > 1 or "hybrid" in sources:
        return "hybrid"
    return next(iter(sources), "fts")
