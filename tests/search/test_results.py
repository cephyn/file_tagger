import json
from pathlib import Path
import struct
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from file_tagger.infrastructure.database import create_session_factory, upgrade_database
from file_tagger.persistence.models import Chunk, Document, File, FileTag, Page, Tag
from file_tagger.search.hybrid import HybridSearch
from file_tagger.search.results import (
    DEFAULT_SNIPPET_LENGTH,
    SearchResult,
    make_result,
    merge_results,
)

DIMENSIONS = 256


def _vector(*values: float) -> tuple[float, ...]:
    return (*values, *((0.0,) * (DIMENSIONS - len(values))))


def _seed_chunk(
    session: Session,
    *,
    path: Path,
    stable_id: str,
    page_number: int,
    text_value: str,
    heading: str,
    section: str,
    vector: tuple[float, ...],
    indexing_state: str = "indexed",
) -> Chunk:
    canonical_path = str(path.resolve())
    file = session.scalar(select(File).where(File.canonical_path == canonical_path))
    if file is None:
        file = File(
            canonical_path=canonical_path,
            mime_type="application/pdf",
            indexing_state=indexing_state,
        )
        session.add(file)
        session.flush()
        document = Document(
            file_id=file.id,
            revision=1,
            file_hash=f"hash-{path.name}",
            extraction_version="test",
            parser="test",
            page_count=max(page_number, 1),
            extraction_status="complete",
            diagnostics=[],
        )
        session.add(document)
        session.flush()
    else:
        file.indexing_state = indexing_state
        document = session.scalar(
            select(Document).where(Document.file_id == file.id)
        )
        assert document is not None

    page = session.scalar(
        select(Page).where(
            Page.document_id == document.id,
            Page.page_number == page_number,
        )
    )
    if page is None:
        page = Page(
            document_id=document.id,
            page_number=page_number,
            ordinal=page_number - 1,
            text=text_value,
            markdown=text_value,
            headings=[heading],
            quality="good",
        )
        session.add(page)
        session.flush()

    next_ordinal = len(
        list(session.scalars(select(Chunk).where(Chunk.document_id == document.id)))
    )
    chunk = Chunk(
        stable_id=stable_id,
        document_id=document.id,
        page_id=page.id,
        ordinal=next_ordinal,
        text=text_value,
        heading=heading,
        section=section,
        content_hash=f"content-{stable_id}",
        model_digest="sha256:test",
        index_version="test-v1",
    )
    session.add(chunk)
    session.flush()
    session.execute(
        text(
            f"INSERT INTO chunk_embeddings_{DIMENSIONS} "
            "(chunk_id, embedding) VALUES (:chunk_id, :embedding)"
        ),
        {
            "chunk_id": chunk.id,
            "embedding": struct.pack(f"<{DIMENSIONS}f", *vector),
        },
    )
    return chunk


def _chunk_data(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "file_path": r"C:\library\adventure.pdf",
        "tags": ["fiction", "travel"],
        "stable_id": "adventure-page-12",
        "page_number": 12,
        "heading": "The crossing",
        "section": "A storm at sea",
        "text": "The expedition crossed the sea before nightfall.",
        "indexing_state": "indexed",
    }
    data.update(overrides)
    return data


def test_search_result_cites_page_and_chunk() -> None:
    result = make_result(_chunk_data())

    citation = result.citations[0]
    assert citation.path.endswith("adventure.pdf")
    assert citation.page_number == 12
    assert citation.chunk_id


def test_make_result_bounds_indexed_text_and_merges_duplicate_chunks() -> None:
    text_value = "indexed evidence " * 100
    duplicate = _chunk_data(text=text_value)

    result = make_result([duplicate, duplicate])

    assert isinstance(result, SearchResult)
    assert len(result.summary) <= DEFAULT_SNIPPET_LENGTH
    assert all(len(section.snippet) <= DEFAULT_SNIPPET_LENGTH for section in result.matched_sections)
    assert [citation.chunk_id for citation in result.citations] == [
        "adventure-page-12"
    ]
    assert not result.stale


def test_hybrid_search_hydrates_one_cited_result_per_file(
    tmp_path: Path,
) -> None:
    database = tmp_path / "results.sqlite3"
    upgrade_database(database)
    factory = create_session_factory(database)
    file_path = tmp_path / "adventure.pdf"
    with factory() as session:
        _seed_chunk(
            session,
            path=file_path,
            stable_id="adventure-opening",
            page_number=12,
            text_value="The adventure starts with a map and a compass.",
            heading="The departure",
            section="The opening",
            vector=_vector(1.0, 0.0),
            indexing_state="stale",
        )
        _seed_chunk(
            session,
            path=file_path,
            stable_id="adventure-crossing",
            page_number=13,
            text_value="The adventure continues across the mountain pass.",
            heading="The crossing",
            section="The ascent",
            vector=_vector(0.9, 0.1),
            indexing_state="stale",
        )
        file = session.scalar(
            select(File).where(File.canonical_path == str(file_path.resolve()))
        )
        assert file is not None
        for name in ("fiction", "travel"):
            tag = Tag(name=name)
            session.add(tag)
            session.flush()
            session.add(FileTag(file_id=file.id, tag_id=tag.id))
        session.commit()

        results = HybridSearch(session).search("adventure", _vector(1.0, 0.0))

        assert len(results) == 1
        result = results[0]
        assert result.file_path.endswith("adventure.pdf")
        assert result.stale
        assert result.tags == ("fiction", "travel")
        headings = [section.heading for section in result.matched_sections]
        assert headings == ["The crossing", "The departure"]
        assert headings == [
            section.heading
            for section in HybridSearch(session)
            .search("adventure", _vector(1.0, 0.0))[0]
            .matched_sections
        ]
        assert {citation.chunk_id for citation in result.citations} == {
            "adventure-opening",
            "adventure-crossing",
        }
        stored_ids = set(
            session.scalars(select(Chunk.stable_id))
        )
        assert all(citation.chunk_id in stored_ids for citation in result.citations)
        assert result.summary == result.matched_sections[0].snippet


def test_result_accepts_hydrated_json_tags() -> None:
    result = make_result(_chunk_data(tags=json.dumps(["fiction", "travel"])))

    assert result.tags == ("fiction", "travel")


def test_duplicate_results_merge_in_rank_order() -> None:
    first = make_result(
        _chunk_data(
            file_path=r"C:\library\first.pdf",
            stable_id="first",
        ),
        source="fts",
        rank=1,
        score=0.02,
    )
    duplicate = make_result(
        [
            _chunk_data(
                file_path=r"C:\library\first.pdf",
                stable_id="first",
            ),
            _chunk_data(
                file_path=r"C:\library\first.pdf",
                stable_id="first-detail",
                page_number=13,
            ),
        ],
        source="vector",
        rank=1,
        score=0.01,
    )
    second = make_result(
        _chunk_data(
            file_path=r"C:\library\second.pdf",
            stable_id="second",
        ),
        rank=2,
        score=0.01,
    )

    expected = merge_results([first, duplicate, second])
    actual = merge_results([second, duplicate, first])

    assert actual == expected
    assert [item.path for item in actual] == [
        r"C:\library\first.pdf",
        r"C:\library\second.pdf",
    ]
    assert actual[0].source == "hybrid"
    assert [section.chunk_id for section in actual[0].matched_sections] == [
        "first",
        "first-detail",
    ]
