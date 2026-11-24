from pathlib import Path
import struct
from typing import Iterable

import pytest
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from file_tagger.infrastructure.database import create_session_factory, upgrade_database
from file_tagger.persistence.models import Chunk, Document, File, FileTag, Page, Tag
from file_tagger.search.hybrid import (
    HybridSearch,
    SearchFilters,
    normalize_fts_query,
)
from file_tagger.search.rrf import FusedHit, reciprocal_rank_fusion

DIMENSIONS = 256
MODEL_DIGEST = "sha256:test"
INDEX_VERSION = "test-v1"


def _vector(*values: float) -> tuple[float, ...]:
    return (*values, *((0.0,) * (DIMENSIONS - len(values))))


def _seed_chunk(
    session: Session,
    *,
    path: Path,
    stable_id: str,
    text_value: str,
    vector: tuple[float, ...],
    tags: Iterable[str] = (),
    mime_type: str = "text/markdown",
    page_number: int = 1,
    heading: str | None = None,
    revision: int = 1,
    model_digest: str = "sha256:test",
    index_version: str = "test-v1",
) -> None:
    canonical_path = str(path.resolve())
    file = session.scalar(select(File).where(File.canonical_path == canonical_path))
    if file is None:
        file = File(
            canonical_path=canonical_path,
            mime_type=mime_type,
            indexing_state="indexed",
        )
        session.add(file)
        session.flush()
        document = Document(
            file_id=file.id,
            revision=revision,
            file_hash=f"hash-{path.name}",
            extraction_version="test",
            parser="test",
            page_count=page_number,
            extraction_status="complete",
            diagnostics=[],
        )
        session.add(document)
        session.flush()
    else:
        existing_document = session.scalar(
            select(Document).where(
                Document.file_id == file.id,
                Document.revision == revision,
            )
        )
        if existing_document is None:
            document = Document(
                file_id=file.id,
                revision=revision,
                file_hash=f"hash-{path.name}-revision-{revision}",
                extraction_version="test",
                parser="test",
                page_count=page_number,
                extraction_status="complete",
                diagnostics=[],
            )
            session.add(document)
            session.flush()
        else:
            document = existing_document

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
            markdown=f"# {heading}\n\n{text_value}" if heading else text_value,
            headings=[heading] if heading else [],
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
        section=heading,
        content_hash=f"content-{stable_id}",
        model_digest=model_digest,
        index_version=index_version,
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

    for name in set(tags):
        tag = session.scalar(select(Tag).where(Tag.name == name))
        if tag is None:
            tag = Tag(name=name)
            session.add(tag)
            session.flush()
        if session.get(FileTag, (file.id, tag.id)) is None:
            session.add(FileTag(file_id=file.id, tag_id=tag.id))


@pytest.fixture
def search_session(tmp_path: Path) -> Iterable[Session]:
    database = tmp_path / "search.sqlite3"
    upgrade_database(database)
    factory = create_session_factory(database)
    with factory() as session:
        yield session


def test_rrf_preserves_complementary_lexical_and_vector_hits() -> None:
    fused = reciprocal_rank_fusion(
        {"lexical": ["exact", "shared"], "vector": ["shared", "semantic"]},
        k=60,
    )

    assert fused[0].chunk_id == "shared"
    assert {item.chunk_id for item in fused} == {"exact", "shared", "semantic"}


def test_rrf_equal_rank_ties_do_not_depend_on_source_order() -> None:
    forward = reciprocal_rank_fusion(
        {"lexical": ("zeta",), "vector": ("alpha",)},
    )
    reversed_sources = reciprocal_rank_fusion(
        {"vector": ("alpha",), "lexical": ("zeta",)},
    )

    assert [item.chunk_id for item in reversed_sources] == [
        item.chunk_id for item in forward
    ]
    assert [item.chunk_id for item in forward] == ["alpha", "zeta"]


def test_exact_term_recall_and_citation_metadata(
    search_session: Session,
    tmp_path: Path,
) -> None:
    _seed_chunk(
        search_session,
        path=tmp_path / "orion.md",
        stable_id="orion-exact",
        text_value="Project Orion has the exact launch identifier.",
        vector=_vector(0.0, 1.0),
        page_number=7,
        heading="Launch plan",
    )
    search_session.commit()

    result = HybridSearch(search_session).search(
        "Orion launch identifier",
        _vector(1.0, 0.0),
        model_digest=MODEL_DIGEST,
        index_version=INDEX_VERSION,
    )

    assert result[0].chunk_id == "orion-exact"
    assert result[0].page_number == 7
    assert result[0].heading == "Launch plan"
    assert result[0].file_path.endswith("orion.md")


def test_semantic_recall_works_without_lexical_match(
    search_session: Session,
    tmp_path: Path,
) -> None:
    _seed_chunk(
        search_session,
        path=tmp_path / "animals.md",
        stable_id="animal-semantic",
        text_value="The red panda lives in mountain forests.",
        vector=_vector(1.0, 0.0),
    )
    _seed_chunk(
        search_session,
        path=tmp_path / "finance.md",
        stable_id="finance-far",
        text_value="Quarterly revenue exceeded the forecast.",
        vector=_vector(0.0, 1.0),
    )
    search_session.commit()

    result = HybridSearch(search_session).search(
        "friendly woodland creature",
        _vector(0.99, 0.01),
        model_digest=MODEL_DIGEST,
        index_version=INDEX_VERSION,
    )

    assert result[0].chunk_id == "animal-semantic"
    assert result[0].source == "vector"


def test_vector_search_without_identity_keeps_only_lexical_results(
    search_session: Session,
    tmp_path: Path,
) -> None:
    _seed_chunk(
        search_session,
        path=tmp_path / "lexical.md",
        stable_id="lexical-only",
        text_value="The literal needle appears here.",
        vector=_vector(0.0, 1.0),
    )
    _seed_chunk(
        search_session,
        path=tmp_path / "semantic.md",
        stable_id="semantic-only",
        text_value="A semantically related document.",
        vector=_vector(1.0, 0.0),
    )
    search_session.commit()

    result = HybridSearch(search_session).search("needle", _vector(1.0, 0.0))

    assert [candidate.chunk_id for candidate in result] == ["lexical-only"]
    assert result[0].source == "fts"


@pytest.mark.parametrize(
    ("model_digest", "index_version"),
    [
        ("sha256:other-model", "test-v1"),
        ("sha256:test", "test-v2"),
    ],
)
def test_vector_search_excludes_same_dimension_different_identity(
    search_session: Session,
    tmp_path: Path,
    model_digest: str,
    index_version: str,
) -> None:
    _seed_chunk(
        search_session,
        path=tmp_path / "different-identity.md",
        stable_id="different-identity",
        text_value="nearest result from another index",
        vector=_vector(1.0, 0.0),
        model_digest=model_digest,
        index_version=index_version,
    )
    _seed_chunk(
        search_session,
        path=tmp_path / "eligible.md",
        stable_id="eligible-identity",
        text_value="semantically close eligible result",
        vector=_vector(0.9, 0.1),
        model_digest="sha256:test",
        index_version="test-v1",
    )
    search_session.commit()

    result = HybridSearch(search_session, candidate_limit=1).search(
        "quasar ziggurat nebula",
        _vector(1.0, 0.0),
        model_digest="sha256:test",
        index_version="test-v1",
    )

    assert [candidate.chunk_id for candidate in result] == ["eligible-identity"]


def test_mixed_search_preserves_lexical_and_vector_only_hits(
    search_session: Session,
    tmp_path: Path,
) -> None:
    _seed_chunk(
        search_session,
        path=tmp_path / "exact.md",
        stable_id="exact",
        text_value="needle needle needle",
        vector=_vector(0.0, 1.0),
    )
    _seed_chunk(
        search_session,
        path=tmp_path / "shared.md",
        stable_id="shared",
        text_value="needle with context",
        vector=_vector(1.0, 0.0),
    )
    _seed_chunk(
        search_session,
        path=tmp_path / "semantic.md",
        stable_id="semantic",
        text_value="related concept without the literal term",
        vector=_vector(0.9, 0.0),
    )
    search_session.commit()

    result = HybridSearch(search_session).search(
        "needle",
        _vector(1.0, 0.0),
        model_digest=MODEL_DIGEST,
        index_version=INDEX_VERSION,
    )

    assert result[0].chunk_id == "shared"
    assert {candidate.chunk_id for candidate in result} == {
        "exact",
        "shared",
        "semantic",
    }


def test_empty_search_returns_no_candidates(
    search_session: Session,
    tmp_path: Path,
) -> None:
    _seed_chunk(
        search_session,
        path=tmp_path / "nonempty.md",
        stable_id="nonempty",
        text_value="This file is indexed.",
        vector=_vector(1.0, 0.0),
    )
    search_session.commit()

    assert HybridSearch(search_session).search("   \t", _vector(1.0, 0.0)) == ()


def test_search_applies_tag_path_and_type_filters_in_sql(
    search_session: Session,
    tmp_path: Path,
) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    _seed_chunk(
        search_session,
        path=reports / "wanted.pdf",
        stable_id="wanted",
        text_value="project atlas",
        vector=_vector(1.0, 0.0),
        tags=("work", "atlas"),
        mime_type="application/pdf",
    )
    _seed_chunk(
        search_session,
        path=reports / "wrong-tag.pdf",
        stable_id="wrong-tag",
        text_value="project atlas",
        vector=_vector(1.0, 0.0),
        tags=("personal",),
        mime_type="application/pdf",
    )
    _seed_chunk(
        search_session,
        path=tmp_path / "wrong-path.pdf",
        stable_id="wrong-path",
        text_value="project atlas",
        vector=_vector(1.0, 0.0),
        tags=("work", "atlas"),
        mime_type="application/pdf",
    )
    _seed_chunk(
        search_session,
        path=reports / "wrong-type.md",
        stable_id="wrong-type",
        text_value="project atlas",
        vector=_vector(1.0, 0.0),
        tags=("work", "atlas"),
        mime_type="text/markdown",
    )
    search_session.commit()

    result = HybridSearch(search_session).search(
        "project atlas",
        _vector(1.0, 0.0),
        tags=("work", "atlas"),
        path=str(reports),
        file_type="pdf",
        model_digest=MODEL_DIGEST,
        index_version=INDEX_VERSION,
    )

    assert [candidate.chunk_id for candidate in result] == ["wanted"]


@pytest.mark.parametrize("file_type", ["strasse", "APPLICATION/STRASSE"])
@pytest.mark.parametrize(
    ("query", "embedding"),
    [
        ("unicode needle", None),
        ("semantic phrase absent from text", _vector(1.0, 0.0)),
    ],
)
def test_unicode_filters_match_lexical_and_vector_rankings(
    search_session: Session,
    tmp_path: Path,
    file_type: str,
    query: str,
    embedding: tuple[float, ...] | None,
) -> None:
    folder = tmp_path / "Straße"
    folder.mkdir()
    _seed_chunk(
        search_session,
        path=folder / "record.STRASSE",
        stable_id="unicode-match",
        text_value="Unicode needle appears in searchable text.",
        vector=_vector(1.0, 0.0),
        tags=("Straße",),
        mime_type="application/Straße",
    )
    search_session.commit()

    result = HybridSearch(search_session).search(
        query,
        embedding,
        tags=("STRASSE",),
        path="strasse",
        file_type=file_type,
        model_digest=MODEL_DIGEST,
        index_version=INDEX_VERSION,
    )

    assert [candidate.chunk_id for candidate in result] == ["unicode-match"]


def test_unicode_filters_are_reapplied_during_hydration(
    search_session: Session,
    tmp_path: Path,
) -> None:
    folder = tmp_path / "Straße"
    folder.mkdir()
    _seed_chunk(
        search_session,
        path=folder / "record.STRASSE",
        stable_id="hydration-match",
        text_value="Hydration candidate.",
        vector=_vector(1.0, 0.0),
        tags=("Straße",),
        mime_type="application/Straße",
    )
    search_session.commit()
    fused = (
        FusedHit(
            chunk_id="hydration-match",
            rank=1,
            score=1.0,
            source_ranks={"lexical": 1},
        ),
    )

    matching = HybridSearch(search_session)._hydrate_candidates(
        fused,
        SearchFilters(
            tags=("STRASSE",),
            path="strasse",
            file_types=("APPLICATION/STRASSE",),
        ),
    )
    excluded = HybridSearch(search_session)._hydrate_candidates(
        fused,
        SearchFilters(tags=("personal",)),
    )

    assert set(matching) == {"hydration-match"}
    assert excluded == {}


def test_vector_filters_are_applied_before_the_nearest_neighbor_limit(
    search_session: Session,
    tmp_path: Path,
) -> None:
    _seed_chunk(
        search_session,
        path=tmp_path / "nearest-but-excluded.pdf",
        stable_id="excluded",
        text_value="unrelated first file",
        vector=_vector(1.0, 0.0),
        tags=("personal",),
        mime_type="application/pdf",
    )
    _seed_chunk(
        search_session,
        path=tmp_path / "eligible.pdf",
        stable_id="eligible",
        text_value="no literal match",
        vector=_vector(0.9, 0.1),
        tags=("work",),
        mime_type="application/pdf",
    )
    search_session.commit()

    result = HybridSearch(search_session, candidate_limit=1).search(
        "unsearchable semantic phrase",
        _vector(1.0, 0.0),
        tags=("work",),
        model_digest=MODEL_DIGEST,
        index_version=INDEX_VERSION,
    )

    assert [candidate.chunk_id for candidate in result] == ["eligible"]


def test_search_pins_one_revision_snapshot_across_all_queries(
    search_session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_path = tmp_path / "revisioned.md"
    _seed_chunk(
        search_session,
        path=file_path,
        stable_id="revision-one",
        text_value="needle from revision one",
        vector=_vector(1.0, 0.0),
    )
    search_session.commit()

    search = HybridSearch(
        search_session,
        model_digest=MODEL_DIGEST,
        index_version=INDEX_VERSION,
    )
    original_vector_rankings = search._vector_rankings
    revision_swapped = False

    def swap_revision_before_vector(
        query_embedding: tuple[float, ...],
        filters: SearchFilters,
        model_digest: str,
        index_version: str,
    ) -> tuple[str, ...]:
        nonlocal revision_swapped
        if not revision_swapped:
            revision_swapped = True
            with Session(bind=search_session.get_bind()) as writer:
                _seed_chunk(
                    writer,
                    path=file_path,
                    stable_id="revision-two",
                    text_value="needle from revision two",
                    vector=_vector(1.0, 0.0),
                    revision=2,
                )
                writer.commit()
        return original_vector_rankings(
            query_embedding,
            filters,
            model_digest,
            index_version,
        )

    monkeypatch.setattr(search, "_vector_rankings", swap_revision_before_vector)

    result = search.search("needle", _vector(1.0, 0.0))

    assert revision_swapped
    assert [candidate.chunk_id for candidate in result] == ["revision-one"]
    with Session(bind=search_session.get_bind()) as verifier:
        assert verifier.scalar(
            select(Document.revision)
            .where(Document.file_id == select(File.id).where(File.canonical_path == str(file_path.resolve())).scalar_subquery())
            .order_by(Document.revision.desc())
            .limit(1)
        ) == 2


def test_fts_query_is_safely_quoted_and_cannot_inject_sql(
    search_session: Session,
    tmp_path: Path,
) -> None:
    _seed_chunk(
        search_session,
        path=tmp_path / "safe.md",
        stable_id="safe",
        text_value="ordinary searchable content",
        vector=_vector(1.0, 0.0),
    )
    search_session.commit()
    payload = '" OR 1=1; DELETE FROM files; --'

    escaped = normalize_fts_query(payload)
    result = HybridSearch(search_session).search(payload, ())

    assert escaped
    assert '"' in escaped
    assert result == ()
    assert search_session.scalar(select(File).with_only_columns(File.id)) is not None


def test_file_grouping_suppresses_duplicate_file_chunks(
    search_session: Session,
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "many-chunks.md"
    _seed_chunk(
        search_session,
        path=first_path,
        stable_id="lexical-chunk",
        text_value="needle appears exactly",
        vector=_vector(0.0, 1.0),
    )
    _seed_chunk(
        search_session,
        path=first_path,
        stable_id="vector-chunk",
        text_value="semantically related material",
        vector=_vector(1.0, 0.0),
        page_number=2,
    )
    _seed_chunk(
        search_session,
        path=tmp_path / "other.md",
        stable_id="other-chunk",
        text_value="needle from another file",
        vector=_vector(0.8, 0.1),
    )
    search_session.commit()

    result = HybridSearch(search_session).search(
        "needle",
        _vector(1.0, 0.0),
        model_digest=MODEL_DIGEST,
        index_version=INDEX_VERSION,
    )

    assert len(result) == 2
    assert len({candidate.file_path for candidate in result}) == len(result)
    assert [candidate.rank for candidate in result] == [1, 2]
