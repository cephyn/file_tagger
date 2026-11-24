from pathlib import Path
from types import SimpleNamespace

from file_tagger.extraction.docling_adapter import (
    ExtractedDocument,
    ExtractedPage,
    extract_document,
)
from file_tagger.extraction.vision_candidates import (
    classify_page,
    select_vision_candidates,
)


def test_born_digital_text_page_is_not_rendered() -> None:
    candidate = classify_page(text_chars=2400, image_area=0.05, tables=0)

    assert candidate.needs_vision_embedding is False


def test_diagram_only_page_is_rendered() -> None:
    candidate = classify_page(text_chars=20, image_area=0.75, tables=0)

    assert candidate.needs_vision_embedding is True
    assert "dominant visual content" in candidate.reasons


def test_dominant_table_page_is_rendered() -> None:
    candidate = classify_page(
        text_chars=1200,
        image_area=0,
        tables=1,
        table_area=0.7,
    )

    assert candidate.needs_vision_embedding is True
    assert "dominant table content" in candidate.reasons


def test_structurally_broken_table_is_rendered() -> None:
    candidate = classify_page(
        text_chars=800,
        image_area=0,
        tables=1,
        table_structure_valid=False,
    )

    assert candidate.needs_vision_embedding is True
    assert "table structure unavailable" in candidate.reasons


def test_ocr_page_with_recovered_text_remains_a_vision_candidate() -> None:
    page = ExtractedPage(
        number=1,
        ordinal=0,
        text="Recovered OCR text",
        markdown="Recovered OCR text",
        text_chars=800,
        quality="ocr",
    )

    assert page.vision_candidate.needs_vision_embedding is True
    assert "low-quality extracted text" in page.vision_candidate.reasons


def test_text_only_profile_returns_explicit_disabled_result() -> None:
    candidate = classify_page(
        text_chars=20,
        image_area=0.75,
        tables=0,
        page_number=3,
    )

    result = select_vision_candidates(
        (candidate,),
        embedding_profile="embeddinggemma-2:270m",
    )

    assert result.enabled is False
    assert result.reason == "visual augmentation disabled"
    assert result.selected_page_numbers == ()


def test_visual_profile_selects_only_flagged_pages() -> None:
    text_page = classify_page(
        text_chars=2400,
        image_area=0.05,
        tables=0,
        page_number=1,
    )
    diagram_page = classify_page(
        text_chars=20,
        image_area=0.75,
        tables=0,
        page_number=2,
    )

    result = select_vision_candidates(
        (text_page, diagram_page),
        embedding_profile="embeddinggemma-2:440m",
    )

    assert result.enabled is True
    assert result.selected_page_numbers == (2,)


def test_visual_capability_can_enable_a_future_embedding_profile() -> None:
    candidate = classify_page(
        text_chars=20,
        image_area=0.75,
        tables=0,
    )
    profile = SimpleNamespace(
        model="embeddinggemma-2:future",
        capabilities=("embedding", "vision"),
    )

    result = select_vision_candidates(
        (candidate,),
        embedding_profile=profile,
    )

    assert result.enabled is True
    assert result.selected_page_numbers == (1,)


def test_extracted_document_exposes_selective_visual_classification() -> None:
    document = ExtractedDocument(
        file_path="sample.pdf",
        file_hash="hash",
        parser_version="test",
        status="complete",
        pages=(
            ExtractedPage(
                number=1,
                ordinal=0,
                text="Text-heavy born-digital page",
                markdown="Text-heavy born-digital page",
                text_chars=2400,
                image_area=0.05,
                tables=0,
            ),
            ExtractedPage(
                number=2,
                ordinal=1,
                text="Diagram",
                markdown="Diagram",
                text_chars=20,
                image_area=0.8,
                tables=0,
            ),
        ),
    )

    result = document.classify_visual_augmentation("embeddinggemma-2:740m")

    assert result.selected_page_numbers == (2,)


def test_adapter_classifies_only_a_dominant_picture_page(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"sample")
    page_size = SimpleNamespace(width=100, height=100)
    picture_box = SimpleNamespace(l=0, r=80, t=0, b=100)
    converted_document = SimpleNamespace(
        pages={
            1: SimpleNamespace(size=page_size),
            2: SimpleNamespace(size=page_size),
        },
        pictures=(
            SimpleNamespace(
                prov=(SimpleNamespace(page_no=2, bbox=picture_box),)
            ),
        ),
        tables=(),
        export_to_markdown=lambda **kwargs: {
            1: "A" * 500,
            2: "Diagram",
        }[kwargs["page_no"]],
    )
    converter = SimpleNamespace(
        convert=lambda source, raises_on_error: SimpleNamespace(
            status=SimpleNamespace(value="success"),
            document=converted_document,
            errors=(),
        )
    )

    document = extract_document(
        source,
        converter_factory=lambda use_ocr: converter,
    )
    result = document.classify_visual_augmentation("embeddinggemma-2:440m")

    assert result.selected_page_numbers == (2,)
    assert document.pages[1].image_area == 0.8
