from pathlib import Path
from types import SimpleNamespace

from file_tagger.extraction.docling_adapter import extract_document


class FakeDocument:
    def __init__(self, pages: dict[int, str]) -> None:
        self.pages = {number: object() for number in pages}
        self._pages = pages

    def export_to_markdown(self, *, page_no=None, **kwargs):
        if page_no is not None:
            return self._pages.get(page_no, "")
        return "\n\n".join(self._pages.values())


class FakeConverter:
    def __init__(self, pages: dict[int, str], *, status="success", errors=()):
        self.pages = pages
        self.status = status
        self.errors = errors
        self.convert_calls = []

    def convert(self, source, raises_on_error=False):
        self.convert_calls.append((source, raises_on_error))
        return SimpleNamespace(
            status=SimpleNamespace(value=self.status),
            document=FakeDocument(self.pages),
            errors=[
                SimpleNamespace(error_message=message)
                for message in self.errors
            ],
        )


def _write_source(tmp_path: Path, suffix: str = ".pdf") -> Path:
    source = tmp_path / f"sample{suffix}"
    source.write_bytes(b"sample document bytes")
    return source


def test_pdf_extraction_preserves_page_provenance(tmp_path: Path) -> None:
    source = _write_source(tmp_path)
    converter = FakeConverter(
        {1: "# Title\n\nFirst page", 2: "## Details\n\nSecond page"}
    )

    document = extract_document(
        source,
        converter_factory=lambda use_ocr: converter,
    )

    assert document.parser == "docling"
    assert document.status == "complete"
    assert [page.number for page in document.pages] == [1, 2]
    assert document.pages[0].markdown.startswith("# Title")
    assert document.pages[1].headings == ("Details",)
    assert document.full_markdown.startswith("# Title")


def test_empty_pdf_text_retries_with_ocr(tmp_path: Path) -> None:
    source = _write_source(tmp_path)
    native = FakeConverter({1: "<!-- image -->"})
    ocr = FakeConverter({1: "# Scanned title\n\nRecognized text"})
    converters = iter((native, ocr))

    document = extract_document(
        source,
        converter_factory=lambda use_ocr: next(converters),
    )

    assert document.used_ocr is True
    assert document.pages[0].quality == "ocr"
    assert "Recognized text" in document.full_markdown


def test_partial_conversion_returns_visible_warnings(tmp_path: Path) -> None:
    source = _write_source(tmp_path, ".docx")
    converter = FakeConverter(
        {1: "Recovered content"},
        status="partial_success",
        errors=("One table could not be reconstructed",),
    )

    document = extract_document(
        source,
        converter_factory=lambda use_ocr: converter,
    )

    assert document.status == "warnings"
    assert any("table" in diagnostic.message for diagnostic in document.diagnostics)


def test_non_paginated_docling_document_exports_as_one_page(tmp_path: Path) -> None:
    source = _write_source(tmp_path, ".md")

    class NonPaginatedDocument:
        pages = {}

        @staticmethod
        def export_to_markdown(**kwargs):
            return "# Markdown\n\nBody text" if kwargs.get("page_no") is None else ""

    converter = SimpleNamespace(
        convert=lambda source, raises_on_error: SimpleNamespace(
            status=SimpleNamespace(value="success"),
            document=NonPaginatedDocument(),
            errors=[],
        )
    )

    document = extract_document(
        source,
        converter_factory=lambda use_ocr: converter,
    )

    assert document.page_count == 1
    assert document.full_markdown.startswith("# Markdown")
def test_encrypted_and_unsupported_documents_have_typed_failures(
    tmp_path: Path,
) -> None:
    encrypted = _write_source(tmp_path)
    unsupported = tmp_path / "sample.bin"
    unsupported.write_bytes(b"unknown")
    encrypted_converter = FakeConverter(
        {},
        status="failure",
        errors=("The document is encrypted and requires a password",),
    )

    encrypted_document = extract_document(
        encrypted,
        converter_factory=lambda use_ocr: encrypted_converter,
    )
    unsupported_document = extract_document(
        unsupported,
        converter_factory=lambda use_ocr: FakeConverter(
            {},
            status="failure",
            errors=("No recognizable format was detected",),
        ),
    )

    assert encrypted_document.failure == "encrypted"
    assert encrypted_document.status == "failed"
    assert encrypted_document.diagnostics[0].severity == "error"
    assert unsupported_document.failure == "unsupported"


def test_docling_adapter_extracts_modern_text(
    tmp_path: Path, monkeypatch
) -> None:
    source = _write_source(tmp_path, ".txt")
    converter = FakeConverter({1: "# Local text"})
    extracted = extract_document(
        source,
        converter_factory=lambda use_ocr: converter,
    )
    content = extracted.full_markdown

    assert content is not None
    assert "# Local text" in content


def test_real_pdf_fixture_extracts_text() -> None:
    source = Path(__file__).resolve().parents[1] / "fixtures" / "library" / "encounters.pdf"

    document = extract_document(source)

    assert document.status in {"complete", "warnings"}
    assert document.full_markdown.strip()
