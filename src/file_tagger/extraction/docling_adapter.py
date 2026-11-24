from collections.abc import Callable
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

from docling.datamodel.accelerator_options import (
    AcceleratorDevice,
    AcceleratorOptions,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.exceptions import ConversionError
from pydantic import BaseModel, ConfigDict, Field

from .quality import (
    ExtractionDiagnostic,
    classify_failure,
    diagnostics_from_errors,
    extract_headings,
    useful_text_length,
)
from .vision_candidates import (
    VisionCandidate,
    VisualAugmentationResult,
    classify_page,
    select_vision_candidates,
)


ConversionStatus = Literal["complete", "warnings", "failed"]
FailureKind = Literal[
    "encrypted",
    "ocr_failed",
    "unsupported",
    "extraction_failed",
]
PageQuality = Literal["good", "degraded", "ocr"]
MINIMUM_PAGE_TEXT_CHARS = 10


class ExtractedPage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    number: int = Field(gt=0)
    ordinal: int = Field(ge=0)
    text: str
    markdown: str
    headings: tuple[str, ...] = ()
    quality: PageQuality = "good"
    text_chars: int = Field(default=0, ge=0)
    image_area: float = Field(default=0, ge=0, le=1)
    table_area: float = Field(default=0, ge=0, le=1)
    tables: int = Field(default=0, ge=0)
    table_structure_valid: bool = True

    @property
    def vision_candidate(self) -> VisionCandidate:
        return classify_page(
            page_number=self.number,
            text_chars=self.text_chars,
            image_area=self.image_area,
            table_area=self.table_area,
            tables=self.tables,
            table_structure_valid=self.table_structure_valid,
            low_quality_text=self.quality in {"degraded", "ocr"},
        )


class ExtractedDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    file_path: str = Field(min_length=1)
    file_hash: str = Field(min_length=1)
    parser: str = "docling"
    parser_version: str
    status: ConversionStatus
    pages: tuple[ExtractedPage, ...] = ()
    full_markdown: str = ""
    used_ocr: bool = False
    diagnostics: tuple[ExtractionDiagnostic, ...] = ()
    failure: FailureKind | None = None

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def classify_visual_augmentation(
        self,
        embedding_profile: object,
    ) -> VisualAugmentationResult:
        return select_vision_candidates(
            (page.vision_candidate for page in self.pages),
            embedding_profile=embedding_profile,
        )


ConverterFactory = Callable[[bool], Any]


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _default_converter(use_ocr: bool) -> DocumentConverter:
    accelerator_options = AcceleratorOptions(
        num_threads=6,
        device=AcceleratorDevice.AUTO,
    )
    pipeline_options = PdfPipelineOptions(
        do_ocr=use_ocr,
        do_table_structure=True,
        accelerator_options=accelerator_options,
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )


def _status_value(result: Any) -> str:
    status = getattr(result, "status", "")
    value = getattr(status, "value", status)
    return str(value)


def _error_messages(result: Any) -> tuple[str, ...]:
    return tuple(
        str(message)
        for item in getattr(result, "errors", ())
        if (message := getattr(item, "error_message", None))
    )


def _conversion_failure(
    path: Path,
    file_hash: str,
    messages: tuple[str, ...],
) -> ExtractedDocument:
    classified_failure = classify_failure(messages)
    if classified_failure == "encrypted":
        failure: FailureKind = "encrypted"
    elif classified_failure == "ocr_failed":
        failure = "ocr_failed"
    elif classified_failure == "unsupported":
        failure = "unsupported"
    else:
        failure = "extraction_failed"
    diagnostics = diagnostics_from_errors(messages, severity="error") or (
        ExtractionDiagnostic(
            code=failure,
            message="Docling conversion did not produce a document.",
            severity="error",
        ),
    )
    return ExtractedDocument(
        file_path=str(path),
        file_hash=file_hash,
        parser_version=version("docling"),
        status="failed",
        diagnostics=diagnostics,
        failure=failure,
    )


def _page_numbers(document: Any) -> list[int]:
    pages = getattr(document, "pages", {})
    if hasattr(pages, "keys"):
        return sorted(int(number) for number in pages.keys())
    return []


def _page_item(document: Any, page_number: int) -> Any:
    pages = getattr(document, "pages", {})
    if not hasattr(pages, "get"):
        return None
    return pages.get(page_number) or pages.get(str(page_number))


def _items_for_page(
    document: Any,
    collection_name: str,
    page_number: int,
) -> tuple[Any, ...]:
    items = getattr(document, collection_name, ())
    if not isinstance(items, (list, tuple)):
        return ()
    return tuple(
        item
        for item in items
        if any(
            getattr(provenance, "page_no", None) == page_number
            for provenance in (getattr(item, "prov", ()) or ())
        )
    )


def _box_area(box: Any) -> float:
    width = getattr(box, "width", None)
    height = getattr(box, "height", None)
    if width is not None and height is not None:
        return max(0.0, float(width)) * max(0.0, float(height))
    try:
        left = float(getattr(box, "l"))
        right = float(getattr(box, "r"))
        top = float(getattr(box, "t"))
        bottom = float(getattr(box, "b"))
    except (AttributeError, TypeError, ValueError):
        return 0.0
    return max(0.0, right - left) * max(0.0, abs(top - bottom))


def _page_area(document: Any, page_number: int) -> float:
    page = _page_item(document, page_number)
    size = getattr(page, "size", None)
    if size is None:
        return 0.0
    try:
        return max(0.0, float(size.width)) * max(0.0, float(size.height))
    except (AttributeError, TypeError, ValueError):
        return 0.0


def _items_area(
    items: tuple[Any, ...],
    *,
    page_number: int,
) -> float:
    area = 0.0
    for item in items:
        for provenance in getattr(item, "prov", ()) or ():
            if getattr(provenance, "page_no", None) == page_number:
                area += _box_area(getattr(provenance, "bbox", None))
    return area


def _coverage_ratio(
    document: Any,
    *,
    page_number: int,
    items: tuple[Any, ...],
) -> float:
    page_area = _page_area(document, page_number)
    if page_area <= 0:
        return 0.0
    return min(1.0, _items_area(items, page_number=page_number) / page_area)


def _has_table_structure(markdown: str) -> bool:
    if "<table" in markdown.casefold():
        return True
    lines = [line.strip() for line in markdown.splitlines() if line.strip()]
    for index, line in enumerate(lines[:-1]):
        separator = lines[index + 1]
        if "|" in line and "|" in separator and all(
            character in "|-: " for character in separator
        ):
            return True
    return False


def _page_layout_metrics(
    document: Any,
    *,
    page_number: int,
    markdown: str,
) -> tuple[float, float, int, bool]:
    tables = _items_for_page(document, "tables", page_number)
    pictures = _items_for_page(document, "pictures", page_number)
    if not pictures:
        pictures = _items_for_page(document, "images", page_number)
    return (
        _coverage_ratio(document, page_number=page_number, items=pictures),
        _coverage_ratio(document, page_number=page_number, items=tables),
        len(tables),
        not tables or _has_table_structure(markdown),
    )


def _extract_pages(document: Any, *, used_ocr: bool) -> tuple[ExtractedPage, ...]:
    numbers = _page_numbers(document)
    if not numbers:
        markdown = str(document.export_to_markdown() or "")
        image_area, table_area, tables, table_structure_valid = (
            _page_layout_metrics(
                document,
                page_number=1,
                markdown=markdown,
            )
        )
        return (
            ExtractedPage(
                number=1,
                ordinal=0,
                text=markdown,
                markdown=markdown,
                headings=extract_headings(markdown),
                text_chars=useful_text_length(markdown),
                image_area=image_area,
                table_area=table_area,
                tables=tables,
                table_structure_valid=table_structure_valid,
                quality=(
                    "ocr"
                    if used_ocr
                    else (
                        "good"
                        if useful_text_length(markdown) >= MINIMUM_PAGE_TEXT_CHARS
                        else "degraded"
                    )
                ),
            ),
        )
    pages: list[ExtractedPage] = []
    for ordinal, number in enumerate(numbers):
        markdown = str(document.export_to_markdown(page_no=number) or "")
        quality: PageQuality = "ocr" if used_ocr else "good"
        if useful_text_length(markdown) < MINIMUM_PAGE_TEXT_CHARS:
            quality = "degraded" if not used_ocr else "ocr"
        image_area, table_area, tables, table_structure_valid = (
            _page_layout_metrics(
                document,
                page_number=number,
                markdown=markdown,
            )
        )
        pages.append(
            ExtractedPage(
                number=number,
                ordinal=ordinal,
                text=markdown,
                markdown=markdown,
                headings=extract_headings(markdown),
                quality=quality,
                text_chars=useful_text_length(markdown),
                image_area=image_area,
                table_area=table_area,
                tables=tables,
                table_structure_valid=table_structure_valid,
            )
        )
    return tuple(pages)


def _from_result(
    path: Path,
    file_hash: str,
    result: Any,
    *,
    used_ocr: bool,
) -> ExtractedDocument:
    status_value = _status_value(result)
    errors = _error_messages(result)
    if status_value not in {"success", "partial_success"}:
        return _conversion_failure(path, file_hash, errors)

    pages = _extract_pages(getattr(result, "document"), used_ocr=used_ocr)
    full_markdown = "\n\n".join(page.markdown for page in pages if page.markdown)
    if used_ocr and not useful_text_length(full_markdown):
        return _conversion_failure(
            path,
            file_hash,
            ("OCR failed to recover usable document text.",),
        )
    diagnostics = diagnostics_from_errors(errors)
    if any(page.quality == "degraded" for page in pages):
        diagnostics += (
            ExtractionDiagnostic(
                code="low_text_quality",
                message="One or more pages contained little extractable text.",
            ),
        )

    return ExtractedDocument(
        file_path=str(path),
        file_hash=file_hash,
        parser_version=version("docling"),
        status="warnings" if diagnostics else "complete",
        pages=pages,
        full_markdown=full_markdown,
        used_ocr=used_ocr,
        diagnostics=diagnostics,
    )


class DoclingAdapter:
    def __init__(
        self,
        converter_factory: ConverterFactory | None = None,
    ) -> None:
        self.converter_factory = converter_factory or _default_converter

    def extract(self, file_path: str | Path) -> ExtractedDocument:
        path = Path(file_path).expanduser().resolve()
        file_hash = _file_hash(path)
        try:
            result = self.converter_factory(False).convert(
                path,
                raises_on_error=False,
            )
        except (ConversionError, OSError, ValueError) as exc:
            return _conversion_failure(path, file_hash, (str(exc),))

        document = _from_result(path, file_hash, result, used_ocr=False)
        if (
            path.suffix.casefold() == ".pdf"
            and document.status != "failed"
            and useful_text_length(document.full_markdown) < MINIMUM_PAGE_TEXT_CHARS
        ):
            try:
                result = self.converter_factory(True).convert(
                    path,
                    raises_on_error=False,
                )
            except (ConversionError, OSError, ValueError) as exc:
                return _conversion_failure(path, file_hash, (f"OCR failed: {exc}",))
            document = _from_result(path, file_hash, result, used_ocr=True)
        return document


def extract_document(
    file_path: str | Path,
    *,
    converter_factory: ConverterFactory | None = None,
) -> ExtractedDocument:
    return DoclingAdapter(converter_factory).extract(file_path)
