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


def _extract_pages(document: Any, *, used_ocr: bool) -> tuple[ExtractedPage, ...]:
    numbers = _page_numbers(document)
    if not numbers:
        markdown = str(document.export_to_markdown() or "")
        return (
            ExtractedPage(
                number=1,
                ordinal=0,
                text=markdown,
                markdown=markdown,
                headings=extract_headings(markdown),
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
        pages.append(
            ExtractedPage(
                number=number,
                ordinal=ordinal,
                text=markdown,
                markdown=markdown,
                headings=extract_headings(markdown),
                quality=quality,
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
