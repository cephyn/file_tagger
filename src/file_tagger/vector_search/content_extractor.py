"""Compatibility extraction boundary backed by the Docling adapter."""

from collections.abc import Callable
import os
import threading
from typing import Any

from PySide6.QtCore import QObject, Signal

from ..extraction.docling_adapter import DoclingAdapter


class ContentExtractionWorker(QObject):
    finished = Signal(dict)
    progress = Signal(str)

    def __init__(
        self,
        file_path: str,
        pdf_extractor: str | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        del pdf_extractor
        self.file_path = file_path
        self.extraction_thread: threading.Thread | None = None
        self.result: dict[str, Any] = {}

    def extract_content(self) -> None:
        self.progress.emit(f"Extracting with Docling: {os.path.basename(self.file_path)}")
        try:
            document = DoclingAdapter().extract(self.file_path)
        except OSError as exc:
            self.result = {
                "success": False,
                "error": str(exc),
                "content": "",
            }
            self.finished.emit(self.result)
            return

        if document.status == "failed":
            self.result = {
                "success": False,
                "error": "; ".join(item.message for item in document.diagnostics),
                "failure": document.failure,
                "content": "",
            }
        else:
            self.result = {
                "success": True,
                "content": (
                    f"Filename: {os.path.basename(self.file_path)}\n\n"
                    f"{document.full_markdown}"
                ),
                "diagnostics": [
                    item.model_dump(mode="json") for item in document.diagnostics
                ],
                "page_count": document.page_count,
                "used_ocr": document.used_ocr,
            }
        self.finished.emit(self.result)


class ContentExtractor:
    @staticmethod
    def extract_file_content(
        file_path: str,
        pdf_extractor: str | None = None,
    ) -> str | None:
        del pdf_extractor
        try:
            document = DoclingAdapter().extract(file_path)
        except OSError:
            return None
        if document.status == "failed":
            return None
        return (
            f"Filename: {os.path.basename(file_path)}\n\n"
            f"{document.full_markdown}"
        )

    @staticmethod
    def extract_file_content_async(
        file_path: str,
        on_complete: Callable[[dict[str, Any]], None],
        on_progress: Callable[[str], None] | None = None,
        pdf_extractor: str | None = None,
    ) -> None:
        worker = ContentExtractionWorker(file_path, pdf_extractor=pdf_extractor)
        worker.finished.connect(on_complete)
        if on_progress is not None:
            worker.progress.connect(on_progress)
        thread = threading.Thread(target=worker.extract_content)
        worker.extraction_thread = thread
        thread.start()
