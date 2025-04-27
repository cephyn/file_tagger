"""Module for extracting content from different file types."""

import os
import mimetypes
import traceback
import threading
from typing import Optional, Callable, Dict, Any
from PySide6.QtCore import QObject, Signal
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.datamodel.base_models import InputFormat


class ContentExtractionWorker(QObject):
    """Worker that handles content extraction in a background thread."""
    finished = Signal(dict)  # Signal emitted when extraction completes
    progress = Signal(str)   # Signal emitted to update progress message

    def __init__(self, file_path: str, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.thread = None
        self.result = {}
    
    def extract_content(self):
        """Extract content in a background thread and emit signals with results."""
        mime_type, _ = mimetypes.guess_type(self.file_path)
        content = ""

        try:
            if mime_type and mime_type.startswith("text/"):
                # Handle text files including markdown
                self.progress.emit(f"Reading text file: {os.path.basename(self.file_path)}")
                with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            elif mime_type == "application/pdf" or self.file_path.lower().endswith(('.pdf')):
                try:
                    # Use docling for PDF extraction
                    self.progress.emit(f"Extracting content from PDF: {os.path.basename(self.file_path)}")
                    converter = DocumentConverter()
                    doc = converter.convert(self.file_path)
                    content = doc.document.export_to_markdown()
                except Exception as pdf_error:
                    print(f"Error extracting PDF text with docling: {str(pdf_error)}")
                    # Fallback to simple placeholder
                    content = f"PDF document: {os.path.basename(self.file_path)}"
            elif self.file_path.lower().endswith('.md'):
                # Handle markdown files explicitly
                self.progress.emit(f"Reading markdown file: {os.path.basename(self.file_path)}")
                with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            elif self.file_path.lower().endswith(('.doc', '.docx')):
                # Use docling for DOCX extraction
                try:
                    self.progress.emit(f"Extracting content from Word document: {os.path.basename(self.file_path)}")
                    converter = DocumentConverter()
                    doc = converter.convert(self.file_path)
                    content = doc.document.export_to_markdown()
                except Exception as docx_error:
                    print(f"Error extracting Word document text with docling: {str(docx_error)}")
                    content = f"Word document: {os.path.basename(self.file_path)}"
            else:
                # For other file types, just use the filename for indexing
                self.progress.emit(f"Processing file: {os.path.basename(self.file_path)}")
                content = f"File: {os.path.basename(self.file_path)}"

            # Include filename in content for better matching
            filename = os.path.basename(self.file_path)
            content = f"Filename: {filename}\n\n{content or 'No extractable text content'}"

            self.result = {'success': True, 'content': content}
            self.finished.emit(self.result)

        except Exception as e:
            print(f"Error extracting content from {self.file_path}: {str(e)}")
            traceback.print_exc()
            self.result = {'success': False, 'error': str(e), 'content': f"Error extracting content: {os.path.basename(self.file_path)}"}
            self.finished.emit(self.result)


class ContentExtractor:
    @staticmethod
    def extract_file_content(file_path: str) -> Optional[str]:
        """
        Extract searchable content from a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            str: Extracted content or None if extraction failed
        """
        mime_type, _ = mimetypes.guess_type(file_path)
        content = ""

        try:
            if mime_type and mime_type.startswith("text/"):
                # Handle text files including markdown
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            elif mime_type == "application/pdf" or file_path.lower().endswith(('.pdf')):
                try:
                    # Use docling for PDF extraction
                    accelerator_options = AcceleratorOptions(
                        num_threads=6, device=AcceleratorDevice.AUTO
                    )
                    pipeline_options = PdfPipelineOptions()
                    pipeline_options.accelerator_options = accelerator_options
                    converter = DocumentConverter(
                        format_options={
                            InputFormat.PDF: PdfFormatOption(
                                pipeline_options=pipeline_options,
                            )
                        }
                    )
                    doc = converter.convert(file_path)
                    content = doc.document.export_to_markdown()
                except Exception as pdf_error:
                    print(f"Error extracting PDF text with docling: {str(pdf_error)}")
                    # Fallback to simple placeholder
                    content = f"PDF document: {os.path.basename(file_path)}"
            elif file_path.lower().endswith('.md'):
                # Handle markdown files explicitly
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            elif file_path.lower().endswith(('.doc', '.docx')):
                # Use docling for DOCX extraction
                try:
                    converter = DocumentConverter()
                    doc = converter.convert(file_path)
                    content = doc.document.export_to_markdown()
                except Exception as docx_error:
                    print(f"Error extracting Word document text with docling: {str(docx_error)}")
                    content = f"Word document: {os.path.basename(file_path)}"
            else:
                # For other file types, just use the filename for indexing
                content = f"File: {os.path.basename(file_path)}"

            # Include filename in content for better matching
            filename = os.path.basename(file_path)
            content = f"Filename: {filename}\n\n{content or 'No extractable text content'}"

            return content

        except Exception as e:
            print(f"Error extracting content from {file_path}: {str(e)}")
            traceback.print_exc()
            return f"Error extracting content: {os.path.basename(file_path)}"
            
    @staticmethod
    def extract_file_content_async(file_path: str, on_complete: Callable[[Dict[str, Any]], None], on_progress: Callable[[str], None] = None) -> None:
        """
        Extract content asynchronously with progress feedback.
        
        Args:
            file_path: Path to the file to extract content from
            on_complete: Callback function that receives result dictionary with 'success' and 'content' keys
            on_progress: Optional callback function to receive progress updates
        """
        worker = ContentExtractionWorker(file_path)
        
        # Connect signals to callbacks
        worker.finished.connect(on_complete)
        if on_progress:
            worker.progress.connect(on_progress)
        
        # Create and start the thread
        thread = threading.Thread(target=worker.extract_content)
        worker.thread = thread  # Keep reference to prevent garbage collection
        thread.start()