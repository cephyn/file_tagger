"""Module for extracting content from different file types."""

import os
import mimetypes
import traceback
from typing import Optional


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
            elif mime_type == "application/pdf":
                try:
                    from pypdf import PdfReader
                    with open(file_path, "rb") as f:
                        reader = PdfReader(f)
                        content = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
                except Exception as pdf_error:
                    print(f"Error extracting PDF text: {str(pdf_error)}")
                    # Return a basic placeholder if PDF extraction fails
                    content = f"PDF document: {os.path.basename(file_path)}"
            elif file_path.lower().endswith('.md'):
                # Handle markdown files explicitly
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            elif file_path.lower().endswith(('.doc', '.docx')):
                # Handle Microsoft Word documents
                try:
                    import docx2txt
                    try:
                        # For .docx files
                        content = docx2txt.process(file_path)
                    except Exception as docx_error:
                        print(f"Error with docx2txt: {str(docx_error)}")
                        
                        # Fallback for .doc files or if docx2txt fails
                        try:
                            # Correct import for win32com
                            import win32com.client
                            word = win32com.client.Dispatch("Word.Application")
                            word.visible = False
                            doc = word.Documents.Open(file_path)
                            content = doc.Content.Text
                            doc.Close()
                            word.Quit()
                        except Exception as doc_error:
                            print(f"Error extracting Word document text: {str(doc_error)}")
                            content = f"Word document: {os.path.basename(file_path)}"
                except ImportError:
                    print("docx2txt module not installed")
                    content = f"Word document (extraction not available): {os.path.basename(file_path)}"
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