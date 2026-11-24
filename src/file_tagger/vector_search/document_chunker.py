"""Module for document chunking strategies in vector search."""

import re
from typing import List


class DocumentChunker:
    @staticmethod
    def chunk_document(text: str, min_chunk_size: int = 200, max_chunk_size: int = 1000, overlap: int = 50) -> List[str]:
        """
        Chunk document into semantic units by analyzing content structure.
        Uses paragraph breaks, headings, and other document structure cues.
        
        Args:
            text (str): The document text to chunk
            min_chunk_size (int): Minimum chunk size in characters
            max_chunk_size (int): Maximum chunk size in characters
            overlap (int): Number of characters to overlap between chunks
            
        Returns:
            List[str]: List of text chunks
        """
        if not text or len(text) <= max_chunk_size:
            return [text] if text else []
        
        # Check if we have a structured document (like markdown with headers)
        has_headers = bool(re.search(r'^#+\s+\w+', text, re.MULTILINE))
        
        # For documents with clear headers, split on headers
        if has_headers:
            return DocumentChunker._chunk_by_headers(text, min_chunk_size, max_chunk_size, overlap)
        else:
            # Otherwise split on paragraph boundaries
            return DocumentChunker._chunk_by_paragraphs(text, min_chunk_size, max_chunk_size, overlap)
    
    @staticmethod
    def _chunk_by_headers(text: str, min_chunk_size: int, max_chunk_size: int, overlap: int) -> List[str]:
        """Split document on header boundaries for more semantic chunks."""
        # Find all headers (markdown style)
        header_pattern = r'^(#+)\s+(.+)$'
        header_matches = list(re.finditer(header_pattern, text, re.MULTILINE))
        
        chunks = []
        if not header_matches:
            # Fallback to paragraph chunking if no headers found
            return DocumentChunker._chunk_by_paragraphs(text, min_chunk_size, max_chunk_size, overlap)
        
        # Process chunks between headers
        for i in range(len(header_matches)):
            start_pos = header_matches[i].start()
            
            # Find end position (next header or end of text)
            if i < len(header_matches) - 1:
                end_pos = header_matches[i+1].start()
            else:
                end_pos = len(text)
            
            header = header_matches[i].group(0)
            content = text[start_pos:end_pos]
            
            # If content is too large, sub-chunk it
            if len(content) > max_chunk_size:
                # Add the header to each sub-chunk for context
                sub_chunks = DocumentChunker._chunk_by_paragraphs(content, min_chunk_size, max_chunk_size - len(header), overlap)
                for sub_chunk in sub_chunks:
                    # Only add header if it's not already there
                    if not sub_chunk.startswith(header):
                        chunks.append(f"{header}\n{sub_chunk}")
                    else:
                        chunks.append(sub_chunk)
            else:
                chunks.append(content)
        
        # Handle the case where there's text before the first header
        if header_matches[0].start() > 0:
            prefix_text = text[:header_matches[0].start()]
            if len(prefix_text.strip()) > min_chunk_size:
                chunks.insert(0, prefix_text)
        
        return chunks
    
    @staticmethod
    def _chunk_by_paragraphs(text: str, min_chunk_size: int, max_chunk_size: int, overlap: int) -> List[str]:
        """Split document on paragraph boundaries."""
        # Split text on paragraph boundaries (empty lines)
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed max size and we have content,
            # store the current chunk and start a new one with overlap
            if len(current_chunk) + len(paragraph) + 2 > max_chunk_size and len(current_chunk) >= min_chunk_size:
                chunks.append(current_chunk)
                
                # Start new chunk with overlap from end of previous chunk
                if overlap > 0 and len(current_chunk) > overlap:
                    # Get last few sentences for overlap
                    last_sentences = re.findall(r'[^.!?]+[.!?]', current_chunk[-overlap*2:])
                    current_chunk = "".join(last_sentences[-2:]) if last_sentences else ""
                else:
                    current_chunk = ""
            
            # Add paragraph to current chunk
            if current_chunk and not current_chunk.endswith("\n"):
                current_chunk += "\n\n"
            current_chunk += paragraph
        
        # Don't forget the last chunk
        if current_chunk and len(current_chunk) >= min_chunk_size:
            chunks.append(current_chunk)
        
        return chunks
    
    @staticmethod
    def extract_chunk_title(chunk: str) -> str:
        """Extract a representative title for the chunk from its content."""
        # Look for header patterns
        header_match = re.search(r'^(#+)\s+(.+)$', chunk, re.MULTILINE)
        if header_match:
            return header_match.group(2).strip()
        
        # If no header found, use first non-empty line
        lines = chunk.split('\n')
        for line in lines:
            if line.strip():
                # Truncate to a reasonable title length
                title = line.strip()
                return title[:50] + '...' if len(title) > 50 else title
        
        # Fallback option
        return "Untitled chunk"