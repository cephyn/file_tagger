"""Module for search-related utilities."""

import re
import os
from typing import List, Dict, Any, Set


class SearchUtils:
    @staticmethod
    def expand_query(query: str) -> List[str]:
        """
        Expand the query with synonyms or related terms to improve recall.
        Returns list of expanded queries to use.
        
        Example: "document text" -> ["document text", "document content", "file text"]
        
        The original query is always included as the first item.
        
        Args:
            query: The original search query
            
        Returns:
            List[str]: Expanded queries list
        """
        expanded = [query]
        
        # For short queries (1-2 words), add some common synonyms
        words = query.lower().split()
        if len(words) <= 3:
            # Common synonym mappings for document search
            synonyms = {
                'document': ['file', 'content', 'text', 'doc'],
                'text': ['content', 'document', 'writing', 'words'],
                'image': ['picture', 'photo', 'jpg', 'jpeg', 'png'],
                'video': ['movie', 'mp4', 'film', 'clip'],
                'pdf': ['document', 'acrobat', 'paper'],
                'presentation': ['slides', 'powerpoint', 'ppt', 'slideshow'],
                'spreadsheet': ['excel', 'xls', 'xlsx', 'table', 'worksheet'],
                'code': ['source', 'programming', 'script', 'py', 'js'],
                'email': ['mail', 'message', 'correspondence'],
                'music': ['audio', 'song', 'mp3', 'sound'],
                'important': ['critical', 'essential', 'key', 'urgent'],
                'report': ['analysis', 'summary', 'document', 'results'],
                'search': ['find', 'query', 'lookup', 'locate'],
                'folder': ['directory', 'collection', 'group'],
                'old': ['archive', 'outdated', 'previous'],
                'new': ['recent', 'latest', 'current', 'updated']
            }
            
            # Create expanded queries by replacing one word at a time
            for i, word in enumerate(words):
                if word in synonyms:
                    for synonym in synonyms[word][:2]:  # Limit to 2 synonyms per word to avoid explosion
                        # Create a new query with this word replaced by synonym
                        new_query = words.copy()
                        new_query[i] = synonym
                        expanded.append(" ".join(new_query))
        
        return expanded[:3]  # Limit to 3 total queries including original

    @staticmethod
    def get_document_type_label(file_path: str) -> str:
        """Return a human-readable document type label based on file extension"""
        ext = os.path.splitext(file_path)[1].lower()
        
        # Common document types
        if ext in ['.py']:
            return "Python Script"
        elif ext in ['.js', '.ts']:
            return "JavaScript/TypeScript"
        elif ext in ['.html', '.htm', '.css']:
            return "Web Document"
        elif ext in ['.md', '.markdown', '.txt']:
            return "Text Document"
        elif ext in ['.pdf']:
            return "PDF Document"
        elif ext in ['.docx', '.doc']:
            return "Word Document" 
        elif ext in ['.xlsx', '.xls', '.csv']:
            return "Spreadsheet"
        elif ext in ['.pptx', '.ppt']:
            return "Presentation"
        elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            return "Image"
        elif ext in ['.mp3', '.wav', '.flac']:
            return "Audio"
        elif ext in ['.mp4', '.mov', '.avi']:
            return "Video"
        elif ext in ['.java', '.c', '.cpp', '.h', '.cs', '.go', '.rb']:
            return "Source Code"
        elif ext in ['.json', '.xml', '.yaml', '.yml']:
            return "Data File"
        elif ext in ['.sql']:
            return "Database File"
        else:
            return "Document"