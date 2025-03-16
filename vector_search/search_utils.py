"""Module for search-related utilities."""

from typing import List, Dict, Any


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
    def extract_relevant_snippets(text: str, query: str, max_snippets: int = 3, snippet_length: int = 150) -> List[str]:
        """
        Extract relevant snippets from text that contain query terms.
        
        Args:
            text: The document text to search in
            query: The search query
            max_snippets: Maximum number of snippets to return
            snippet_length: Target length of each snippet in characters
            
        Returns:
            List of text snippets containing query terms
        """
        if not text or not query:
            return []
            
        # Normalize text and query for better matching
        text_lower = text.lower()
        query_terms = [term.lower() for term in query.split() if len(term) > 2]
        
        # Handle case with no meaningful query terms
        if not query_terms:
            # Return beginning of document
            return [f"{text[:snippet_length]}..."]
        
        snippets = []
        
        # Get positions of query terms in the text
        term_positions = []
        for term in query_terms:
            pos = 0
            while True:
                pos = text_lower.find(term, pos)
                if pos == -1:
                    break
                term_positions.append((pos, term))
                pos += 1
                
        # Sort positions
        term_positions.sort()
        
        # If no terms found, return beginning of document
        if not term_positions:
            return [f"{text[:snippet_length]}..."]
        
        # Group nearby positions to form snippets
        current_snippet_start = None
        current_snippet_end = None
        
        for pos, term in term_positions:
            # If we have enough snippets, stop
            if len(snippets) >= max_snippets:
                break
                
            # Calculate snippet boundaries
            start = max(0, pos - snippet_length // 2)
            end = min(len(text), pos + len(term) + snippet_length // 2)
            
            # Check if this position can extend the current snippet
            if current_snippet_start is not None and start <= current_snippet_end + snippet_length // 3:
                # Extend current snippet
                current_snippet_end = end
            else:
                # If we have a current snippet, add it to the list
                if current_snippet_start is not None:
                    # Find sentence boundaries if possible
                    refined_start = SearchUtils._find_sentence_boundary(text, current_snippet_start, False)
                    refined_end = SearchUtils._find_sentence_boundary(text, current_snippet_end, True)
                    
                    snippet = text[refined_start:refined_end]
                    # Add ellipsis if needed
                    prefix = "..." if refined_start > 0 else ""
                    suffix = "..." if refined_end < len(text) else ""
                    snippet = f"{prefix}{snippet}{suffix}"
                    
                    snippets.append(snippet)
                
                # Start new snippet
                current_snippet_start = start
                current_snippet_end = end
        
        # Add the last snippet if it exists
        if current_snippet_start is not None and len(snippets) < max_snippets:
            refined_start = SearchUtils._find_sentence_boundary(text, current_snippet_start, False)
            refined_end = SearchUtils._find_sentence_boundary(text, current_snippet_end, True)
            
            snippet = text[refined_start:refined_end]
            # Add ellipsis if needed
            prefix = "..." if refined_start > 0 else ""
            suffix = "..." if refined_end < len(text) else ""
            snippet = f"{prefix}{snippet}{suffix}"
            
            snippets.append(snippet)
            
        return snippets
    
    @staticmethod
    def _find_sentence_boundary(text: str, pos: int, find_end: bool) -> int:
        """
        Find the nearest sentence boundary (beginning or end) from the given position.
        
        Args:
            text: The text to search in
            pos: The position to start from
            find_end: If True, find the end of the sentence; if False, find the beginning
            
        Returns:
            The position of the sentence boundary
        """
        # Define sentence ending punctuation
        end_punctuation = ['.', '!', '?']
        
        if find_end:
            # Find the end of the sentence
            for i in range(pos, min(len(text), pos + 150)):
                if text[i] in end_punctuation:
                    # Find the next non-punctuation, non-whitespace character
                    for j in range(i + 1, min(len(text), i + 10)):
                        if text[j] not in end_punctuation and not text[j].isspace():
                            return j
                    return i + 1
            # If no sentence end found, return the original end position
            return min(len(text), pos + 150)
        else:
            # Find the beginning of the sentence
            for i in range(pos, max(0, pos - 150), -1):
                if i > 0 and text[i-1] in end_punctuation:
                    return i
            # If no sentence beginning found, return the original start position
            return max(0, pos - 150)