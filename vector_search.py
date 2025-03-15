import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from models import File, Tag
import json
import traceback
import re  # Added for regex pattern matching in chunking

class VectorSearch:
    def __init__(self, db_session, collection_name: str = "file_contents"):
        self.db_session = db_session
        self.collection_name = collection_name
        print("\nInitializing vector search...")
        
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(path=".chroma")
            
            # Use a more advanced embedding model with higher dimensionality for better semantic matching
            # Options: 'all-mpnet-base-v2' (768d), 'multi-qa-mpnet-base-dot-v1' (768d), or 'all-MiniLM-L12-v2' (384d)
            self.embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")
            
            # Log ChromaDB version for debugging
            print(f"ChromaDB version: {chromadb.__version__}")
            
            # Try to get collection or create if it doesn't exist
            try:
                self.collection = self.client.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function,
                )
                count = self.collection.count()
                print(f"Collection exists with {count} documents")
            except Exception as e:
                print(f"Collection not found, creating new: {str(e)}")
                self.collection = self.client.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function,
                )
                print(f"Created new collection: {collection_name}")
                
        except Exception as e:
            print(f"Error initializing vector search: {str(e)}")
            traceback.print_exc()
            # Create a placeholder collection to prevent errors
            self.collection = None

    def chunk_document(self, text: str, min_chunk_size: int = 200, max_chunk_size: int = 1000, overlap: int = 50) -> List[str]:
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
            return self._chunk_by_headers(text, min_chunk_size, max_chunk_size, overlap)
        else:
            # Otherwise split on paragraph boundaries
            return self._chunk_by_paragraphs(text, min_chunk_size, max_chunk_size, overlap)
    
    def _chunk_by_headers(self, text: str, min_chunk_size: int, max_chunk_size: int, overlap: int) -> List[str]:
        """Split document on header boundaries for more semantic chunks."""
        # Find all headers (markdown style)
        header_pattern = r'^(#+)\s+(.+)$'
        header_matches = list(re.finditer(header_pattern, text, re.MULTILINE))
        
        chunks = []
        if not header_matches:
            # Fallback to paragraph chunking if no headers found
            return self._chunk_by_paragraphs(text, min_chunk_size, max_chunk_size, overlap)
        
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
                sub_chunks = self._chunk_by_paragraphs(content, min_chunk_size, max_chunk_size - len(header), overlap)
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
    
    def _chunk_by_paragraphs(self, text: str, min_chunk_size: int, max_chunk_size: int, overlap: int) -> List[str]:
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
    
    def _extract_chunk_title(self, chunk: str) -> str:
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

    def index_file(self, file_path: str, content: str, metadata: Optional[Dict] = None):
        """Index a file's content in the vector database using chunking strategy."""
        if self.collection is None:
            print("Vector database not initialized properly")
            return
            
        print(f"Indexing file: {file_path}")
        
        if not metadata:
            metadata = {}

        # Add basic metadata
        metadata.update(
            {
                "path": file_path,
                "filename": os.path.basename(file_path),
                "indexed_at": datetime.utcnow().isoformat(),
            }
        )

        # Add file's current tags
        file_obj = self.db_session.query(File).filter_by(path=file_path).first()
        if file_obj:
            # Convert tags list to a string that ChromaDB can handle
            tag_names = [tag.name for tag in file_obj.tags]
            # Store as a JSON string since ChromaDB doesn't accept lists
            metadata["tags"] = json.dumps(tag_names)
            print(f"File tags: {tag_names}")
        else:
            metadata["tags"] = "[]"  # Empty JSON array as string

        # Check if we already have this file indexed
        try:
            existing_docs = self.collection.get(
                ids=[file_path], include=["metadatas"]
            )
            
            if existing_docs and existing_docs["ids"] and len(existing_docs["ids"]) > 0:
                # Delete existing document and its chunks
                self.remove_file(file_path)
        except Exception as e:
            print(f"Error checking document existence: {str(e)}")

        # Chunk the document for better semantic search
        chunks = self.chunk_document(content)
        num_chunks = len(chunks)
        
        print(f"Document split into {num_chunks} chunks")
        
        # If document is small, just index as a single chunk
        if num_chunks <= 1:
            try:
                # Add as a single document
                self.collection.add(
                    ids=[file_path],
                    metadatas=[metadata],
                    documents=[content]
                )
                print("Indexed as single document")
            except Exception as e:
                print(f"Error indexing {file_path}: {str(e)}")
                traceback.print_exc()
            return
            
        # For chunked documents, add each chunk with chunk-specific metadata
        try:
            for i, chunk in enumerate(chunks):
                # Create chunk-specific metadata and ID
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_id": i,
                    "chunk_total": num_chunks,
                    "chunk_title": self._extract_chunk_title(chunk),
                    "is_chunk": True
                })
                
                # Create a compound ID to allow retrieving specific chunks
                chunk_id = f"{file_path}#chunk{i}"
                
                # Index this chunk
                self.collection.add(
                    ids=[chunk_id],
                    metadatas=[chunk_metadata],
                    documents=[chunk]
                )
            
            # Also add the full document as a single entry for simple retrieval
            # and to ensure we can find it by file path ID
            full_metadata = metadata.copy()
            full_metadata.update({
                "has_chunks": True,
                "num_chunks": num_chunks,
                "is_chunk": False
            })
            
            # Add a shortened version of the full content
            summary_length = min(1500, len(content))
            summary = content[:summary_length] + ("..." if len(content) > summary_length else "")
            
            self.collection.add(
                ids=[file_path],
                metadatas=[full_metadata],
                documents=[summary]  # Store a summarized version of the full content
            )
            
            print(f"Successfully indexed {num_chunks} chunks for {file_path}")
            
        except Exception as e:
            print(f"Error indexing chunks for {file_path}: {str(e)}")
            traceback.print_exc()

    def update_metadata(self, file_path: str):
        """Update metadata for an existing document."""
        if self.collection is None:
            print("Vector database not initialized properly")
            return False

        try:
            # Get file from database
            file_obj = self.db_session.query(File).filter_by(path=file_path).first()
            if not file_obj:
                print(f"File not found in database: {file_path}")
                return False

            # Create metadata
            metadata = {
                "path": file_path,
                "filename": os.path.basename(file_path),
                "indexed_at": datetime.utcnow().isoformat(),
                "tags": json.dumps([tag.name for tag in file_obj.tags]),  # Store as JSON string
            }

            # Check if document exists
            existing_docs = self.collection.get(
                ids=[file_path], include=["metadatas"]
            )

            if existing_docs and existing_docs['ids'] and len(existing_docs['ids']) > 0:
                # Update document metadata
                self.collection.update(
                    ids=[file_path], metadatas=[metadata]
                )
                print(f"Updated metadata for {file_path}")
                return True
            else:
                print(f"Document not found in collection: {file_path}")
                return False

        except Exception as e:
            print(f"Error updating metadata: {str(e)}")
            traceback.print_exc()
            return False

    def fix_all_metadata(self):
        """Fix metadata for all documents in the collection."""
        if self.collection is None:
            print("Vector database not initialized properly")
            return

        try:
            print("\nFixing metadata for all documents...")

            # Get all document IDs
            try:
                all_ids = self.collection.get(
                    include=["metadatas"], limit=1000
                )["ids"]

                if not all_ids:
                    print("No documents found in collection")
                    return
            except Exception as e:
                print(f"Error getting document IDs: {str(e)}")
                # Try using peek instead
                try:
                    peek_results = self.collection.peek(limit=1000)
                    all_ids = peek_results["ids"]
                    if not all_ids:
                        print("No documents found in collection")
                        return
                except Exception as e2:
                    print(f"Error peeking collection: {str(e2)}")
                    return

            fixed = 0
            for doc_id in all_ids:
                if self.update_metadata(doc_id):
                    fixed += 1

            print(f"Fixed metadata for {fixed}/{len(all_ids)} documents")

        except Exception as e:
            print(f"Error fixing metadata: {str(e)}")
            traceback.print_exc()

    def search(
        self,
        query: str,
        tag_filter: Optional[List[str]] = None,
        use_and: bool = True,
        limit: int = 20,
    ) -> List[Dict]:
        """
        Search for files using semantic search with optional tag filtering.
        Returns results with relevant text snippets showing why files matched.
        """
        if self.collection is None:
            print("Vector database not initialized properly")
            return []

        print(f"\nSearching with query: {query}")
        print(f"Tag filter: {tag_filter}, use_and: {use_and}")

        # Debug: Check collection health
        try:
            print("\nDebug: Collection health check")
            count = self.collection.count()
            print(f"Collection has {count} documents")

            if count == 0:
                print("Warning: Empty collection - no documents to search")
                return []

            # Test a sample document
            try:
                sample = self.collection.peek(limit=1)
                if sample and sample['ids'] and len(sample['ids']) > 0:
                    print(f"Sample document ID: {sample['ids'][0]}")
                    # Check the metadata format for tags
                    if sample['metadatas'] and len(sample['metadatas']) > 0:
                        print(f"Sample metadata: {sample['metadatas'][0]}")
                        if 'tags' in sample['metadatas'][0]:
                            print(f"Sample tags: {sample['metadatas'][0]['tags']}")
                else:
                    print("Warning: Collection peek returned no results")
            except Exception as e:
                print(f"Error peeking collection: {str(e)}")
        except Exception as e:
            print(f"Error checking collection health: {str(e)}")

        try:
            print(f"Executing query: '{query}' with limit={limit}")

            # Enhance query by using query expansion and improved parameters
            expanded_queries = self._expand_query(query)
            
            # Get potentially matching chunks - include document contents for snippet extraction
            try:
                # Use multiple queries for better recall if we have expansions
                if expanded_queries:
                    results = self.collection.query(
                        query_texts=expanded_queries,
                        n_results=100 if tag_filter else limit * 3,  # Get more results for filtering and chunked docs
                        include=["metadatas", "documents", "distances"]
                    )
                else:
                    results = self.collection.query(
                        query_texts=[query],
                        n_results=100 if tag_filter else limit * 3,  # Get more results for filtering and chunked docs
                        include=["metadatas", "documents", "distances"]
                    )
            except Exception as query_err:
                print(f"Error during query: {str(query_err)}")
                traceback.print_exc()
                return []

            # Debug: print raw results
            print(f"\nRaw query results:")
            print(
                f"  IDs: {len(results['ids'][0]) if results['ids'] and len(results['ids']) > 0 else 0} items"
            )
            print(f"  Distances: {results['distances'][0][:3] if results['distances'] and len(results['distances']) > 0 else []}")

            if not results or not results['metadatas'] or len(results['metadatas']) == 0 or len(results['metadatas'][0]) == 0:
                print("No results found")
                return []

            initial_count = len(results['metadatas'][0])
            print(f"Initial results found: {initial_count}")
            
            # Group results by file path to combine chunks from the same document
            grouped_results = {}
            
            # Process and filter results
            for i, (distance, metadata, document) in enumerate(zip(results['distances'][0], results['metadatas'][0], results['documents'][0])):
                # Extract file path (remove chunk identifier if present)
                doc_id = results['ids'][0][i]
                file_path = doc_id.split('#')[0]  # Remove chunk identifier
                
                # Parse tags from JSON string
                tags = []
                if "tags" in metadata:
                    try:
                        tags = json.loads(metadata["tags"])
                        tags_lower = [t.lower() for t in tags]
                    except json.JSONDecodeError:
                        print(f"  Failed to parse tags: {metadata.get('tags')}")
                        tags = []
                        tags_lower = []
                else:
                    tags = []
                    tags_lower = []

                # Apply tag filtering if specified
                if tag_filter:
                    # Make tag filter case-insensitive
                    tag_filter_lower = [tag.lower() for tag in tag_filter]
                    
                    # Check if this document matches the tag filter
                    if use_and:
                        # AND logic: all filter tags must be in document tags
                        matched = all(filter_tag in tags_lower for filter_tag in tag_filter_lower)
                        if not matched:
                            continue  # Skip this document
                    else:
                        # OR logic: at least one filter tag must be in document tags
                        matched = any(filter_tag in tags_lower for filter_tag in tag_filter_lower)
                        if not matched:
                            continue  # Skip this document

                # Calculate similarity score
                if distance is not None:
                    # Base similarity calculation
                    raw_similarity = 1.0 - (float(distance) / 2.0)
                    
                    # Apply scaling to increase contrast in scores
                    boost_power = 0.65
                    similarity = raw_similarity ** boost_power
                    
                    # Scale the final result
                    similarity = 0.2 + (similarity * 0.8)
                    
                    # Ensure it's between 0 and 1
                    similarity = max(0.0, min(1.0, similarity))
                else:
                    similarity = 0.0

                # Extract snippet from the document
                snippets = self._extract_relevant_snippets(document, query)
                if not snippets:
                    # If no specific snippets found, get the beginning of the document
                    preview_length = min(150, len(document))
                    snippets = [f"{document[:preview_length]}..."]

                # Get chunk information
                is_chunk = metadata.get('is_chunk', False)
                chunk_id = metadata.get('chunk_id', 0) if is_chunk else 0
                chunk_total = metadata.get('chunk_total', 1) if is_chunk else 1
                chunk_title = metadata.get('chunk_title', '') if is_chunk else ''

                # If this file is not in the grouped results yet, add it
                if file_path not in grouped_results:
                    grouped_results[file_path] = {
                        'path': file_path,
                        'filename': metadata.get('filename', os.path.basename(file_path)),
                        'tags': tags,
                        'score': similarity,  # Will be updated as we find better chunks
                        'snippets': [],
                        'chunks_found': 0,
                        'chunk_titles': {},
                    }
                
                # Update existing result with this chunk's info
                current_result = grouped_results[file_path]
                
                # Update score (take max of all chunks)
                current_result['score'] = max(current_result['score'], similarity)
                
                # Increment chunks found
                current_result['chunks_found'] += 1
                
                # Add snippet if it's good quality (has a high score)
                if similarity > 0.4:  # Only add high quality snippets
                    # Add context from chunk title if available
                    context = f"[{chunk_title}] " if chunk_title else ""
                    
                    # Only add a limited number of snippets per file
                    if len(current_result['snippets']) < 3:
                        for snippet in snippets[:1]:  # Limit to 1 snippet per chunk
                            decorated_snippet = f"{context}{snippet}"
                            current_result['snippets'].append(decorated_snippet)
                    
                # Keep track of chunk titles for showing document structure
                if chunk_title:
                    current_result['chunk_titles'][chunk_id] = chunk_title

            # Create final result list from the grouped results
            filtered_results = list(grouped_results.values())
            
            # If we have no results but had initial results with tag filter, something went wrong with filtering
            filtered_count = len(filtered_results)
            print(f"Final results after filtering: {filtered_count}")
            if filtered_count == 0 and initial_count > 0 and tag_filter:
                print("WARNING: All results filtered out! Check if tag names match exactly.")
                
                # If we have a large mismatch, try a more relaxed approach by using the raw results
                filtered_results = []
                for i, (distance, metadata, document) in enumerate(zip(results['distances'][0], results['metadatas'][0], results['documents'][0])):
                    # Extract file path
                    doc_id = results['ids'][0][i]
                    file_path = doc_id.split('#')[0]  # Remove chunk identifier
                    
                    # Basic result with minimal filtering
                    result = {
                        'path': file_path,
                        'filename': metadata.get('filename', os.path.basename(file_path)),
                        'score': 1.0 - (float(distance) / 2.0) if distance is not None else 0.0,
                        'snippets': self._extract_relevant_snippets(document, query) or [document[:150] + "..."]
                    }
                    
                    # Parse tags if available
                    if "tags" in metadata:
                        try:
                            result['tags'] = json.loads(metadata["tags"])
                        except:
                            result['tags'] = []
                    else:
                        result['tags'] = []
                        
                    filtered_results.append(result)
            
            # Sort by score and limit results
            filtered_results.sort(key=lambda x: x['score'], reverse=True)
            
            # Remove duplicates (keeping the first/highest scored occurrence)
            seen_paths = set()
            unique_results = []
            
            for result in filtered_results:
                if result['path'] not in seen_paths:
                    seen_paths.add(result['path'])
                    unique_results.append(result)
            
            return unique_results[:limit]

        except Exception as e:
            print(f"Search error: {str(e)}")
            traceback.print_exc()
            return []
    
    def _expand_query(self, query: str) -> List[str]:
        """
        Expand the query with synonyms or related terms to improve recall.
        Returns list of expanded queries to use.
        
        Example: "document text" -> ["document text", "document content", "file text"]
        
        The original query is always included as the first item.
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
            
    def _extract_relevant_snippets(self, text: str, query: str, max_snippets: int = 3, snippet_length: int = 150) -> List[str]:
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
                    refined_start = self._find_sentence_boundary(text, current_snippet_start, False)
                    refined_end = self._find_sentence_boundary(text, current_snippet_end, True)
                    
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
            refined_start = self._find_sentence_boundary(text, current_snippet_start, False)
            refined_end = self._find_sentence_boundary(text, current_snippet_end, True)
            
            snippet = text[refined_start:refined_end]
            # Add ellipsis if needed
            prefix = "..." if refined_start > 0 else ""
            suffix = "..." if refined_end < len(text) else ""
            snippet = f"{prefix}{snippet}{suffix}"
            
            snippets.append(snippet)
            
        return snippets
        
    def _find_sentence_boundary(self, text: str, pos: int, find_end: bool) -> int:
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

    def reindex_all_files(self, progress_callback=None):
        """Reindex all files in the database."""
        if self.collection is None:
            if progress_callback:
                progress_callback("Vector database not initialized properly", 0)
            return

        try:
            # Try to delete the collection and recreate it
            try:
                self.client.delete_collection(self.collection_name)
                print(f"Deleted collection: {self.collection_name}")
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function,
                )
                print(f"Recreated collection: {self.collection_name}")
            except Exception as e:
                print(f"Error recreating collection: {str(e)}")
                # Try clearing instead
                try:
                    self.collection.delete(
                        where={},
                    )
                    print("Cleared all documents from collection")
                except Exception as e2:
                    print(f"Error clearing collection: {str(e2)}")

            # Get all files from database
            files = self.db_session.query(File).all()
            total_files = len(files)

            if progress_callback:
                progress_callback(f"Reindexing {total_files} files", 5)

            indexed = 0
            for i, file_obj in enumerate(files):
                if progress_callback:
                    progress = 5 + int((i / total_files) * 90)  # 5-95% for indexing
                    progress_callback(f"Indexing {i+1}/{total_files}: {file_obj.path}", progress)

                try:
                    if os.path.exists(file_obj.path):
                        # Get file content
                        content = self._extract_file_content(file_obj.path)
                        if content:
                            self.index_file(file_obj.path, content)
                            indexed += 1
                except Exception as e:
                    print(f"Error indexing {file_obj.path}: {str(e)}")

            if progress_callback:
                progress_callback(f"Indexed {indexed}/{total_files} files successfully", 95)

            # Verify the indexing
            try:
                count = self.collection.count()
                if progress_callback:
                    progress_callback(f"Verification: {count} documents in collection", 98)
            except Exception as e:
                print(f"Error verifying collection count: {str(e)}")

            if progress_callback:
                progress_callback("Indexing complete", 100)

        except Exception as e:
            print(f"Error during reindexing: {str(e)}")
            traceback.print_exc()
            if progress_callback:
                progress_callback(f"Error: {str(e)}", 0)

    def _extract_file_content(self, file_path: str) -> Optional[str]:
        """Extract searchable content from a file."""
        import mimetypes
        from pypdf import PdfReader

        mime_type, _ = mimetypes.guess_type(file_path)
        content = ""

        try:
            if mime_type and mime_type.startswith("text/"):
                # Handle text files including markdown
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            elif mime_type == "application/pdf":
                try:
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
            return f"Error extracting content: {os.path.basename(file_path)}"

    def debug_check_file(self, file_path: str):
        """Debug helper to check if a specific file is properly indexed."""
        if self.collection is None:
            print("Vector database not initialized properly")
            return

        print(f"\nChecking specific file: {file_path}")

        # Check if file exists in database
        file_obj = self.db_session.query(File).filter_by(path=file_path).first()
        if file_obj:
            print("File found in database")
            print(f"Tags: {[tag.name for tag in file_obj.tags]}")
        else:
            print("File not found in database")
            return

        # Check if file exists in vector collection
        try:
            results = self.collection.get(
                ids=[file_path], include=['metadatas', 'documents']
            )
            if results and results['ids'] and len(results['ids']) > 0:
                print("\nFile found in vector collection:")
                print(f"Metadata: {results['metadatas'][0]}")
                print(f"Document length: {len(results['documents'][0]) if results['documents'] and len(results['documents']) > 0 else 0}")

                # Test search for this document
                print("\nTesting search for this document...")
                test_results = self.collection.query(
                    query_texts=[os.path.basename(file_path)],  # Use filename as query
                    n_results=5,
                )

                if test_results and test_results['ids'] and len(test_results['ids']) > 0:
                    print(f"Search results:")
                    for i, (doc_id, distance) in enumerate(zip(test_results['ids'][0], test_results['distances'][0])):
                        similarity = 1.0 - (float(distance) / 2.0)
                        print(f"  {i + 1}. {os.path.basename(doc_id)} (Score: {similarity:.4f})")
                        if doc_id == file_path:
                            print("    ✓ Target file found in results!")
                else:
                    print("No search results")
            else:
                print("\nFile not found in vector collection")
        except Exception as e:
            print(f"\nError checking vector collection: {str(e)}")
            traceback.print_exc()

    def remove_file(self, file_path: str) -> bool:
        """
        Remove a specific file from the vector database.
        
        Args:
            file_path: The path of the file to remove
            
        Returns:
            bool: True if removed successfully, False otherwise
        """
        if self.collection is None:
            print("Vector database not initialized properly")
            return False
            
        try:
            # First, check for any chunks associated with this file
            try:
                # Use a "where" filter to find all chunks (prefix match would be better but not supported)
                all_ids = self.collection.get(include=["ids"])["ids"]
                chunk_ids = [doc_id for doc_id in all_ids if doc_id.startswith(file_path + "#")]
                
                # Delete all chunks for this file
                if chunk_ids:
                    print(f"Removing {len(chunk_ids)} chunks for {file_path}")
                    self.collection.delete(ids=chunk_ids)
            except Exception as e:
                print(f"Error removing chunks: {str(e)}")
                
            # Now remove the main document entry
            existing_docs = self.collection.get(
                ids=[file_path], include=["metadatas"]
            )
            
            if existing_docs and existing_docs['ids'] and len(existing_docs['ids']) > 0:
                # Remove the document
                self.collection.delete(ids=[file_path])
                print(f"Successfully removed {file_path} from vector database")
                return True
            else:
                print(f"File {file_path} not found in vector database")
                # If we at least removed some chunks, consider it a partial success
                return bool(chunk_ids)
            
        except Exception as e:
            print(f"Error removing {file_path} from vector database: {str(e)}")
            traceback.print_exc()
            return False