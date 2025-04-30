"""
Main module for vector search functionality.
Provides vector database management for semantic search of file contents.
"""

import importlib
import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import json
import traceback

from .document_chunker import DocumentChunker
from .content_extractor import ContentExtractor
from .search_utils import SearchUtils
from ai_service import AIService
from config import Config


class VectorSearch:
    def __init__(
        self, db_session, config: Config, collection_name: str = "file_contents"
    ):
        """
        Initialize vector search with database session and collection name.

        Args:
            db_session: SQLAlchemy database session
            config: Configuration object
            collection_name: Name for the ChromaDB collection
        """
        self.db_session = db_session
        self.config = config  # Store the config object
        self.collection_name = collection_name
        print("\nInitializing vector search...")

        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(path=".chroma")

            # Use a more advanced embedding model with higher dimensionality for better semantic matching
            # Options: 'all-mpnet-base-v2' (768d), 'multi-qa-mpnet-base-dot-v1' (768d), or 'all-MiniLM-L12-v2' (384d)
            self.embedding_function = SentenceTransformerEmbeddingFunction(
                model_name="intfloat/e5-base-v2"
            )

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

    def index_file(self, file_path: str, content: str, metadata: Optional[Dict] = None):
        """
        Index a file's content in the vector database using chunking strategy.

        Args:
            file_path: Path to the file
            content: Text content of the file
            metadata: Optional additional metadata
        """
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

        # Generate a summary of the document using AI
        summary = self.generate_document_summary(file_path, content)
        if summary:
            metadata["summary"] = summary

        # Add file's current tags
        from models import File

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
            existing_docs = self.collection.get(ids=[file_path], include=["metadatas"])

            if existing_docs and existing_docs["ids"] and len(existing_docs["ids"]) > 0:
                # Delete existing document and its chunks
                self.remove_file(file_path)
        except Exception as e:
            print(f"Error checking document existence: {str(e)}")

        # Chunk the document for better semantic search
        chunks = DocumentChunker.chunk_document(content)
        num_chunks = len(chunks)

        print(f"Document split into {num_chunks} chunks")

        # If document is small, just index as a single chunk
        if num_chunks <= 1:
            try:
                # Add as a single document
                self.collection.add(
                    ids=[file_path], metadatas=[metadata], documents=[content]
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
                chunk_metadata.update(
                    {
                        "chunk_id": i,
                        "chunk_total": num_chunks,
                        "chunk_title": DocumentChunker.extract_chunk_title(chunk),
                        "is_chunk": True,
                    }
                )

                # Create a compound ID to allow retrieving specific chunks
                chunk_id = f"{file_path}#chunk{i}"

                # Index this chunk
                self.collection.add(
                    ids=[chunk_id], metadatas=[chunk_metadata], documents=[chunk]
                )

            # Also add the full document as a single entry for simple retrieval
            # and to ensure we can find it by file path ID
            full_metadata = metadata.copy()
            full_metadata.update(
                {"has_chunks": True, "num_chunks": num_chunks, "is_chunk": False}
            )

            # Add a shortened version of the full content
            summary_length = min(1500, len(content))
            summary_content = content[:summary_length] + (
                "..." if len(content) > summary_length else ""
            )

            self.collection.add(
                ids=[file_path],
                metadatas=[full_metadata],
                documents=[
                    summary_content
                ],  # Store a summarized version of the full content
            )

            print(f"Successfully indexed {num_chunks} chunks for {file_path}")

        except Exception as e:
            print(f"Error indexing chunks for {file_path}: {str(e)}")
            traceback.print_exc()

    def update_metadata(self, file_path: str):
        """
        Update metadata for an existing document.

        Args:
            file_path: Path to the file

        Returns:
            bool: True if metadata was updated, False otherwise
        """
        if self.collection is None:
            print("Vector database not initialized properly")
            return False

        try:
            # Get file from database
            from models import File

            file_obj = self.db_session.query(File).filter_by(path=file_path).first()
            if not file_obj:
                print(f"File not found in database: {file_path}")
                return False

            # Create metadata
            metadata = {
                "path": file_path,
                "filename": os.path.basename(file_path),
                "indexed_at": datetime.utcnow().isoformat(),
                "tags": json.dumps(
                    [tag.name for tag in file_obj.tags]
                ),  # Store as JSON string
            }

            # Check if document exists and get existing metadata
            existing_docs = self.collection.get(ids=[file_path], include=["metadatas"])

            # Preserve existing summary if available
            existing_summary = None
            if (
                existing_docs
                and existing_docs["ids"]
                and len(existing_docs["ids"]) > 0
                and existing_docs["metadatas"]
                and len(existing_docs["metadatas"]) > 0
            ):

                existing_metadata = existing_docs["metadatas"][0]
                if "summary" in existing_metadata and existing_metadata["summary"]:
                    print(f"Preserving existing summary for {file_path}")
                    existing_summary = existing_metadata["summary"]
                    metadata["summary"] = existing_summary

            if existing_docs and existing_docs["ids"] and len(existing_docs["ids"]) > 0:
                # If we don't have an existing summary, try to generate one                if not existing_summary and os.path.exists(file_path):
                try:
                    from .content_extractor import ContentExtractor

                    # Use the configured PDF extractor preference
                    pdf_extractor = self.get_pdf_extractor_preference()
                    content = ContentExtractor.extract_file_content(
                        file_path, pdf_extractor=pdf_extractor
                    )
                    if content:
                        summary = self.generate_document_summary(file_path, content)
                        if summary:
                            metadata["summary"] = summary
                except Exception as e:
                    print(f"Error generating summary during metadata update: {str(e)}")

                # Update document metadata
                self.collection.update(ids=[file_path], metadatas=[metadata])
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
                all_ids = self.collection.get(include=["metadatas"], limit=1000)["ids"]

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

        Args:
            query: Search query text
            tag_filter: Optional list of tags to filter by
            use_and: If True, use AND logic for tag filtering; if False, use OR logic
            limit: Maximum number of results to return

        Returns:
            List of dictionaries with search results
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
                if sample and sample["ids"] and len(sample["ids"]) > 0:
                    print(f"Sample document ID: {sample['ids'][0]}")
                    # Check the metadata format for tags
                    if sample["metadatas"] and len(sample["metadatas"]) > 0:
                        print(f"Sample metadata: {sample['metadatas'][0]}")
                        if "tags" in sample["metadatas"][0]:
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
            expanded_queries = SearchUtils.expand_query(query)

            # Get potentially matching chunks - include document contents for search
            try:
                # Use multiple queries for better recall if we have expansions
                if expanded_queries:
                    results = self.collection.query(
                        query_texts=expanded_queries,
                        n_results=(
                            100 if tag_filter else limit * 3
                        ),  # Get more results for filtering and chunked docs
                        include=["metadatas", "distances"],
                    )
                else:
                    results = self.collection.query(
                        query_texts=[query],
                        n_results=(
                            100 if tag_filter else limit * 3
                        ),  # Get more results for filtering and chunked docs
                        include=["metadatas", "distances"],
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
            print(
                f"  Distances: {results['distances'][0][:3] if results['distances'] and len(results['distances']) > 0 else []}"
            )

            if (
                not results
                or not results["metadatas"]
                or len(results["metadatas"]) == 0
                or len(results["metadatas"][0]) == 0
            ):
                print("No results found")
                return []

            initial_count = len(results["metadatas"][0])
            print(f"Initial results found: {initial_count}")

            # Group results by file path to combine chunks from the same document
            grouped_results = {}

            # Process and filter results
            for i, (distance, metadata) in enumerate(
                zip(results["distances"][0], results["metadatas"][0])
            ):
                # Extract file path (remove chunk identifier if present)
                doc_id = results["ids"][0][i]
                file_path = doc_id.split("#")[0]  # Remove chunk identifier

                # Parse tags from JSON string
                tags = []
                if "tags" in metadata:
                    try:
                        tags = json.loads(metadata["tags"])
                        # Ensure tags is a list, even if metadata format is unexpected
                        if not isinstance(tags, list):
                            tags = [str(tags)]
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
                        matched = all(
                            filter_tag in tags_lower for filter_tag in tag_filter_lower
                        )
                        if not matched:
                            continue  # Skip this document
                    else:
                        # OR logic: at least one filter tag must be in document tags
                        matched = any(
                            filter_tag in tags_lower for filter_tag in tag_filter_lower
                        )
                        if not matched:
                            continue  # Skip this document

                # Calculate similarity score
                if distance is not None:
                    # Base similarity calculation
                    raw_similarity = 1.0 - (float(distance) / 2.0)

                    # Apply scaling to increase contrast in scores
                    boost_power = 0.65
                    similarity = raw_similarity**boost_power

                    # Scale the final result
                    similarity = 0.2 + (similarity * 0.8)

                    # Ensure it's between 0 and 1
                    similarity = max(0.0, min(1.0, similarity))
                else:
                    similarity = 0.0

                # Get chunk information
                is_chunk = metadata.get("is_chunk", False)
                chunk_id = metadata.get("chunk_id", 0) if is_chunk else 0
                chunk_total = metadata.get("chunk_total", 1) if is_chunk else 1
                chunk_title = metadata.get("chunk_title", "") if is_chunk else ""
                file_type = (
                    os.path.splitext(file_path)[1][1:].lower()
                    if os.path.splitext(file_path)[1]
                    else ""
                )

                # Get document summary if available
                document_summary = metadata.get("summary", "")

                # If this file is not in the grouped results yet, add it
                if file_path not in grouped_results:
                    grouped_results[file_path] = {
                        "path": file_path,
                        "filename": metadata.get(
                            "filename", os.path.basename(file_path)
                        ),
                        "file_type": file_type,
                        "tags": tags,
                        "score": similarity,  # Will be updated as we find better chunks
                        "chunks_found": 0,
                        "best_chunk_similarity": 0.0,
                        "summary": document_summary,  # Add summary to results
                    }

                # Update existing result with this chunk's info
                current_result = grouped_results[file_path]

                # Update score (take max of all chunks)
                current_result["score"] = max(current_result["score"], similarity)

                # Track best chunk similarity for sorting results
                current_result["best_chunk_similarity"] = max(
                    current_result["best_chunk_similarity"], similarity
                )

                # Increment chunks found
                current_result["chunks_found"] += 1

                # Keep the summary from highest-level document (not chunk)
                if not is_chunk and document_summary and not current_result["summary"]:
                    current_result["summary"] = document_summary

            # Post-process the grouped results
            final_results = []
            for file_path, result in grouped_results.items():
                # Calculate a document relevance summary
                chunks_text = (
                    f"({result['chunks_found']} matching sections)"
                    if result["chunks_found"] > 1
                    else ""
                )
                result["relevance"] = f"Relevance: {result['score']:.1%} {chunks_text}"

                # Add document type context
                result["document_type"] = SearchUtils.get_document_type_label(file_path)

                final_results.append(result)

            # Sort by score and limit results
            final_results.sort(key=lambda x: x["score"], reverse=True)

            # Remove duplicates (keeping the first/highest scored occurrence)
            seen_paths = set()
            unique_results = []

            for result in final_results:
                if result["path"] not in seen_paths:
                    seen_paths.add(result["path"])
                    unique_results.append(result)

            return unique_results[:limit]

        except Exception as e:
            print(f"Search error: {str(e)}")
            traceback.print_exc()
            return []

    def reindex_all(self, progress_callback=None):
        """
        Reindex all files in the database.

        Args:
            progress_callback: Optional callback function for progress updates
        """
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
            from models import File

            files = self.db_session.query(File).all()
            total_files = len(files)

            if progress_callback:
                progress_callback(f"Reindexing {total_files} files", 5)

            indexed = 0
            for i, file_obj in enumerate(files):
                if progress_callback:
                    progress = 5 + int((i / total_files) * 90)  # 5-95% for indexing
                    progress_callback(
                        f"Indexing {i+1}/{total_files}: {file_obj.path}", progress
                    )
                    try:
                        if os.path.exists(file_obj.path):
                            # Get the PDF extractor preference
                            pdf_extractor = self.get_pdf_extractor_preference()

                            print(
                                f"Reindexing {file_obj.path} using {pdf_extractor} extraction mode"
                            )
                            # Get file content
                            content = ContentExtractor.extract_file_content(
                                file_obj.path, pdf_extractor=pdf_extractor
                            )
                            if content:
                                self.index_file(file_obj.path, content)
                                indexed += 1
                    except Exception as e:
                        print(f"Error indexing {file_obj.path}: {str(e)}")

            if progress_callback:
                progress_callback(
                    f"Indexed {indexed}/{total_files} files successfully", 95
                )

            # Verify the indexing
            try:
                count = self.collection.count()
                if progress_callback:
                    progress_callback(
                        f"Verification: {count} documents in collection", 98
                    )
            except Exception as e:
                print(f"Error verifying collection count: {str(e)}")

            if progress_callback:
                progress_callback("Indexing complete", 100)

        except Exception as e:
            print(f"Error during reindexing: {str(e)}")
            traceback.print_exc()
            if progress_callback:
                progress_callback(f"Error: {str(e)}", 0)

    def debug_check_file(self, file_path: str):
        """
        Debug helper to check if a specific file is properly indexed.

        Args:
            file_path: Path to the file to check
        """
        if self.collection is None:
            print("Vector database not initialized properly")
            return

        print(f"\nChecking specific file: {file_path}")

        # Check if file exists in database
        from models import File

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
                ids=[file_path], include=["metadatas", "documents"]
            )
            if results and results["ids"] and len(results["ids"]) > 0:
                print("\nFile found in vector collection:")
                print(f"Metadata: {results['metadatas'][0]}")
                print(
                    f"Document length: {len(results['documents'][0]) if results['documents'] and len(results['documents']) > 0 else 0}"
                )

                # Test search for this document
                print("\nTesting search for this document...")
                test_results = self.collection.query(
                    query_texts=[os.path.basename(file_path)],  # Use filename as query
                    n_results=5,
                )

                if (
                    test_results
                    and test_results["ids"]
                    and len(test_results["ids"]) > 0
                ):
                    print(f"Search results:")
                    for i, (doc_id, distance) in enumerate(
                        zip(test_results["ids"][0], test_results["distances"][0])
                    ):
                        similarity = 1.0 - (float(distance) / 2.0)
                        print(
                            f"  {i + 1}. {os.path.basename(doc_id)} (Score: {similarity:.4f})"
                        )
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
                chunk_ids = [
                    doc_id for doc_id in all_ids if doc_id.startswith(file_path + "#")
                ]

                # Delete all chunks for this file
                if chunk_ids:
                    print(f"Removing {len(chunk_ids)} chunks for {file_path}")
                    self.collection.delete(ids=chunk_ids)
            except Exception as e:
                print(f"Error removing chunks: {str(e)}")

            # Now remove the main document entry
            existing_docs = self.collection.get(ids=[file_path], include=["metadatas"])

            if existing_docs and existing_docs["ids"] and len(existing_docs["ids"]) > 0:
                # Remove the document
                self.collection.delete(ids=[file_path])
                print(f"Successfully removed {file_path} from vector database")
                return True
            else:
                print(
                    f"File {file_path} not found in vector database"
                )  # If we at least removed some chunks, consider it a partial success
                return bool(chunk_ids)
        except Exception as e:
            print(f"Error removing {file_path} from vector database: {str(e)}")
            traceback.print_exc()
            return False

    def generate_document_summary(self, file_path: str, content: str) -> Optional[str]:
        """
        Generate a brief summary of the document content using AI.

        Args:
            file_path: Path to the file
            content: Text content of the file

        Returns:
            str: Generated summary or None if generation failed
        """
        try:
            # Create an AI service using the config
            ai_service = self.config.get_ai_service(self.db_session)

            if not ai_service:
                print("Could not create AI service, skipping summary generation")
                return None

            # Use the AI service to generate the summary
            return ai_service.generate_document_summary(file_path, content)

        except Exception as e:
            print(f"Error generating summary for {file_path}: {str(e)}")
            traceback.print_exc()
            return self._extract_basic_summary(content)

        except Exception as e:
            print(f"Error generating summary for {file_path}: {str(e)}")
            traceback.print_exc()
            # Fallback to basic extraction
            print("Falling back to basic extraction due to error")
            return self._extract_basic_summary(content)

    def _extract_basic_summary(self, content: str) -> str:
        """Extract a basic summary from content when AI summarization fails."""
        # Take first paragraph that's not empty and has reasonable length
        paragraphs = content.split("\n\n")
        for p in paragraphs:
            clean_p = p.strip()
            if len(clean_p) > 30 and len(clean_p) < 200:
                return clean_p

        # If no suitable paragraph found, just take first 150 chars
        return content[:150] + "..." if len(content) > 150 else content

    def get_pdf_extractor_preference(self) -> str:
        """
        Get the PDF extractor preference from the configuration.
        Returns 'fast' or 'accurate' based on the current settings.

        Returns:
            str: The PDF extractor preference ('fast' or 'accurate')
        """
        # Default to accurate extraction
        pdf_extractor = "accurate"
        # Check if config module can be imported
        config_module = None
        if importlib.util.find_spec("config") is not None:
            try:
                import config

                config_module = config
            except ImportError:
                pass

        # Try to get preference from db_session if it has config attached
        if hasattr(self.db_session, "config") and self.db_session.config:
            pdf_extractor = self.db_session.config.get_pdf_extractor()
        # Otherwise try to load config directly
        elif config_module and hasattr(config_module, "Config"):
            try:
                from password_management import get_password

                try:
                    password = get_password()
                    if password:
                        cfg = config_module.Config(password)
                        pdf_extractor = cfg.get_pdf_extractor()
                except:
                    # This might be first run where password isn't set yet
                    pass
            except Exception as e:
                print(f"Could not load PDF extractor preference from config: {e}")

        return pdf_extractor
