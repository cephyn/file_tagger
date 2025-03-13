import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from models import File, Tag
import json
import traceback

class VectorSearch:
    def __init__(self, db_session, collection_name: str = "file_contents"):
        self.db_session = db_session
        self.collection_name = collection_name
        print("\nInitializing vector search...")
        
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(path=".chroma")
            self.embedding_function = SentenceTransformerEmbeddingFunction()
            
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
        """Index a file's content in the vector database."""
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

        # Use file path as ID to avoid duplicates
        try:
            print(f"Adding to collection with content length: {len(content)}")

            # Use get method to check if document exists
            try:
                existing_docs = self.collection.get(
                    ids=[file_path], include=["metadatas"]
                )

                if existing_docs and existing_docs["ids"] and len(existing_docs["ids"]) > 0:
                    # Update existing document
                    self.collection.update(
                        ids=[file_path], metadatas=[metadata], documents=[content]
                    )
                    print("Successfully updated existing document")
                else:
                    # Add new document
                    self.collection.add(
                        ids=[file_path], metadatas=[metadata], documents=[content]
                    )
                    print("Successfully added new document")
            except Exception as e:
                print(f"Error checking document existence: {str(e)}")
                # Try adding as new
                self.collection.add(
                    ids=[file_path], metadatas=[metadata], documents=[content]
                )
                print("Added as new document after error")

        except Exception as e:
            print(f"Error indexing {file_path}: {str(e)}")
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

        # Since ChromaDB doesn't support complex string matching within metadata,
        # we'll retrieve all documents first, then filter them manually
        if tag_filter:
            print(f"Using tag filter: {tag_filter} (Will be applied post-query)")
            # Print tag filter in uppercase and lowercase for debugging
            print(f"Tag filter (lowercase): {[tag.lower() for tag in tag_filter]}")

        try:
            print(f"Executing query: '{query}' with limit={limit}")

            # Get all potentially matching documents
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=100 if tag_filter else limit,  # Get more results if we need to filter
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
            
            # Format results and apply tag filtering manually
            filtered_results = []

            # Check if the results structure is as expected
            if (not results['distances'] or len(results['distances']) == 0 or not results['metadatas'] or len(results['metadatas']) == 0 or len(results['distances'][0]) != len(results['metadatas'][0])):
                print("Warning: Unexpected results structure")
                return []

            for i, (distance, metadata) in enumerate(zip(results['distances'][0], results['metadatas'][0])):
                result = metadata.copy()
                
                # Parse tags from JSON string
                tags = []
                if "tags" in result:
                    try:
                        # Debug: print raw tag string for this result
                        print(f"\nResult {i}: Tag string: {result['tags']}")
                        
                        tags = json.loads(result["tags"])
                        # Normalize tags to lowercase for case-insensitive comparison
                        tags_lower = [t.lower() for t in tags]
                        result["tags"] = tags
                        
                        # Debug: print parsed tags
                        print(f"  Parsed tags: {tags}")
                    except json.JSONDecodeError:
                        print(f"  Failed to parse tags: {result.get('tags')}")
                        result["tags"] = []
                        tags_lower = []
                else:
                    print(f"  No tags found in result {i}")
                    result["tags"] = []
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
                            # Debug: Show which tags are missing
                            missing_tags = [tag for tag in tag_filter if tag.lower() not in tags_lower]
                            print(f"  Skipping result {i} (missing tags: {missing_tags})")
                            continue  # Skip this document
                    else:
                        # OR logic: at least one filter tag must be in document tags
                        matched = any(filter_tag in tags_lower for filter_tag in tag_filter_lower)
                        if not matched:
                            print(f"  Skipping result {i} (no matching tags)")
                            continue  # Skip this document
                    
                    print(f"  Result {i} matched filter: {matched}")

                # ChromaDB returns cosine distance (0-2 where 0 is identical)
                # Convert to similarity score (0-1 where 1 is identical)
                # For cosine distance: similarity = 1 - (distance / 2)
                if distance is not None:
                    similarity = 1.0 - (float(distance) / 2.0)
                    # Ensure it's between 0 and 1
                    similarity = max(0.0, min(1.0, similarity))
                    result['score'] = similarity

                    # Add 0.01 to avoid showing 0.0 scores (for UX purposes) if non-zero
                    if 0 < similarity < 0.01:
                        result['score'] = 0.01
                else:
                    result['score'] = 0.0

                # Debug individual result
                print(f"  Path: {result.get('path', 'N/A')}")
                print(f"  Score: {result['score']:.4f}")
                
                filtered_results.append(result)
            
            filtered_count = len(filtered_results)
            print(f"Final results after filtering: {filtered_count}")
            if filtered_count == 0 and initial_count > 0 and tag_filter:
                print("WARNING: All results filtered out! Check if tag names match exactly.")
                
                # Try a relaxed version with partial tag matching as a fallback
                print("\nTrying relaxed tag matching as fallback...")
                
                filtered_results = []
                for i, (distance, metadata) in enumerate(zip(results['distances'][0], results['metadatas'][0])):
                    result = metadata.copy()
                    
                    # Parse tags from JSON string
                    tags = []
                    if "tags" in result:
                        try:
                            tags = json.loads(result["tags"])
                            # Store normalized tags for matching
                            result["tags"] = tags
                        except json.JSONDecodeError:
                            result["tags"] = []
                    else:
                        result["tags"] = []
                        
                    # Calculate similarity score
                    if distance is not None:
                        similarity = 1.0 - (float(distance) / 2.0)
                        result['score'] = max(0.0, min(1.0, similarity))
                        if 0 < similarity < 0.01:
                            result['score'] = 0.01
                    else:
                        result['score'] = 0.0
                        
                    # No tag filtering in the fallback mode
                    filtered_results.append(result)
                
                print(f"Fallback returning all {len(filtered_results)} results without tag filtering")

            # Sort by score and limit results
            filtered_results.sort(key=lambda x: x['score'], reverse=True)
            return filtered_results[:limit]

        except Exception as e:
            print(f"Search error: {str(e)}")
            traceback.print_exc()
            return []
        
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