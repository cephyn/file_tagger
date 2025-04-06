"""
Debugging script to test vector search indexing functionality.
This will try to index a sample file and verify that it's properly added to the vector store.
"""

import os
import sys
from models import init_db, File, Tag
from vector_search import VectorSearch
from vector_search.content_extractor import ContentExtractor

def test_indexing():
    print("\n=== Vector Search Debugging ===\n")
    
    # Step 1: Initialize the database session
    print("Step 1: Initializing database session...")
    db_session = init_db()
    
    # Step 2: Initialize vector search
    print("\nStep 2: Initializing vector search...")
    try:
        vector_search = VectorSearch(db_session)
        
        # Verify if collection is initialized
        if vector_search.collection is None:
            print("ERROR: Vector search collection is None!")
            print("Check for errors during vector search initialization.")
            return False
        else:
            count = vector_search.collection.count()
            print(f"Vector search is working. Collection contains {count} documents.")
    except Exception as e:
        print(f"ERROR initializing vector search: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Test indexing with this script file
    test_file = os.path.abspath(__file__)
    print(f"\nStep 3: Testing indexing with file: {test_file}")
    
    try:
        # First, extract content
        print("Extracting content from test file...")
        content = ContentExtractor.extract_file_content(test_file)
        if content:
            print(f"Content extracted successfully ({len(content)} characters)")
        else:
            print("ERROR: No content extracted from file!")
            return False
            
        # Now try to index the file
        print("Indexing file...")
        vector_search.index_file(test_file, content)
        print("File indexed successfully!")
        
        # Check if file is in the collection
        results = vector_search.collection.get(
            ids=[test_file], include=['metadatas']
        )
        
        if results and results['ids'] and len(results['ids']) > 0:
            print("Verification successful! File found in vector store.")
            print(f"Metadata: {results['metadatas'][0]}")
            
            # Try to search for the file
            print("\nTesting search functionality...")
            search_term = "debugging script"
            search_results = vector_search.search(search_term)
            
            if search_results and any(r['path'] == test_file for r in search_results):
                print(f"Search successful! Found file with query '{search_term}'")
            else:
                print(f"File not found in search results with query '{search_term}'")
                print(f"Got {len(search_results) if search_results else 0} results")
                
            return True
        else:
            print("ERROR: File not found in vector store after indexing!")
            return False
            
    except Exception as e:
        print(f"ERROR during indexing test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        

def check_chroma_directory():
    """Check the state of the ChromaDB directory."""
    print("\n=== ChromaDB Directory Check ===\n")
    
    chroma_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".chroma")
    
    if not os.path.exists(chroma_dir):
        print(f"ChromaDB directory does not exist: {chroma_dir}")
        print("This suggests ChromaDB has never been successfully initialized.")
        return False
    
    print(f"ChromaDB directory exists: {chroma_dir}")
    
    # Check contents
    try:
        items = os.listdir(chroma_dir)
        print(f"Directory contains {len(items)} items:")
        for item in items:
            item_path = os.path.join(chroma_dir, item)
            if os.path.isdir(item_path):
                sub_items = len(os.listdir(item_path))
                print(f"  - {item}/ (directory with {sub_items} items)")
            else:
                size = os.path.getsize(item_path)
                print(f"  - {item} ({size} bytes)")
        
        return len(items) > 0
    except Exception as e:
        print(f"Error checking ChromaDB directory: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def check_permissions():
    """Check file system permissions for the ChromaDB directory."""
    print("\n=== File System Permissions Check ===\n")
    
    try:
        # Try to create a test file in the current directory
        test_file = "permission_test.txt"
        try:
            with open(test_file, "w") as f:
                f.write("Testing write permissions")
            print(f"Successfully wrote to test file: {test_file}")
            os.remove(test_file)
            print("Successfully deleted test file")
        except Exception as e:
            print(f"ERROR: Cannot write to current directory: {str(e)}")
            return False
        
        # Check if .chroma directory is writable
        chroma_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".chroma")
        if os.path.exists(chroma_dir):
            try:
                test_file = os.path.join(chroma_dir, "permission_test.txt")
                with open(test_file, "w") as f:
                    f.write("Testing write permissions")
                print(f"Successfully wrote to ChromaDB directory: {test_file}")
                os.remove(test_file)
                print("Successfully deleted test file from ChromaDB directory")
            except Exception as e:
                print(f"ERROR: Cannot write to ChromaDB directory: {str(e)}")
                return False
        
        return True
    except Exception as e:
        print(f"Error checking permissions: {str(e)}")
        return False


if __name__ == "__main__":
    print("\n=== Vector Search Debug Tool ===")
    print("This tool will help diagnose issues with the vector search functionality.")
    
    # Check ChromaDB directory
    chroma_dir_ok = check_chroma_directory()
    
    # Check permissions
    permissions_ok = check_permissions()
    
    # Test indexing
    indexing_ok = test_indexing()
    
    # Summary
    print("\n=== Summary ===")
    print(f"ChromaDB directory check: {'PASSED' if chroma_dir_ok else 'FAILED'}")
    print(f"Permissions check: {'PASSED' if permissions_ok else 'FAILED'}")
    print(f"Indexing test: {'PASSED' if indexing_ok else 'FAILED'}")
    
    if not chroma_dir_ok or not permissions_ok or not indexing_ok:
        print("\nThere are issues with your vector search setup that need to be fixed.")
        print("See the detailed output above for specific errors and suggestions.")
    else:
        print("\nAll tests passed! Vector search appears to be working correctly.")
        print("If you're still having issues, the problem might be in your tagging code.")