"""
Quick diagnostic script to check if vector search is initialized properly
"""

import os
from models import init_db, File
from vector_search import VectorSearch

def main():
    print("\n=== Vector Search Initialization Check ===\n")
    
    try:
        print("Initializing database session...")
        db_session = init_db()
        print("Database session initialized")
        
        print("\nInitializing vector search...")
        vector_search = VectorSearch(db_session)
        
        if vector_search.collection is None:
            print("ERROR: Vector search collection is None!")
            return
            
        print(f"Vector search initialized successfully with {vector_search.collection.count()} documents")
        
        # Test a recent file
        files = db_session.query(File).order_by(File.id.desc()).limit(10).all()
        if files:
            for file in files:
                print(f"\nFound recent file: {file.path}")
                print(f"Tags: {[tag.name for tag in file.tags]}")
                
                # Check if file is in vector store
                try:
                    results = vector_search.collection.get(
                        ids=[file.path], include=["metadatas"]
                    )
                    if results and results['ids'] and len(results['ids']) > 0:
                        print(f"✓ File IS indexed in vector store")
                        print(f"Metadata: {results['metadatas'][0]}")
                    else:
                        print(f"✗ File is NOT indexed in vector store!")
                except Exception as e:
                    print(f"Error checking file in vector store: {str(e)}")
                break
        else:
            print("No files found in database")
            
        print("\nVector search is configured properly.")
    except Exception as e:
        import traceback
        print(f"\nError checking vector search: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()