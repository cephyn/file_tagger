from models import init_db, Tag, File
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Check database
db = init_db()
print("\nDatabase Status:")
print("----------------")
print("All tags:", [t.name for t in db.query(Tag).all()])
print("Total files:", db.query(File).count())

# Check vector collection
print("\nVector Collection Status:")
print("------------------------")
client = chromadb.PersistentClient(path=".chroma")
embedding_function = SentenceTransformerEmbeddingFunction()

try:
    collection = client.get_collection(
        name="file_contents",
        embedding_function=embedding_function
    )
    count = collection.count()
    print(f"Collection exists with {count} documents")
    
    # Try a basic search to verify functionality
    results = collection.query(
        query_texts=["test query"],
        n_results=1
    )
    print("\nTest query results:", results)
    
except Exception as e:
    print("Error accessing collection:", str(e))