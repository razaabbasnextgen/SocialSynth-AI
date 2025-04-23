import os
import sys
import traceback
from vector_store import EnhancedVectorStore

def test_basic_functionality():
    """Test basic functionality of the EnhancedVectorStore."""
    print("Creating vector store instance...")
    vector_store = EnhancedVectorStore(collection_name="test_collection_simple")
    
    # Sample documents
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "A quick brown dog runs in the park",
        "Machine learning is a field of artificial intelligence"
    ]
    
    # Add documents
    print("Adding documents...")
    vector_store.add_documents(documents)
    
    # Test basic hybrid search
    print("Testing hybrid search...")
    results = vector_store.hybrid_search("quick brown", k=2)
    
    # Display results
    print("\nSearch Results:")
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"Content: {result.get('content')}")
        print(f"Score: {result.get('score')}")
        print()
    
    # Get collection stats
    print("Getting collection stats...")
    stats = vector_store.get_collection_stats()
    print(f"Document count: {stats.get('document_count')}")
    print(f"Vector count: {stats.get('vector_count')}")
    print(f"BM25 initialized: {stats.get('bm25_initialized')}")
    
    # Delete collection
    print("Cleaning up...")
    vector_store.delete_collection()
    print("Test completed successfully!")

if __name__ == "__main__":
    try:
        test_basic_functionality()
    except Exception as e:
        print(f"Error during test: {e}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1) 