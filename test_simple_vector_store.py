import sys
from simple_vector_store import SimpleVectorStore
import logging
from datetime import datetime, timedelta

# Configure basic logging
logging.basicConfig(level=logging.INFO)

def test_simple_vector_store():
    """Test the basic functionality of SimpleVectorStore."""
    print("Creating simple vector store instance...")
    store = SimpleVectorStore(collection_name="test_simple_collection")
    
    # Sample documents
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "A quick brown dog runs in the park",
        "Machine learning is a field of artificial intelligence",
        "Python is a popular programming language used in data science",
        "Natural language processing helps computers understand human language"
    ]
    
    # Sample metadata with dates
    today = datetime.now()
    metadata = [
        {"source": "sample1", "date": today - timedelta(days=4)},
        {"source": "sample2", "date": today - timedelta(days=3)},
        {"source": "sample3", "date": today - timedelta(days=2)},
        {"source": "sample4", "date": today - timedelta(days=1)},
        {"source": "sample5", "date": today}
    ]
    
    # Add documents
    print("Adding documents with metadata...")
    store.add_documents(documents, metadata)
    
    # Get stats
    print("Getting store stats...")
    stats = store.get_stats()
    print(f"Document count: {stats['document_count']}")
    print(f"BM25 initialized: {stats['bm25_initialized']}")
    
    # Basic search
    print("\nTesting search for 'quick brown'...")
    results = store.search("quick brown", k=2)
    
    # Display results
    print("\nSearch Results:")
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"Content: {result['content']}")
        print(f"Metadata: {result['metadata']}")
        print(f"Score: {result['score']:.4f}")
        print()
    
    # Test custom filter function
    print("\nTesting search with custom filter (only 'sample1' source)...")
    
    def custom_filter(metadata):
        return metadata.get("source") == "sample1"
    
    results = store.search("quick", k=5, filter_func=custom_filter)
    
    print("\nFiltered Search Results:")
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"Content: {result['content']}")
        print(f"Metadata: {result['metadata']}")
        print(f"Score: {result['score']:.4f}")
        print()
    
    # Test date range search
    print("\nTesting search by date range (last 2 days)...")
    
    start_date = today - timedelta(days=2)
    end_date = today
    
    results = store.search_by_date_range(
        "language",
        start_date=start_date,
        end_date=end_date,
        k=3
    )
    
    print("\nDate Range Search Results:")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"Content: {result['content']}")
        print(f"Metadata: {result['metadata']}")
        print(f"Date: {result['metadata']['date'].date()}")
        print(f"Score: {result['score']:.4f}")
        print()
    
    # Clear the store
    print("Clearing the store...")
    store.clear()
    
    # Verify cleared state
    stats = store.get_stats()
    print(f"Document count after clearing: {stats['document_count']}")
    print(f"BM25 initialized after clearing: {stats['bm25_initialized']}")
    
    print("Test completed successfully!")

if __name__ == "__main__":
    try:
        test_simple_vector_store()
    except Exception as e:
        print(f"Error during test: {e}")
        sys.exit(1) 