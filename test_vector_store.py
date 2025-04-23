import unittest
from datetime import datetime, timedelta
from vector_store import EnhancedVectorStore

class TestEnhancedVectorStore(unittest.TestCase):
    def setUp(self):
        self.vector_store = EnhancedVectorStore(collection_name="test_collection")
        
        # Sample documents
        self.documents = [
            "The quick brown fox jumps over the lazy dog",
            "A quick brown dog runs in the park",
            "The weather is nice today",
            "Python is a great programming language",
            "Machine learning is fascinating"
        ]
        
        # Sample metadata
        self.metadata = [
            {"date": datetime.now() - timedelta(days=i)} for i in range(len(self.documents))
        ]
        
        # Add documents to the store
        self.vector_store.add_documents(self.documents, self.metadata)

    def test_hybrid_search(self):
        # Test basic hybrid search
        results = self.vector_store.hybrid_search("quick brown", k=2)
        self.assertEqual(len(results), 2)
        self.assertTrue(any("quick brown fox" in result["content"] for result in results))
        
        # Test with different weights
        results = self.vector_store.hybrid_search("quick brown", k=2, vector_weight=0.3)
        self.assertEqual(len(results), 2)

    def test_temporal_search(self):
        # Test temporal search
        start_date = datetime.now() - timedelta(days=2)
        end_date = datetime.now()
        
        results = self.vector_store.temporal_search(
            "quick",
            start_date=start_date,
            end_date=end_date,
            k=2
        )
        self.assertLessEqual(len(results), 2)

    def test_collection_stats(self):
        stats = self.vector_store.get_collection_stats()
        self.assertEqual(stats["document_count"], len(self.documents))
        self.assertTrue(stats["bm25_initialized"])

    def tearDown(self):
        self.vector_store.delete_collection()

if __name__ == "__main__":
    unittest.main() 