from typing import List, Dict, Any, Optional, Callable
import numpy as np
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
import logging
from datetime import datetime

# Download required NLTK data
nltk.download('punkt')

class SimpleVectorStore:
    """A simplified vector store that only uses BM25 search."""
    
    def __init__(self, collection_name: str = "documents"):
        """Initialize the SimpleVectorStore."""
        self.collection_name = collection_name
        self.documents = []
        self.metadata = []
        self.bm25 = None
        logging.info(f"Initialized SimpleVectorStore with collection: {collection_name}")

    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add documents and optional metadata to the store."""
        if not documents:
            return
            
        # Store documents and metadata
        start_idx = len(self.documents)
        self.documents.extend(documents)
        
        # Add metadata if provided, otherwise use empty dicts
        if metadata:
            if len(metadata) != len(documents):
                raise ValueError("Number of metadata items must match number of documents")
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{} for _ in range(len(documents))])
        
        # Update BM25 index
        tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]
        tokenized_all_docs = [word_tokenize(doc.lower()) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_all_docs)
        
        logging.info(f"Added {len(documents)} documents to BM25 index")
        return

    def search(
        self,
        query: str,
        k: int = 5,
        filter_func: Optional[Callable[[Dict[str, Any]], bool]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search documents using BM25 with optional metadata filtering.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_func: Optional function that takes metadata dict and returns bool
            
        Returns:
            List of documents with scores
        """
        if not self.bm25:
            return []
            
        # BM25 search
        tokenized_query = word_tokenize(query.lower())
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Apply filtering if provided
        filtered_indices = []
        if filter_func:
            for i, meta in enumerate(self.metadata):
                if filter_func(meta):
                    filtered_indices.append(i)
            
            # Get scores only for filtered indices
            filtered_scores = [(i, bm25_scores[i]) for i in filtered_indices]
            # Sort by score
            filtered_scores.sort(key=lambda x: x[1], reverse=True)
            # Get top k indices
            top_indices = [i for i, _ in filtered_scores[:k]]
        else:
            # Get top k results without filtering
            top_indices = np.argsort(bm25_scores)[-k:][::-1]
        
        # Format results
        results = []
        for idx in top_indices:
            results.append({
                "content": self.documents[idx],
                "metadata": self.metadata[idx],
                "score": bm25_scores[idx]
            })
        
        return results
        
    def search_by_date_range(
        self,
        query: str,
        start_date: datetime,
        end_date: datetime,
        date_field: str = "date",
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search documents within a date range.
        
        Args:
            query: Search query
            start_date: Start date for filtering
            end_date: End date for filtering
            date_field: Name of the date field in metadata
            k: Number of results to return
            
        Returns:
            List of documents with scores
        """
        def date_filter(metadata: Dict[str, Any]) -> bool:
            if date_field not in metadata:
                return False
                
            # Handle date as string or datetime
            doc_date = metadata[date_field]
            if isinstance(doc_date, str):
                try:
                    doc_date = datetime.fromisoformat(doc_date)
                except ValueError:
                    return False
                    
            return start_date <= doc_date <= end_date
            
        return self.search(query, k=k, filter_func=date_filter)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        return {
            "collection_name": self.collection_name,
            "document_count": len(self.documents),
            "bm25_initialized": self.bm25 is not None
        }

    def clear(self) -> None:
        """Clear all documents from the store."""
        self.documents = []
        self.metadata = []
        self.bm25 = None
        logging.info(f"Cleared collection: {self.collection_name}") 