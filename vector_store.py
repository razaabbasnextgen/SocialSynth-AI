from typing import List, Dict, Any, Optional
import os
from datetime import datetime
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_qdrant import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

# Download required NLTK data
nltk.download('punkt')

class EnhancedVectorStore:
    def __init__(
        self,
        collection_name: str = "scripts",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None
    ):
        self.collection_name = collection_name
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Initialize Qdrant client
        if qdrant_url and qdrant_api_key:
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            self.client = QdrantClient(":memory:")
            
        # Create collection if it doesn't exist
        try:
            self.client.get_collection(collection_name=collection_name)
        except Exception:
            # Collection doesn't exist, create it
            # all-MiniLM-L6-v2 has 384 dimensions
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=384,  # Fixed size for the specific model
                    distance=models.Distance.COSINE
                )
            )
        
        # Initialize vector store
        self.vector_store = Qdrant(
            client=self.client,
            collection_name=collection_name,
            embeddings=self.embeddings
        )
        
        # Initialize BM25
        self.bm25 = None
        self.documents = []
        self.tfidf = TfidfVectorizer()
        
        logging.info(f"Initialized EnhancedVectorStore with collection: {collection_name}")

    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add documents to both vector store and BM25 index."""
        if not documents:
            return
            
        # Add to vector store
        self.vector_store.add_texts(documents, metadatas=metadata)
        
        # Add to BM25
        self.documents.extend(documents)
        tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        logging.info(f"Added {len(documents)} documents to vector store and BM25 index")

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        vector_weight: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using both vector similarity and BM25.
        
        Args:
            query: Search query
            k: Number of results to return
            vector_weight: Weight for vector similarity scores (0-1)
            
        Returns:
            List of documents with combined scores
        """
        # Vector similarity search
        vector_results = self.vector_store.similarity_search_with_score(
            query,
            k=k
        )
        
        # BM25 search
        tokenized_query = word_tokenize(query.lower())
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[-k:][::-1]
        
        # Combine results
        combined_results = []
        seen_docs = set()
        
        # Process vector results
        for doc, score in vector_results:
            doc_id = hash(doc.page_content)
            if doc_id not in seen_docs:
                combined_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score * vector_weight
                })
                seen_docs.add(doc_id)
        
        # Process BM25 results
        for idx in bm25_indices:
            doc = self.documents[idx]
            doc_id = hash(doc)
            if doc_id not in seen_docs:
                combined_results.append({
                    "content": doc,
                    "metadata": {},
                    "score": bm25_scores[idx] * (1 - vector_weight)
                })
                seen_docs.add(doc_id)
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        return combined_results[:k]

    def temporal_search(
        self,
        query: str,
        start_date: datetime,
        end_date: datetime,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform temporal-aware search using date filters.
        
        Args:
            query: Search query
            start_date: Start date for filtering
            end_date: End date for filtering
            k: Number of results to return
            
        Returns:
            List of documents within the date range
        """
        # Create date filter
        date_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="date",
                    range=models.DateTimeRange(
                        gte=start_date,
                        lte=end_date
                    )
                )
            ]
        )
        
        # Perform filtered search
        results = self.vector_store.similarity_search_with_score(
            query,
            k=k,
            filter=date_filter
        )
        
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            }
            for doc, score in results
        ]

    def delete_collection(self) -> None:
        """Delete the collection and reset indices."""
        self.client.delete_collection(collection_name=self.collection_name)
        self.documents = []
        self.bm25 = None
        logging.info(f"Deleted collection: {self.collection_name}")

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        collection_info = self.client.get_collection(collection_name=self.collection_name)
        return {
            "collection_name": self.collection_name,
            "vector_count": collection_info.vectors_count,
            "document_count": len(self.documents),
            "bm25_initialized": self.bm25 is not None
        } 