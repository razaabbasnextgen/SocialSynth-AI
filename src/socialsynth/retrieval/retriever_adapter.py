"""
Retriever Adapter

This module provides an adapter for various retriever implementations, 
ensuring consistent interface for document retrieval.
"""

import logging
import os
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document

# Configure logging
logger = logging.getLogger("socialsynth.retrieval.retriever_adapter")

class RetrieverAdapter:
    """
    Adapter class for document retrieval, providing a unified interface 
    across different retriever implementations.
    """
    
    def __init__(
        self,
        embedding_model=None,
        retrieval_mode="hybrid",
        youtube_api_key=None,
        news_api_key=None,
        blog_search_api_key=None,
        use_advanced_rag=False,
        max_results_per_source: int = 10,
        min_relevance_score: float = 0.5,
        include_entities: bool = True
    ):
        """
        Initialize the RetrieverAdapter.
        
        Args:
            embedding_model: The embedding model to use for semantic search
            retrieval_mode: The retrieval mode to use (semantic, keyword, hybrid)
            youtube_api_key: The YouTube API key for retrieving video data
            news_api_key: The News API key for retrieving news data
            blog_search_api_key: The Blog Search API key for retrieving blog data
            use_advanced_rag: Whether to use advanced RAG techniques
            max_results_per_source: Maximum number of results per source
            min_relevance_score: Minimum relevance score to keep documents
            include_entities: Whether to extract entities from documents
        """
        self.embedding_model = embedding_model
        self.retrieval_mode = retrieval_mode
        self.youtube_api_key = youtube_api_key
        self.news_api_key = news_api_key
        self.blog_search_api_key = blog_search_api_key
        self.use_advanced_rag = use_advanced_rag
        self.max_results_per_source = max_results_per_source
        self.min_relevance_score = min_relevance_score
        self.include_entities = include_entities
        
        # Initialize vector store as None - required field that was missing
        self.vector_store = None
        
        # Set up logging
        self.logger = logging.getLogger("socialsynth.retrieval")
        
        # Load API keys from environment if not provided
        if not youtube_api_key:
            self.youtube_api_key = os.environ.get("YOUTUBE_API_KEY")
        if not news_api_key:
            self.news_api_key = os.environ.get("NEWS_API_KEY")
        if not blog_search_api_key:
            self.blog_search_api_key = os.environ.get("BLOG_SEARCH_API_KEY")
            
        try:
            # Initialize TF-IDF vectorizer for keyword retrieval
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words="english"
            )
            
            # Initialize embedding model if provided
            if self.embedding_model:
                self.logger.info(f"Using embedding model: {self.embedding_model}")
            else:
                self.logger.info("No embedding model provided, using TF-IDF only")
                
            self.logger.info(f"RetrieverAdapter initialized with mode: {retrieval_mode}")
            
            # Initialize the underlying retriever
            self._initialize_retriever()
            
        except Exception as e:
            self.logger.error(f"Error initializing RetrieverAdapter: {e}")
            raise
        
    def _initialize_retriever(self):
        """Initialize the underlying retriever implementation."""
        try:
            # First try to import the project's EnhancedRetriever
            from .enhanced_retriever import EnhancedRetriever
            
            logger.info("Successfully imported EnhancedRetriever")
            self.retriever = EnhancedRetriever(
                youtube_api_key=self.youtube_api_key,
                news_api_key=self.news_api_key,
                blog_search_api_key=self.blog_search_api_key,
                max_results_per_source=self.max_results_per_source,
                min_relevance_score=self.min_relevance_score,
                include_entities=self.include_entities
            )
        except (ImportError, Exception) as e:
            logger.error(f"Error initializing retriever: {e}")
            self.retriever = None
    
    def retrieve(self, query: str, sources: List[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve documents for a query.
        
        Args:
            query: The search query
            sources: List of sources to retrieve from (optional, for compatibility)
            
        Returns:
            List of document dictionaries
        """
        if not self.retriever:
            logger.error("Retriever not initialized")
            return []
        
        try:
            # First try get_relevant_documents method
            try:
                # Check if retriever supports sources parameter
                if hasattr(self.retriever, 'get_relevant_documents') and 'sources' in self.retriever.get_relevant_documents.__code__.co_varnames:
                    documents = self.retriever.get_relevant_documents(query, sources=sources)
                else:
                    documents = self.retriever.get_relevant_documents(query)
                return self._convert_documents(documents)
            except (AttributeError, Exception) as e:
                logger.warning(f"Error with get_relevant_documents: {e}")
                # Fall back to _get_relevant_documents method
                try:
                    # Check if retriever supports sources parameter
                    if hasattr(self.retriever, '_get_relevant_documents') and 'sources' in self.retriever._get_relevant_documents.__code__.co_varnames:
                        documents = self.retriever._get_relevant_documents(query, sources=sources)
                    else:
                        documents = self.retriever._get_relevant_documents(query)
                    return self._convert_documents(documents)
                except Exception as e2:
                    logger.error(f"Error with _get_relevant_documents: {e2}")
                    return []
                
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
            
    def _convert_documents(self, documents: List[Any]) -> List[Dict[str, Any]]:
        """
        Convert various document formats to a consistent dictionary format.
        
        Args:
            documents: List of documents in potentially different formats
            
        Returns:
            List of document dictionaries with consistent format
        """
        results = []
        
        for doc in documents:
            # Handle Document objects
            if isinstance(doc, Document):
                doc_dict = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "source": doc.metadata.get("source", "unknown")
                }
                results.append(doc_dict)
            # Handle dictionary format
            elif isinstance(doc, dict):
                # Make sure it has the required fields
                if "content" not in doc and "page_content" in doc:
                    doc["content"] = doc["page_content"]
                if "metadata" not in doc:
                    doc["metadata"] = {}
                if "source" not in doc:
                    doc["source"] = doc.get("metadata", {}).get("source", "unknown")
                results.append(doc)
            # Handle other formats
            else:
                try:
                    # Try to convert to string
                    doc_dict = {
                        "content": str(doc),
                        "metadata": {},
                        "source": "unknown"
                    }
                    results.append(doc_dict)
                except Exception:
                    logger.warning(f"Could not convert document to dictionary format: {type(doc)}")
                    
        return results
        
    def get_relevant_documents(self, query: str) -> List[Dict[str, Any]]:
        """
        Alias for retrieve method to maintain compatibility with both interfaces.
        
        Args:
            query: The search query
            
        Returns:
            List of document dictionaries
        """
        return self.retrieve(query) 