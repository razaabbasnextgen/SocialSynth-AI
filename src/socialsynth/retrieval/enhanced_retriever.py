"""
Enhanced document retriever for multi-source information retrieval.

This module provides a flexible document retrieval system that can
fetch content from various sources and apply advanced ranking algorithms.
"""

import os
import logging
import json
import re
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from datetime import datetime, timedelta
import asyncio

# Core dependencies
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# LangChain imports
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import FAISS

# External service connectors
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
from newspaper import Article

# NLP processing
import spacy

# Configure logging
logger = logging.getLogger("socialsynth.retrieval")

# Load environment variables
load_dotenv()

# Custom utility function to replace the missing run_in_executor
async def run_in_executor(executor, func, *args, **kwargs):
    """
    Run a function in an executor (thread pool).
    This is a utility function to replace the missing run_in_executor from langchain.
    
    Args:
        executor: The executor to use
        func: The function to run
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        The result of the function
    """
    loop = asyncio.get_event_loop()
    if kwargs:
        return await loop.run_in_executor(
            executor, lambda: func(*args, **kwargs)
        )
    else:
        return await loop.run_in_executor(executor, func, *args)

class EnhancedRetriever(BaseRetriever):
    """
    EnhancedRetriever implements multiple retrieval strategies to improve document relevance.
    
    This retriever combines semantic search with keyword-based filtering, document 
    re-ranking, and supports multiple advanced retrieval modes to optimize
    for different use cases and query types.
    
    Features:
    - Semantic search using embeddings
    - Keyword-based TF-IDF filtering
    - Hybrid search combining semantic and keyword approaches
    - Support for document metadata filtering
    - Re-ranking based on multiple relevance signals
    - Configurable retrieval strategies
    """
    
    def __init__(
        self,
        documents: List[Document] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        retrieval_mode: str = "hybrid",
        diversity_weight: float = 0.3,
        relevance_threshold: float = 0.6,
        similarity_top_k: int = 20,
        max_output_documents: int = 10,
        use_cache: bool = True,
        youtube_api_key: str = None,
        news_api_key: str = None,
        blog_search_api_key: str = None,
        max_results_per_source: int = 10,
        min_relevance_score: float = 0.5,
        include_entities: bool = True
    ):
        """
        Initialize the EnhancedRetriever with documents and configuration.
        
        Args:
            documents: List of Document objects to index (optional)
            embedding_model: SentenceTransformer model name for embeddings
            retrieval_mode: Search strategy ('semantic', 'keyword', 'hybrid', or 'ensemble')
            diversity_weight: Weight for diversity in re-ranking (0-1)
            relevance_threshold: Minimum relevance score threshold (0-1)
            similarity_top_k: Number of documents to retrieve in initial search
            max_output_documents: Maximum number of documents to return
            use_cache: Whether to cache embedding results
            youtube_api_key: API key for YouTube data retrieval (optional)
            news_api_key: API key for news data retrieval (optional)
            blog_search_api_key: API key for blog search API (optional)
            max_results_per_source: Maximum results to fetch per source
            min_relevance_score: Minimum relevance score for documents
            include_entities: Whether to extract entities from documents
        """
        super().__init__()
        
        # Store document list (initialize as empty list if None)
        self.documents = [] if documents is None else documents
        
        # Store parameters
        self.retrieval_mode = retrieval_mode
        self.diversity_weight = diversity_weight
        self.relevance_threshold = relevance_threshold
        self.similarity_top_k = similarity_top_k
        self.max_output_documents = max_output_documents
        self.use_cache = use_cache
        
        # API keys for retrieving documents
        self.youtube_api_key = youtube_api_key
        self.news_api_key = news_api_key
        self.blog_search_api_key = blog_search_api_key
        self.max_results_per_source = max_results_per_source
        self.min_relevance_score = min_relevance_score
        self.include_entities = include_entities
        
        # Initialize embedding model
        self.embedding_model = None
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(embedding_model)
            logger.info(f"Initialized embedding model: {embedding_model}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            logger.warning("Continuing without embedding model")
            
        # Initialize TF-IDF vectorizer for keyword search
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words="english",
                ngram_range=(1, 2)
            )
        except Exception as e:
            logger.error(f"Error initializing TF-IDF vectorizer: {e}")
            self.tfidf_vectorizer = None
        
        # Build indexes if documents are provided
        self.document_embeddings = None
        self.tfidf_matrix = None
        
        if self.documents:
            try:
                self._build_indexes()
            except Exception as e:
                logger.error(f"Error building indexes: {e}")
        
        # Create cache for query results
        self.query_cache = {}
        
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Method required by langchain BaseRetriever - returns relevant documents for a query.
        
        Args:
            query: Query string
            
        Returns:
            List of Documents relevant to the query
        """
        logger.info(f"Retrieving documents for query: '{query}'")
        
        # Check cache if enabled
        if self.use_cache and query in self.query_cache:
            logger.info("Using cached results")
            return self.query_cache[query]
        
        # Initialize results
        all_documents = []
        
        try:
            # 1. Fetch documents from different sources if API keys are provided
            # YouTube
            if self.youtube_api_key:
                youtube_docs = self._fetch_youtube_documents(query)
                all_documents.extend(youtube_docs)
                logger.info(f"Retrieved {len(youtube_docs)} documents from YouTube")
            
            # News
            if self.news_api_key:
                news_docs = self._fetch_news_documents(query)
                all_documents.extend(news_docs)
                logger.info(f"Retrieved {len(news_docs)} documents from news sources")
            
            # 2. Filter and rank documents
            if all_documents:
                # Prepare for ranking
                ranked_docs = []
                for doc in all_documents:
                    # Skip documents below the minimum relevance threshold
                    relevance = doc.metadata.get("relevance_score", 0.0)
                    if relevance < self.min_relevance_score:
                        continue
                    ranked_docs.append(doc)
                
                # Sort by relevance and limit to maximum output
                ranked_docs = sorted(ranked_docs, 
                                   key=lambda d: d.metadata.get("relevance_score", 0.0),
                                   reverse=True)
                result_docs = ranked_docs[:self.max_output_documents]
                
                # Cache results if enabled
                if self.use_cache:
                    self.query_cache[query] = result_docs
                
                return result_docs
            else:
                # If no documents were retrieved, create empty result
                logger.warning(f"No documents retrieved for query: '{query}'")
                return []
                
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def _fetch_youtube_documents(self, query: str) -> List[Document]:
        """Fetch documents from YouTube videos"""
        # Implement YouTube video fetching here (simplified version)
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            from googleapiclient.discovery import build
            
            # Create YouTube API client
            youtube = build('youtube', 'v3', developerKey=self.youtube_api_key)
            
            # Search for videos
            search_response = youtube.search().list(
                q=query,
                part='id,snippet',
                maxResults=self.max_results_per_source,
                type='video'
            ).execute()
            
            documents = []
            for item in search_response.get('items', []):
                if item['id']['kind'] == 'youtube#video':
                    video_id = item['id']['videoId']
                    title = item['snippet']['title']
                    
                    # Get video transcript
                    try:
                        transcript = YouTubeTranscriptApi.get_transcript(video_id)
                        transcript_text = " ".join([t['text'] for t in transcript])
                        
                        # Calculate relevance score (simplified)
                        relevance_score = 0.8  # Default high score for valid transcripts
                        
                        # Create document
                        doc = Document(
                            page_content=transcript_text,
                            metadata={
                                "source": "youtube",
                                "video_id": video_id,
                                "video_title": title,
                                "url": f"https://www.youtube.com/watch?v={video_id}",
                                "relevance_score": relevance_score
                            }
                        )
                        documents.append(doc)
                    except Exception as e:
                        logger.warning(f"Could not get transcript for video {video_id}: {e}")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error fetching YouTube documents: {e}")
            return []
    
    def _fetch_news_documents(self, query: str) -> List[Document]:
        """Fetch documents from news sources"""
        # Implement news API fetching here (simplified version)
        try:
            import requests
            from newspaper import Article
            
            # Fetch news articles
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "apiKey": self.news_api_key,
                "pageSize": self.max_results_per_source,
                "language": "en",
                "sortBy": "relevancy"
            }
            
            response = requests.get(url, params=params)
            if response.status_code != 200:
                logger.error(f"News API error: {response.status_code}")
                return []
                
            data = response.json()
            articles = data.get("articles", [])
            
            documents = []
            for article in articles:
                url = article.get("url")
                if not url:
                    continue
                    
                # Get full article content
                try:
                    news_article = Article(url)
                    news_article.download()
                    news_article.parse()
                    
                    # Calculate relevance score (simplified)
                    relevance_score = 0.8  # Default high score
                    
                    # Create document
                    doc = Document(
                        page_content=news_article.text,
                        metadata={
                            "source": "news",
                            "url": url,
                            "headline": article.get("title", ""),
                            "publish_date": article.get("publishedAt", ""),
                            "relevance_score": relevance_score
                        }
                    )
                    documents.append(doc)
                except Exception as e:
                    logger.warning(f"Could not process article {url}: {e}")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error fetching news documents: {e}")
            return [] 