"""
Enhanced Retriever Adapter

This module provides adapters for retrieving documents from various sources,
including external APIs for real-time data retrieval.
"""

import os
import logging
import time
from typing import List, Dict, Any, Optional, Union
import json
import re
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import Document class with fallback
try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document

logger = logging.getLogger("socialsynth.retrieval")

# Try to import required API clients
try:
    import googleapiclient.discovery
    from googleapiclient.errors import HttpError
    YOUTUBE_API_AVAILABLE = True
except ImportError:
    logger.warning("Google API client not found. YouTube retrieval will be disabled.")
    YOUTUBE_API_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    logger.warning("Requests library not found. External API retrieval will be limited.")
    REQUESTS_AVAILABLE = False

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

class RetrieverAdapter:
    """
    Adapter for retrieving documents from various sources.
    """
    
    def __init__(
        self, 
        youtube_api_key: Optional[str] = None,
        news_api_key: Optional[str] = None,
        blog_search_api_key: Optional[str] = None,
        relevance_threshold: float = 0.5,
        max_results_per_source: int = 5
    ):
        """
        Initialize the adapter with API keys and configuration.
        
        Args:
            youtube_api_key: API key for YouTube Data API
            news_api_key: API key for News API
            blog_search_api_key: API key for blog search API
            relevance_threshold: Minimum relevance score for documents
            max_results_per_source: Maximum number of results to return per source
        """
        # Initialize API keys from parameters or environment variables
        self.youtube_api_key = youtube_api_key or os.environ.get("YOUTUBE_API_KEY", "")
        self.news_api_key = news_api_key or os.environ.get("NEWS_API_KEY", "")
        self.blog_search_api_key = blog_search_api_key or os.environ.get("BLOG_SEARCH_API_KEY", "")
        
        # Configuration
        self.relevance_threshold = relevance_threshold
        self.max_results_per_source = max_results_per_source
        
        # Log available API keys (masked for security)
        logger.info(f"YouTube API Key available: {bool(self.youtube_api_key)}")
        logger.info(f"News API Key available: {bool(self.news_api_key)}")
        logger.info(f"Blog Search API Key available: {bool(self.blog_search_api_key)}")
    
    def retrieve(self, query: str, sources: List[str] = None) -> List[Document]:
        """
        Retrieve documents from specified sources based on the query.
        
        Args:
            query: The search query
            sources: List of sources to retrieve from (youtube, news, blogs)
                    If None, all available sources will be used
                    
        Returns:
            List of Document objects
        """
        if not query:
            logger.warning("Empty query provided to retrieve method")
            return []
        
        documents = []
        
        # Determine which sources to use
        if sources is None:
            # Use all available sources based on API keys
            sources = []
            if self.youtube_api_key and YOUTUBE_API_AVAILABLE:
                sources.append("youtube")
            if self.news_api_key and REQUESTS_AVAILABLE:
                sources.append("news")
            if self.blog_search_api_key and REQUESTS_AVAILABLE:
                sources.append("blogs")
                
            if not sources:
                logger.warning("No API keys available for external retrieval")
                return []
        
        logger.info(f"Retrieving documents for query '{query}' from sources: {sources}")
        
        # Retrieve from each source
        for source in sources:
            source_docs = []
            
            if source.lower() == "youtube" and self.youtube_api_key and YOUTUBE_API_AVAILABLE:
                source_docs = self._retrieve_from_youtube(query)
            elif source.lower() == "news" and self.news_api_key and REQUESTS_AVAILABLE:
                source_docs = self._retrieve_from_news(query)
            elif source.lower() == "blogs" and self.blog_search_api_key and REQUESTS_AVAILABLE:
                source_docs = self._retrieve_from_blogs(query)
            else:
                logger.warning(f"Unsupported or unavailable source: {source}")
                
            logger.info(f"Retrieved {len(source_docs)} documents from {source}")
            documents.extend(source_docs)
            
        logger.info(f"Total documents retrieved: {len(documents)}")
        return documents
    
    def _retrieve_from_youtube(self, query: str) -> List[Document]:
        """
        Retrieve documents from YouTube based on the query.
        
        Args:
            query: The search query
            
        Returns:
            List of Document objects
        """
        if not self.youtube_api_key or not YOUTUBE_API_AVAILABLE:
            logger.warning("YouTube API key not available or Google API client not installed")
            return []
            
        try:
            # Initialize YouTube API client
            youtube = googleapiclient.discovery.build(
                "youtube", "v3", developerKey=self.youtube_api_key, cache_discovery=False
            )
            
            # Search for videos
            search_response = youtube.search().list(
                q=query,
                part="id,snippet",
                maxResults=self.max_results_per_source,
                type="video"
            ).execute()
            
            # Extract video IDs
            video_ids = [item["id"]["videoId"] for item in search_response.get("items", [])]
            
            if not video_ids:
                logger.info("No YouTube videos found for query")
                return []
                
            # Get video details
            videos_response = youtube.videos().list(
                id=",".join(video_ids),
                part="snippet,contentDetails,statistics"
            ).execute()
            
            # Convert to Document objects
            documents = []
            for item in videos_response.get("items", []):
                snippet = item.get("snippet", {})
                statistics = item.get("statistics", {})
                
                content = f"{snippet.get('title', '')}\n\n{snippet.get('description', '')}"
                
                metadata = {
                    "title": snippet.get("title", ""),
                    "source": "youtube",
                    "url": f"https://www.youtube.com/watch?v={item['id']}",
                    "channel": snippet.get("channelTitle", ""),
                    "published_at": snippet.get("publishedAt", ""),
                    "view_count": statistics.get("viewCount", "0"),
                    "like_count": statistics.get("likeCount", "0"),
                    "comment_count": statistics.get("commentCount", "0")
                }
                
                documents.append(Document(page_content=content, metadata=metadata))
                
            logger.info(f"Retrieved {len(documents)} documents from YouTube")
            return documents
            
        except HttpError as e:
            logger.error(f"YouTube API error: {e}")
            return []
        except Exception as e:
            logger.error(f"Error retrieving from YouTube: {e}")
            return []
    
    def _retrieve_from_news(self, query: str) -> List[Document]:
        """
        Retrieve documents from News API based on the query.
        
        Args:
            query: The search query
            
        Returns:
            List of Document objects
        """
        if not self.news_api_key or not REQUESTS_AVAILABLE:
            logger.warning("News API key not available or requests library not installed")
            return []
            
        try:
            # Calculate date range (last 7 days)
            today = datetime.now()
            week_ago = today - timedelta(days=7)
            from_date = week_ago.strftime("%Y-%m-%d")
            
            # Make API request
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "from": from_date,
                "sortBy": "relevancy",
                "language": "en",
                "apiKey": self.news_api_key,
                "pageSize": self.max_results_per_source
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                logger.error(f"News API error: {response.status_code} - {response.text}")
                return []
                
            data = response.json()
            
            # Convert to Document objects
            documents = []
            for article in data.get("articles", []):
                content = f"{article.get('title', '')}\n\n{article.get('description', '')}\n\n{article.get('content', '')}"
                
                metadata = {
                    "title": article.get("title", ""),
                    "source": "news",
                    "url": article.get("url", ""),
                    "source_name": article.get("source", {}).get("name", ""),
                    "published_at": article.get("publishedAt", ""),
                    "author": article.get("author", "")
                }
                
                documents.append(Document(page_content=content, metadata=metadata))
                
            logger.info(f"Retrieved {len(documents)} documents from News API")
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving from News API: {e}")
            return []
    
    def _retrieve_from_blogs(self, query: str) -> List[Document]:
        """
        Retrieve documents from blog search API based on the query.
        
        Args:
            query: The search query
            
        Returns:
            List of Document objects
        """
        if not self.blog_search_api_key or not REQUESTS_AVAILABLE:
            logger.warning("Blog search API key not available or requests library not installed")
            return []
            
        try:
            # This is a placeholder implementation that would need to be
            # replaced with your actual blog search API integration
            
            # Make API request (example using a generic search API)
            url = "https://api.example.com/search/blogs"  # Replace with actual endpoint
            headers = {
                "X-API-Key": self.blog_search_api_key,
                "Content-Type": "application/json"
            }
            
            params = {
                "query": query,
                "limit": self.max_results_per_source
            }
            
            # Simulate response for testing (remove this in production)
            # In a real implementation, this would be:
            # response = requests.get(url, headers=headers, params=params)
            
            # Mock response for demonstration
            logger.warning("Using mock blog search results (not real API call)")
            mock_results = [
                {
                    "title": f"Blog post about {query}",
                    "description": f"This is a sample blog post about {query} with detailed information.",
                    "content": f"Lorem ipsum dolor sit amet, consectetur adipiscing elit. {query} is an important topic in many fields...",
                    "url": "https://example.com/blog/123",
                    "author": "John Doe",
                    "date": datetime.now().isoformat()
                }
            ]
            
            # Convert to Document objects
            documents = []
            for article in mock_results:
                content = f"{article.get('title', '')}\n\n{article.get('description', '')}\n\n{article.get('content', '')}"
                
                metadata = {
                    "title": article.get("title", ""),
                    "source": "blog",
                    "url": article.get("url", ""),
                    "published_at": article.get("date", ""),
                    "author": article.get("author", "")
                }
                
                documents.append(Document(page_content=content, metadata=metadata))
                
            logger.info(f"Retrieved {len(documents)} documents from blog search")
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving from blog search API: {e}")
            return [] 