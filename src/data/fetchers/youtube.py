"""YouTube data fetcher module"""

import os
from typing import Dict, List, Optional
import logging
from youtubesearchpython import VideosSearch
from ..core.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)

class YouTubeFetcher:
    """YouTube data fetcher with circuit breaker protection"""
    
    def __init__(self):
        """Initialize YouTube fetcher"""
        self.circuit_breaker = CircuitBreaker()
        logger.info("YouTubeFetcher initialized")
    
    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search YouTube videos
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of video data dictionaries
        """
        try:
            return self.circuit_breaker.execute(
                self._search_videos,
                query,
                max_results
            )
        except Exception as e:
            logger.error(f"Error searching YouTube: {e}")
            return []
    
    def _search_videos(self, query: str, max_results: int) -> List[Dict]:
        """Execute YouTube search"""
        videos_search = VideosSearch(query, limit=max_results)
        results = videos_search.result()
        
        videos = []
        for video in results.get("result", []):
            videos.append({
                "id": video.get("id"),
                "title": video.get("title"),
                "description": video.get("description"),
                "url": video.get("link"),
                "thumbnail": video.get("thumbnails", [{}])[0].get("url"),
                "duration": video.get("duration"),
                "views": video.get("viewCount", {}).get("text"),
                "channel": video.get("channel", {}).get("name"),
                "tags": video.get("keywords", [])
            })
        
        return videos
    
    def get_video_details(self, video_id: str) -> Optional[Dict]:
        """
        Get detailed information about a video
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Video details dictionary or None if not found
        """
        try:
            return self.circuit_breaker.execute(
                self._get_video_details,
                video_id
            )
        except Exception as e:
            logger.error(f"Error getting video details: {e}")
            return None
    
    def _get_video_details(self, video_id: str) -> Optional[Dict]:
        """Execute video details fetch"""
        videos_search = VideosSearch(video_id, limit=1)
        results = videos_search.result()
        
        if not results.get("result"):
            return None
        
        video = results["result"][0]
        return {
            "id": video.get("id"),
            "title": video.get("title"),
            "description": video.get("description"),
            "url": video.get("link"),
            "thumbnail": video.get("thumbnails", [{}])[0].get("url"),
            "duration": video.get("duration"),
            "views": video.get("viewCount", {}).get("text"),
            "channel": video.get("channel", {}).get("name"),
            "tags": video.get("keywords", []),
            "publish_date": video.get("publishDate"),
            "likes": video.get("likes", {}).get("text"),
            "comments": video.get("comments", {}).get("text")
        } 