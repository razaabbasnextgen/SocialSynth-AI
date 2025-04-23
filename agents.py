import os
import time
import logging
import requests
from typing import List, Dict, Any, Optional
from youtubesearchpython import VideosSearch
from newspaper import Article
from langchain.schema import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("agents")

# Circuit breaker configuration
MAX_FAILURES = 5
COOLDOWN_PERIOD = 300  # 5 minutes

# Circuit breaker state
service_status = {
    "youtube_api": {"failures": 0, "last_failure": 0, "disabled": False},
    "blog_api": {"failures": 0, "last_failure": 0, "disabled": False},
}

def update_service_status(service_name: str, success: bool = False):
    """Update service status for circuit breaker pattern"""
    if service_name not in service_status:
        return

    current_time = time.time()

    # Reset on success
    if success:
        service_status[service_name]["failures"] = 0
        service_status[service_name]["disabled"] = False
        return

    # Update on failure
    service_status[service_name]["failures"] += 1
    service_status[service_name]["last_failure"] = current_time

    # Check if we should disable the service
    if service_status[service_name]["failures"] >= MAX_FAILURES:
        service_status[service_name]["disabled"] = True
        logger.warning(f"Circuit breaker triggered for {service_name}")

    # Check for cooldown expiration
    if (service_status[service_name]["disabled"] and 
        current_time - service_status[service_name]["last_failure"] > COOLDOWN_PERIOD):
        logger.info(f"Cooldown period expired for {service_name}, re-enabling")
        service_status[service_name]["failures"] = 0
        service_status[service_name]["disabled"] = False

def get_youtube_top_videos(query: str, max_results: int = 5) -> List[Document]:
    """
    Fetch top YouTube videos for a given query using YouTube Data API
    with improved error handling and circuit breaker pattern
    """
    # Check if service is disabled by circuit breaker
    if service_status["youtube_api"]["disabled"]:
        logger.info("YouTube API is currently disabled by circuit breaker")
        return []

    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        logger.error("YOUTUBE_API_KEY not found in environment variables")
        return []

    docs = []
    try:
        # First approach: Use YouTube Data API directly
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet", 
            "q": query, 
            "maxResults": max_results,
            "type": "video", 
            "key": api_key,
            "relevanceLanguage": "en"
        }

        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            logger.warning(f"YouTube API returned status code {response.status_code}")
            # If official API fails, fall back to youtubesearchpython
            videos_search = VideosSearch(query, limit=max_results)
            results = videos_search.result().get("result", [])

            for item in results:
                title = item.get("title", "")
                description = item.get("descriptionSnippet", [{}])[0].get("text", "")
                video_url = item.get("link", "")
                channel = item.get("channel", {}).get("name", "Unknown Channel")

                # Only add valid entries
            if title and video_url:
                docs.append(Document(
                    page_content=f"Title: {title}\nChannel: {channel}\nDescription: {description}",
                    metadata={"source": video_url, "type": "youtube"}
                ))
        else:
            data = response.json()
            for item in data.get("items", []):
                title = item["snippet"]["title"]
                description = item["snippet"].get("description", "")
                channel = item["snippet"].get("channelTitle", "Unknown Channel")
                video_url = f"https://www.youtube.com/watch?v={item['id']['videoId']}"

                docs.append(Document(
                    page_content=f"Title: {title}\nChannel: {channel}\nDescription: {description}",
                    metadata={"source": video_url, "type": "youtube"}
                ))

        # Mark service as successful
        update_service_status("youtube_api", True)
        logger.info(f"Successfully fetched {len(docs)} YouTube videos")
        return docs

    except requests.exceptions.Timeout:
        logger.warning("YouTube API request timed out")
        update_service_status("youtube_api")
    except Exception as e:
        logger.error(f"YouTube API Error: {e}")
        update_service_status("youtube_api")

    # Fallback to youtubesearchpython on any error
    try:
        videos_search = VideosSearch(query, limit=max_results)
        results = videos_search.result().get("result", [])

        for item in results:
            title = item.get("title", "")
            description = item.get("descriptionSnippet", [{}])[0].get("text", "")
            video_url = item.get("link", "")
            channel = item.get("channel", {}).get("name", "Unknown Channel")

            # Only add valid entries
            if title and video_url:
                docs.append(Document(
                    page_content=f"Title: {title}\nChannel: {channel}\nDescription: {description}",
                    metadata={"source": video_url, "type": "youtube"}
                ))

        logger.info(f"Used fallback method to fetch {len(docs)} YouTube videos")
        return docs
    except Exception as e:
        logger.error(f"YouTube fallback search failed: {e}")
        return []

def get_blog_articles(query: str, api_key: str, cx: str, max_results: int = 5) -> List[Document]:
    """
    Fetch blog articles using Google Custom Search API with improved error handling
    """
    # Check if service is disabled by circuit breaker
    if service_status["blog_api"]["disabled"]:
        logger.info("Blog API is currently disabled by circuit breaker")
        return []

    if not api_key or not cx:
        logger.error("Missing API key or CX for blog search")
        return []

    docs = []
    try:
        url = f"https://www.googleapis.com/customsearch/v1"
        params = {
            "q": query,
            "key": api_key,
            "cx": cx,
            "num": max_results
        }

        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            logger.warning(f"Blog search API returned status code {response.status_code}")
            update_service_status("blog_api")
            return []

        res = response.json()

        for item in res.get("items", [])[:max_results]:
            source_url = item.get("link")
            if not source_url:
                continue

            # Skip YouTube and video links in blog search
            if "youtube.com" in source_url or "youtu.be" in source_url:
                continue

            try:
                # Try to extract article text with timeout
                article = Article(source_url)
                article.download()
                article.parse()

                # Skip very short articles
                if len(article.text) < 100:
                    continue

                # Create a document with truncated content to save space
                docs.append(Document(
                    page_content=f"Title: {article.title}\n\n{article.text[:1500]}",
                    metadata={
                        "source": source_url,
                        "type": "blog", 
                        "title": article.title,
                        "publish_date": str(article.publish_date) if article.publish_date else None
                    }
                ))
            except Exception as e:
                logger.error(f"Failed to parse article at {source_url}: {e}")
                # Add minimal document based on Google's search snippet
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                if title and snippet:
                    docs.append(Document(
                        page_content=f"Title: {title}\n\n{snippet}",
                        metadata={"source": source_url, "type": "blog", "title": title}
                    ))

        # Mark service as successful
        update_service_status("blog_api", True)
        logger.info(f"Successfully fetched {len(docs)} blog articles")
        return docs
    except requests.exceptions.Timeout:
        logger.warning("Blog API request timed out")
        update_service_status("blog_api")
        return []
    except Exception as e:
        logger.error(f"Blog API Error: {e}")
        update_service_status("blog_api")
        return []

def get_related_videos(video_id: str, api_key: str, max_results: int = 3) -> List[Document]:
    """
    Fetch related videos for a given video ID
    """
    if not api_key or not video_id:
        return []

    docs = []
    try:
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "relatedToVideoId": video_id,
            "type": "video",
            "maxResults": max_results,
            "key": api_key
        }

        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            logger.warning(f"Related videos API returned status code {response.status_code}")
            return []

        data = response.json()
        for item in data.get("items", []):
            title = item["snippet"]["title"]
            description = item["snippet"].get("description", "")
            video_url = f"https://www.youtube.com/watch?v={item['id']['videoId']}"

            docs.append(Document(
                page_content=f"Related Video - Title: {title}\nDescription: {description}",
                metadata={"source": video_url, "type": "youtube_related"}
            ))

        return docs
    except Exception as e:
        logger.error(f"Related videos error: {e}")
        return []

def extract_video_id(url: str) -> Optional[str]:
    """
    Extract video ID from a YouTube URL
    """
    if "youtube.com/watch" in url and "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return None

def get_video_comments(video_id: str, api_key: str, max_results: int = 10) -> List[Document]:
    """
    Fetch top comments for a YouTube video
    """
    if not api_key or not video_id:
        return []

    docs = []
    try:
        url = "https://www.googleapis.com/youtube/v3/commentThreads"
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": max_results,
            "order": "relevance",
            "key": api_key
        }

        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            logger.warning(f"Comments API returned status code {response.status_code}")
            return []

        data = response.json()
        for item in data.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            likes = item["snippet"]["topLevelComment"]["snippet"]["likeCount"]

            # Only add comments with substance (more than just a few words)
            if len(comment.split()) > 5:
                docs.append(Document(
                    page_content=f"Comment: {comment}\nLikes: {likes}",
                    metadata={"source": f"https://www.youtube.com/watch?v={video_id}", "type": "youtube_comment"}
                ))

        return docs
    except Exception as e:
        logger.error(f"Comments API error: {e}")
        return []
