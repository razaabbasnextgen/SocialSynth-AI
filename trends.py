import logging
import time
from typing import List, Dict, Any, Optional
from pytrends.request import TrendReq

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("trends")

# Circuit breaker configuration
MAX_FAILURES = 5
COOLDOWN_PERIOD = 300  # 5 minutes

# Circuit breaker state
service_status = {
    "trends_api": {"failures": 0, "last_failure": 0, "disabled": False},
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

def get_trending_keywords(topic: str, limit: int = 5) -> List[str]:
    """
    Get trending keywords related to a topic with improved error handling
    and circuit breaker pattern
    """
    # Check if service is disabled by circuit breaker
    if service_status["trends_api"]["disabled"]:
        logger.info("Trends API is currently disabled by circuit breaker")
        return []

    # Handle empty or very short topics
    if not topic or len(topic) < 3:
        logger.warning(f"Topic too short for trend analysis: '{topic}'")
        return []

    # Try to normalize topic for better results
    normalized_topic = topic.lower().strip()

    try:
        # Configure PyTrends with timeout and retries
        pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25), retries=2, backoff_factor=0.5)

        # Try to build payload with proper error handling
        try:
            pytrends.build_payload([normalized_topic], cat=0, timeframe='now 7-d')
        except Exception as e:
            logger.error(f"Failed to build PyTrends payload: {e}")
            update_service_status("trends_api")
            return []

        # Get related queries with proper error handling
        try:
            related = pytrends.related_queries()
        except Exception as e:
            logger.error(f"Failed to get related queries: {e}")
            update_service_status("trends_api")
            return []

        # Comprehensive validation of results
        if not related:
            logger.warning(f"No related data returned for topic: '{topic}'")
            return []

        if normalized_topic not in related:
            logger.warning(f"Topic '{topic}' not found in related queries results")
            return []

        top_data = related.get(normalized_topic, {})
        if not top_data:
            logger.warning(f"No top data found for topic: '{topic}'")
            return []

        if "top" not in top_data or top_data["top"] is None:
            logger.warning(f"No top trending data for topic: '{topic}'")
            return []

        if top_data["top"].empty:
            logger.warning(f"Empty trending data for topic: '{topic}'")
            return []

        # Extract keywords with proper error handling
        try:
            result = top_data["top"]["query"].head(limit).tolist()

            # Mark service as successful
            update_service_status("trends_api", True)
            logger.info(f"Successfully fetched {len(result)} trending keywords for '{topic}'")
            return result
        except Exception as e:
            logger.error(f"Failed to extract keywords from trending data: {e}")
            update_service_status("trends_api")
            return []

    except IndexError as e:
        logger.error(f"PyTrends IndexError: {e} - likely empty results")
        update_service_status("trends_api")
    except Exception as e:
        logger.error(f"PyTrends Error: {e}")
        update_service_status("trends_api")

    # Try alternate method if primary method fails
    try:
        # Try with a different timeframe
        pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25))
        pytrends.build_payload([normalized_topic], cat=0, timeframe='today 1-m')

        # Get related topics instead of queries
        related_topics = pytrends.related_topics()

        if normalized_topic in related_topics and related_topics[normalized_topic]:
            top_topics = related_topics[normalized_topic].get('top')
            if top_topics is not None and not top_topics.empty:
                result = top_topics['topic_title'].head(limit).tolist()
                logger.info(f"Used fallback method to fetch {len(result)} related topics")
                return result
    except Exception as e:
        logger.error(f"PyTrends fallback method failed: {e}")

    return []

def get_interest_over_time(topic: str) -> Dict[str, Any]:
    """
    Get interest over time data for a topic
    """
    # Check if service is disabled by circuit breaker
    if service_status["trends_api"]["disabled"]:
        logger.info("Trends API is currently disabled by circuit breaker")
        return {}

    try:
        pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25))
        pytrends.build_payload([topic], cat=0, timeframe='today 3-m')

        interest = pytrends.interest_over_time()
        if interest.empty:
            logger.warning(f"No interest over time data for topic: '{topic}'")
            return {}

        # Convert to a simple dictionary format
        dates = interest.index.tolist()
        values = interest[topic].tolist()

        result = {
            "dates": [str(date) for date in dates],
            "values": values,
            "topic": topic
        }

        return result
    except Exception as e:
        logger.error(f"Interest over time error: {e}")
        update_service_status("trends_api")
        return {}

def get_trending_by_region(topic: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Get regions where the topic is trending
    """
    # Check if service is disabled by circuit breaker
    if service_status["trends_api"]["disabled"]:
        logger.info("Trends API is currently disabled by circuit breaker")
        return []

    try:
        pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25))
        pytrends.build_payload([topic], cat=0, timeframe='today 3-m')

        by_region = pytrends.interest_by_region(resolution='COUNTRY', inc_low_vol=True)
        if by_region.empty:
            logger.warning(f"No regional interest data for topic: '{topic}'")
            return []

        # Sort by value descending and get top regions
        sorted_regions = by_region.sort_values(by=topic, ascending=False).head(limit)

        # Convert to a list of dictionaries
        result = []
        for country, value in sorted_regions[topic].items():
            result.append({"country": country, "interest": value})

        return result
    except Exception as e:
        logger.error(f"Regional interest error: {e}")
        update_service_status("trends_api")
        return []