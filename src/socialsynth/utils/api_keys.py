"""
API Key Management for SocialSynth-AI

This module provides functions for loading and validating API keys
from environment variables or .env files.
"""

import os
import logging
from typing import Dict, Optional

from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger("socialsynth.utils")

def load_api_keys() -> Dict[str, str]:
    """
    Load API keys from environment variables or .env file
    
    Returns:
        Dictionary with API keys for various services
    """
    # Ensure .env file is loaded
    load_dotenv()
    
    keys = {
        "youtube_api_key": os.getenv("YOUTUBE_API_KEY"),
        "news_api_key": os.getenv("NEWS_API_KEY"),
        "google_api_key": os.getenv("GOOGLE_API_KEY"),
       # "openai_api_key": os.getenv("OPENAI_API_KEY"),
       # "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
        "blog_search_api_key": os.getenv("BLOG_SEARCH_API_KEY"),
        "blog_search_cx": os.getenv("BLOG_SEARCH_CX"),
    }
    
    # Log which keys are available (without showing actual keys)
    available_keys = [k for k, v in keys.items() if v]
    logger.info(f"Loaded API keys: {', '.join(available_keys)}")
    
    return keys 