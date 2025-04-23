"""Content generation module with RAG and knowledge graph integration"""

import os
import logging
from typing import Dict, Any, List, Optional
import google.generativeai as genai
from dotenv import load_dotenv
from ..data.fetchers.youtube import YouTubeFetcher
from ..data.fetchers.blog import BlogFetcher
from ..core.knowledge_graph import KnowledgeGraph
from ..core.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

class ContentGenerator:
    """Content generation with RAG and knowledge graph integration"""
    
    def __init__(self):
        """Initialize the content generator with all components"""
        self.model = genai.GenerativeModel('gemini-pro')
        self.knowledge_graph = KnowledgeGraph()
        self.youtube_fetcher = YouTubeFetcher()
        self.blog_fetcher = BlogFetcher()
        self.circuit_breaker = CircuitBreaker()
        logger.info("ContentGenerator initialized successfully")
    
    def generate_content(
        self,
        query: str,
        tone: str = "professional",
        format_type: str = "blog",
        target_audience: str = "general",
        length: str = "medium"
    ) -> Dict[str, Any]:
        """
        Generate content using RAG and knowledge graph
        
        Args:
            query: Main topic or query
            tone: Content tone (professional, casual, academic)
            format_type: Content format (blog, social media, email)
            target_audience: Intended audience
            length: Content length (short, medium, long)
            
        Returns:
            Dictionary containing generated content and metadata
        """
        try:
            # 1. Fetch relevant data
            youtube_data = self.youtube_fetcher.search(query)
            blog_data = self.blog_fetcher.search(query)
            
            # 2. Build knowledge graph
            self.knowledge_graph.add_data(youtube_data, blog_data)
            
            # 3. Get relevant context
            context = self.knowledge_graph.get_relevant_context(query)
            
            # 4. Create system prompt
            system_prompt = self._create_system_prompt(
                tone, format_type, target_audience, length
            )
            
            # 5. Generate content
            response = self.model.generate_content(
                f"{system_prompt}\n\nContext: {context}\n\nTopic: {query}",
                generation_config=self._get_generation_config(length)
            )
            
            return {
                "content": response.text,
                "metadata": {
                    "sources": self.knowledge_graph.get_sources(),
                    "graph": self.knowledge_graph.get_graph_data(),
                    "model": "gemini-pro",
                    "tone": tone,
                    "format": format_type,
                    "audience": target_audience,
                    "length": length
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating content: {e}")
            return {
                "error": str(e),
                "content": None
            }
    
    def _create_system_prompt(
        self,
        tone: str,
        format_type: str,
        target_audience: str,
        length: str
    ) -> str:
        """Create a system prompt based on parameters"""
        return f"""
        You are a professional content writer. Generate {format_type} content with the following specifications:
        - Tone: {tone}
        - Target Audience: {target_audience}
        - Length: {length}
        
        Use the provided context to create engaging, informative, and well-structured content.
        """
    
    def _get_generation_config(self, length: str) -> Dict:
        """Get generation configuration based on length"""
        max_tokens = self._get_max_tokens(length)
        
        return {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": max_tokens,
        }
    
    def _get_max_tokens(self, length: str) -> int:
        """Determine max tokens based on length"""
        length_map = {
            "short": 500,
            "medium": 1000,
            "long": 2000
        }
        return length_map.get(length.lower(), 1000)

# Create singleton instance
content_generator = ContentGenerator() 