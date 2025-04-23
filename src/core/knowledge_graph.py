"""Knowledge graph module for data integration and context retrieval"""

import networkx as nx
from typing import Dict, Any, List, Optional
import logging
from ..data.fetchers.youtube import YouTubeFetcher
from ..data.fetchers.blog import BlogFetcher

logger = logging.getLogger(__name__)

class KnowledgeGraph:
    """Knowledge graph for integrating and retrieving context"""
    
    def __init__(self):
        """Initialize the knowledge graph"""
        self.graph = nx.DiGraph()
        self.sources = []
        logger.info("KnowledgeGraph initialized")
    
    def add_data(self, youtube_data: List[Dict], blog_data: List[Dict]):
        """
        Add data to the knowledge graph
        
        Args:
            youtube_data: List of YouTube video data
            blog_data: List of blog post data
        """
        try:
            # Add YouTube data
            for video in youtube_data:
                self._add_video_node(video)
            
            # Add blog data
            for post in blog_data:
                self._add_blog_node(post)
            
            logger.info(f"Added {len(youtube_data)} videos and {len(blog_data)} blog posts to graph")
        except Exception as e:
            logger.error(f"Error adding data to graph: {e}")
    
    def get_relevant_context(self, query: str) -> str:
        """
        Get relevant context for a query
        
        Args:
            query: The search query
            
        Returns:
            Relevant context as a string
        """
        try:
            # Find relevant nodes
            relevant_nodes = self._find_relevant_nodes(query)
            
            # Extract context
            context = self._extract_context(relevant_nodes)
            
            return context
        except Exception as e:
            logger.error(f"Error getting context: {e}")
            return ""
    
    def get_sources(self) -> List[Dict]:
        """Get list of sources used in the graph"""
        return self.sources
    
    def get_graph_data(self) -> Dict:
        """Get graph data for visualization"""
        return {
            "nodes": list(self.graph.nodes(data=True)),
            "edges": list(self.graph.edges(data=True))
        }
    
    def _add_video_node(self, video: Dict):
        """Add a YouTube video node to the graph"""
        video_id = video.get("id")
        if not video_id:
            return
        
        # Add video node
        self.graph.add_node(
            video_id,
            type="video",
            title=video.get("title"),
            description=video.get("description"),
            url=video.get("url")
        )
        
        # Add to sources
        self.sources.append({
            "type": "video",
            "id": video_id,
            "title": video.get("title"),
            "url": video.get("url")
        })
        
        # Add edges to related content
        for tag in video.get("tags", []):
            self.graph.add_edge(video_id, tag, type="has_tag")
    
    def _add_blog_node(self, post: Dict):
        """Add a blog post node to the graph"""
        post_id = post.get("id")
        if not post_id:
            return
        
        # Add blog node
        self.graph.add_node(
            post_id,
            type="blog",
            title=post.get("title"),
            content=post.get("content"),
            url=post.get("url")
        )
        
        # Add to sources
        self.sources.append({
            "type": "blog",
            "id": post_id,
            "title": post.get("title"),
            "url": post.get("url")
        })
        
        # Add edges to related content
        for tag in post.get("tags", []):
            self.graph.add_edge(post_id, tag, type="has_tag")
    
    def _find_relevant_nodes(self, query: str) -> List[str]:
        """Find nodes relevant to the query"""
        # Simple keyword matching for now
        # TODO: Implement more sophisticated relevance scoring
        relevant_nodes = []
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            if query.lower() in node_data.get("title", "").lower():
                relevant_nodes.append(node)
        return relevant_nodes
    
    def _extract_context(self, nodes: List[str]) -> str:
        """Extract context from relevant nodes"""
        context = []
        for node in nodes:
            node_data = self.graph.nodes[node]
            if node_data["type"] == "video":
                context.append(f"Video: {node_data['title']}\n{node_data['description']}")
            elif node_data["type"] == "blog":
                context.append(f"Blog: {node_data['title']}\n{node_data['content']}")
        return "\n\n".join(context) 