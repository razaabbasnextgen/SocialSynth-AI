#!/usr/bin/env python3
"""
Enhanced Knowledge Graph Builder Demo

This script demonstrates how to build and visualize a knowledge graph
using the enhanced knowledge graph builder.
"""

import logging
import os
import sys
import networkx as nx
import nest_asyncio
from pathlib import Path
from typing import List, Dict, Any

# Apply nest_asyncio to handle asyncio in Jupyter-like environments
nest_asyncio.apply()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("enhanced_demo")

# Add the src directory to the path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, src_path)

# Import necessary components
from socialsynth.knowledge_graph.enhanced_knowledge_graph_adapter import EnhancedKnowledgeGraphAdapter

def create_sample_documents() -> List[Dict[str, Any]]:
    """
    Create sample documents for demonstration purposes.
    """
    logger.info("Creating sample documents")
    documents = [
        {
            "content": "Artificial intelligence is transforming how we interact with technology. "
                      "Machine learning algorithms enable computers to learn from data and improve "
                      "over time without explicit programming. Deep learning, a subset of machine "
                      "learning, uses neural networks to analyze complex patterns.",
            "metadata": {
                "title": "Introduction to AI",
                "source": "Tech Magazine",
                "url": "https://example.com/ai-intro"
            }
        },
        {
            "content": "Knowledge graphs represent information as entities and their relationships. "
                      "They are used to organize facts and support semantic search capabilities. "
                      "Google, Facebook, and other tech companies use knowledge graphs to enhance "
                      "their services and provide more relevant search results.",
            "metadata": {
                "title": "Understanding Knowledge Graphs",
                "source": "Data Science Blog",
                "url": "https://example.com/knowledge-graphs"
            }
        },
        {
            "content": "Natural language processing (NLP) enables computers to understand, interpret, "
                      "and respond to human language. NLP applications include machine translation, "
                      "sentiment analysis, and chatbots. Recent advances in transformer models have "
                      "significantly improved NLP capabilities.",
            "metadata": {
                "title": "Natural Language Processing",
                "source": "AI Research Journal",
                "url": "https://example.com/nlp-advances"
            }
        }
    ]
    logger.info(f"Created {len(documents)} sample documents")
    return documents

def main():
    """
    Main demonstration function.
    """
    logger.info("Starting enhanced knowledge graph demo")
    
    # Ensure output directory exists
    output_dir = Path("static")
    output_dir.mkdir(exist_ok=True)
    
    # Create visualization output path
    output_path = output_dir / "enhanced_demo_graph.html"
    
    # Initialize the enhanced knowledge graph adapter
    adapter = EnhancedKnowledgeGraphAdapter(
        relevance_threshold=0.3,  # Lower threshold to catch more entities
        max_entities_per_doc=20,   # Allow more entities per document
        max_keywords=15,           # Limit the number of keywords
        height="800px",            # Taller visualization
        width="100%",              # Full width
        use_gpu=False              # Set to True if GPU is available
    )
    
    # Create sample documents
    documents = create_sample_documents()
    
    try:
        # Build the knowledge graph
        logger.info("Building knowledge graph from documents")
        graph = adapter.build_graph(documents)
        
        logger.info(f"Built knowledge graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        
        # Get a summary of the graph
        summary = adapter.get_graph_summary(graph)
        logger.info(f"Graph summary: {summary}")
        
        # Top entities by centrality
        if 'top_central_entities' in summary:
            logger.info("Top central entities:")
            for entity in summary['top_central_entities']:
                logger.info(f"  - {entity['label']} (centrality: {entity['centrality']:.4f})")
        
        # Visualize the graph
        logger.info(f"Saving visualization to {output_path}")
        viz_path = adapter.visualize(graph, str(output_path))
        
        if viz_path:
            logger.info(f"Visualization saved to: {viz_path}")
            logger.info(f"Open this file in a web browser to view the interactive graph")
        else:
            logger.error("Failed to save visualization")
    
    except Exception as e:
        logger.error(f"Error in demo: {e}", exc_info=True)
    
    logger.info("Enhanced knowledge graph demo completed")

if __name__ == "__main__":
    main() 