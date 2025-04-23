#!/usr/bin/env python3
"""
Simple Demo for SocialSynth-AI Enhanced Components

This script demonstrates the basic functionality of the enhanced
knowledge graph builder and retriever without using heavy resources.
"""

import logging
import networkx as nx
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("simple_demo")

# Import enhanced implementations
try:
    from enhanced_knowledge_graph_v2 import EnhancedKnowledgeGraphBuilder
    from enhanced_rag_retriever import EnhancedRetriever
    logger.info("Successfully imported enhanced implementations")
except ImportError as e:
    logger.error(f"Error importing enhanced implementations: {e}")
    exit(1)

def create_sample_documents() -> List[Dict[str, Any]]:
    """Create a small set of sample documents."""
    logger.info("Creating sample documents")
    
    return [
        {
            "content": "Artificial intelligence is transforming how we interact with technology.",
            "metadata": {
                "source": "doc1",
                "title": "AI Overview"
            }
        },
        {
            "content": "Machine learning models can recognize patterns in large datasets.",
            "metadata": {
                "source": "doc2",
                "title": "Machine Learning"
            }
        },
        {
            "content": "Neural networks are inspired by the human brain's structure.",
            "metadata": {
                "source": "doc3",
                "title": "Neural Networks"
            }
        }
    ]

def build_simple_knowledge_graph(documents: List[Dict[str, Any]]) -> nx.DiGraph:
    """Build a knowledge graph from sample documents."""
    logger.info("Building knowledge graph")
    
    # Create knowledge graph builder
    kg_builder = EnhancedKnowledgeGraphBuilder()
    
    # Build the graph
    graph = kg_builder.build_from_documents(documents)
    
    logger.info(f"Knowledge graph built with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    return graph

def main():
    """Run the simple demo."""
    logger.info("Starting simple demo")
    
    # Create sample documents
    documents = create_sample_documents()
    
    # Build knowledge graph
    graph = build_simple_knowledge_graph(documents)
    
    # Show some basic graph information
    print("\nKnowledge Graph Information:")
    print(f"Number of nodes: {len(graph.nodes)}")
    print(f"Number of edges: {len(graph.edges)}")
    
    # Print node types
    node_types = {}
    for _, attrs in graph.nodes(data=True):
        node_type = attrs.get('type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    print("\nNode types:")
    for node_type, count in node_types.items():
        print(f"  - {node_type}: {count}")
    
    logger.info("Simple demo completed successfully")

if __name__ == "__main__":
    main() 