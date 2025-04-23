#!/usr/bin/env python3
"""
Test script for EnhancedKnowledgeGraphAdapter

This script tests the functionality of the EnhancedKnowledgeGraphAdapter class.
"""

import os
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_adapter")

# First, try to import the adapter
try:
    from src.socialsynth.knowledge_graph.enhanced_knowledge_graph_adapter import (
        KnowledgeGraphBuilderAdapter,
        EnhancedKnowledgeGraphAdapter
    )
    logger.info("Successfully imported adapter classes")
except ImportError as e:
    logger.error(f"Error importing adapter classes: {e}")
    exit(1)

def create_test_documents() -> List[Dict[str, Any]]:
    """Create a set of test documents for the knowledge graph"""
    documents = [
        {
            "content": """
            Artificial Intelligence (AI) is revolutionizing many industries. 
            Machine learning algorithms can analyze large datasets to find patterns.
            Companies like Google and Microsoft are investing heavily in AI research.
            """,
            "metadata": {
                "title": "AI Revolution",
                "source": "Test",
                "url": "https://example.com/ai"
            }
        },
        {
            "content": """
            Climate change is affecting global weather patterns.
            Rising temperatures are causing more extreme weather events.
            Scientists warn that immediate action is needed to reduce carbon emissions.
            The Paris Agreement was signed to address climate change globally.
            """,
            "metadata": {
                "title": "Climate Change Impact",
                "source": "Test",
                "url": "https://example.com/climate"
            }
        },
        {
            "content": """
            Renewable energy sources like solar and wind are becoming more affordable.
            Many countries are transitioning away from fossil fuels.
            The energy transition is necessary to combat climate change.
            Technology improvements have made renewable energy more efficient.
            """,
            "metadata": {
                "title": "Renewable Energy Transition",
                "source": "Test",
                "url": "https://example.com/energy"
            }
        }
    ]
    
    logger.info(f"Created {len(documents)} test documents")
    return documents

def test_basic_adapter():
    """Test the basic KnowledgeGraphBuilderAdapter"""
    logger.info("Testing basic KnowledgeGraphBuilderAdapter")
    
    # Create the adapter
    adapter = KnowledgeGraphBuilderAdapter(
        relevance_threshold=0.3,
        max_entities_per_doc=20,
        max_keywords=15
    )
    
    # Create test documents
    documents = create_test_documents()
    
    # Convert to format expected by the adapter
    langchain_docs = []
    for i, doc in enumerate(documents):
        langchain_docs.append(type('Document', (), {
            'page_content': doc['content'],
            'metadata': doc['metadata']
        }))
    
    # Build the graph
    graph = adapter.build_graph(langchain_docs)
    
    # Print basic graph statistics
    logger.info(f"Basic adapter created graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    # Visualize the graph
    output_path = "output/basic_graph.html"
    adapter.visualize(graph, output_path)
    logger.info(f"Basic graph visualization saved to {output_path}")
    
    # Get graph summary
    summary = adapter.get_graph_summary(graph)
    logger.info(f"Basic graph summary: {summary}")
    
    return graph

def test_enhanced_adapter():
    """Test the EnhancedKnowledgeGraphAdapter"""
    logger.info("Testing EnhancedKnowledgeGraphAdapter")
    
    # Create the adapter
    adapter = EnhancedKnowledgeGraphAdapter(
        relevance_threshold=0.3,
        max_entities_per_doc=20,
        max_keywords=15
    )
    
    # Create test documents
    documents = create_test_documents()
    
    # Convert to format expected by the adapter
    langchain_docs = []
    for i, doc in enumerate(documents):
        langchain_docs.append(type('Document', (), {
            'page_content': doc['content'],
            'metadata': doc['metadata']
        }))
    
    # Build the graph
    graph = adapter.build_graph(langchain_docs)
    
    # Print basic graph statistics
    logger.info(f"Enhanced adapter created graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    # Visualize the graph
    output_path = "output/enhanced_graph.html"
    adapter.visualize(graph, output_path)
    logger.info(f"Enhanced graph visualization saved to {output_path}")
    
    # Get graph summary
    summary = adapter.get_graph_summary(graph)
    logger.info(f"Enhanced graph summary: {summary}")
    
    return graph

def main():
    """Main function to run the tests"""
    logger.info("Starting adapter tests")
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Test basic adapter
    basic_graph = test_basic_adapter()
    
    # Test enhanced adapter
    enhanced_graph = test_enhanced_adapter()
    
    logger.info("Adapter tests completed successfully")

if __name__ == "__main__":
    main() 