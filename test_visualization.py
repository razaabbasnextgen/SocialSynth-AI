#!/usr/bin/env python3
"""
Test script for the knowledge graph visualization functionality.
"""

import logging
import os
import sys
import networkx as nx
from pathlib import Path

# Set up logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("visualization_test")

# Add the src directory to the path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, src_path)

# Import the visualization adapter
from socialsynth.knowledge_graph.enhanced_knowledge_graph_adapter import EnhancedKnowledgeGraphAdapter

def create_test_graph():
    """Create a simple test graph for visualization"""
    graph = nx.DiGraph()
    
    # Add document node
    graph.add_node(
        "doc_1",
        label="Sample Document",
        type="document",
        source="Test Data",
        url="",
        color="#3498db"  # Blue for documents
    )
    
    # Add entity nodes
    for i in range(1, 4):
        graph.add_node(
            f"entity_{i}",
            label=f"Entity {i}",
            type="entity",
            color="#e67e22"  # Orange for entities
        )
        
        # Connect document to entity
        graph.add_edge(
            "doc_1",
            f"entity_{i}",
            relation="contains"
        )
    
    # Add concept nodes
    for i in range(1, 3):
        graph.add_node(
            f"concept_{i}",
            label=f"Concept {i}",
            type="concept",
            color="#2ecc71"  # Green for concepts
        )
    
    # Add relationships between entities and concepts
    graph.add_edge("entity_1", "concept_1", relation="related to")
    graph.add_edge("entity_2", "concept_1", relation="defines")
    graph.add_edge("entity_3", "concept_2", relation="part of")
    graph.add_edge("concept_1", "concept_2", relation="includes")
    
    logger.info(f"Created test graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    return graph

def main():
    """Main test function"""
    logger.info("Starting visualization test")
    
    # Ensure static directory exists
    static_dir = Path("static")
    static_dir.mkdir(exist_ok=True)
    
    # Create output path for visualization
    output_path = static_dir / "test_knowledge_graph.html"
    
    # Create an instance of the adapter
    adapter = EnhancedKnowledgeGraphAdapter(
        relevance_threshold=0.3,
        max_entities_per_doc=30,
        max_keywords=20,
        height="700px",
        width="100%",
        use_gpu=False
    )
    
    # Create test graph
    graph = create_test_graph()
    
    # Visualize the graph
    try:
        visualization_path = adapter.visualize(graph, str(output_path))
        logger.info(f"Visualization saved to: {visualization_path}")
        
        # Get graph summary
        summary = adapter.get_graph_summary(graph)
        logger.info(f"Graph summary: {summary}")
        
    except Exception as e:
        logger.error(f"Error during visualization: {e}", exc_info=True)
        
    logger.info("Visualization test completed")

if __name__ == "__main__":
    main() 