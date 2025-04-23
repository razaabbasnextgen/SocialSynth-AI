import os
import logging
import argparse
from typing import List, Dict, Any
from langchain.schema import Document
from dotenv import load_dotenv

# Import our classes
from enhanced_rag_retriever import EnhancedRetriever
from knowledge_graph_builder import KnowledgeGraphBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_api_keys() -> Dict[str, str]:
    """Load API keys from environment variables."""
    load_dotenv()
    
    keys = {
        "youtube_api_key": os.getenv("YOUTUBE_API_KEY"),
        "news_api_key": os.getenv("NEWS_API_KEY"),
    }
    
    # Check if keys are available
    missing_keys = [k for k, v in keys.items() if not v]
    if missing_keys:
        logger.warning(f"Missing API keys: {', '.join(missing_keys)}")
    
    return keys

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate knowledge graph from enhanced retrieval")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--max_results", type=int, default=5, help="Maximum results per source")
    parser.add_argument("--min_score", type=float, default=0.6, help="Minimum relevance score")
    parser.add_argument("--output", default="knowledge_graph.html", help="Output file path")
    args = parser.parse_args()
    
    # Get API keys
    api_keys = get_api_keys()
    
    # Initialize the enhanced retriever
    retriever = EnhancedRetriever(
        youtube_api_key=api_keys.get("youtube_api_key"),
        news_api_key=api_keys.get("news_api_key"),
        max_results_per_source=args.max_results,
        min_relevance_score=args.min_score,
        include_entities=True
    )
    
    # Retrieve documents
    logger.info(f"Retrieving documents for query: '{args.query}'")
    documents = retriever.get_relevant_documents(args.query)
    logger.info(f"Retrieved {len(documents)} documents")
    
    # Initialize the knowledge graph builder
    graph_builder = KnowledgeGraphBuilder(
        min_relevance=args.min_score,
        max_entities_per_doc=10
    )
    
    # Build knowledge graph
    graph = graph_builder.build_from_documents(documents, args.query)
    
    # Visualize the graph
    graph_builder.visualize(output_path=args.output, 
                         title=f"Knowledge Graph for '{args.query}'")
    
    # Get stats
    stats = graph_builder.get_summary_stats()
    logger.info("Knowledge Graph Stats:")
    for key, value in stats.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for k, v in value.items():
                logger.info(f"    {k}: {v}")
        else:
            logger.info(f"  {key}: {value}")
    
    # Export data
    graph_builder.export_data(output_dir="./knowledge_graph_data")
    
    logger.info(f"Knowledge graph visualization saved to '{args.output}'")
    logger.info(f"Knowledge graph data exported to './knowledge_graph_data'")

if __name__ == "__main__":
    main() 