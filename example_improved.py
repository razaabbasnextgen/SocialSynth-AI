#!/usr/bin/env python
"""
SocialSynth-AI: Enhanced Knowledge Graph Example
================================================

This script demonstrates the usage of the improved knowledge graph builder
with advanced visualization and top keywords functionality.

Created by: Raza Abbas
"""

import os
import logging
import argparse
import time
from typing import List, Dict, Any
from langchain.schema import Document
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import webbrowser
import json

# Import our classes
from enhanced_rag_retriever import EnhancedRetriever
from enhanced_knowledge_graph_v2 import EnhancedKnowledgeGraphV2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project branding
PROJECT_NAME = "SocialSynth-AI"
PROJECT_AUTHOR = "Raza Abbas"
LOGO_PATH = "logo.png"

def get_api_keys() -> Dict[str, str]:
    """Load API keys from environment variables."""
    load_dotenv()
    
    keys = {
        "youtube_api_key": os.getenv("YOUTUBE_API_KEY"),
        "news_api_key": os.getenv("NEWS_API_KEY"),
        "google_api_key": os.getenv("GOOGLE_API_KEY"),
    }
    
    # Check if keys are available
    missing_keys = [k for k, v in keys.items() if not v]
    if missing_keys:
        logger.warning(f"Missing API keys: {', '.join(missing_keys)}")
    
    return keys

def retrieve_documents(query: str, max_results: int, min_score: float) -> List[Document]:
    """
    Retrieve relevant documents for the query.
    
    Args:
        query: Search query
        max_results: Maximum results per source
        min_score: Minimum relevance score
        
    Returns:
        List of retrieved documents
    """
    # Get API keys
    api_keys = get_api_keys()
    
    # Initialize the enhanced retriever
    retriever = EnhancedRetriever(
        youtube_api_key=api_keys.get("youtube_api_key"),
        news_api_key=api_keys.get("news_api_key"),
        max_results_per_source=max_results,
        min_relevance_score=min_score,
        include_entities=True  # Important for knowledge graph
    )
    
    logger.info(f"Retrieving documents for query: '{query}'")
    start_time = time.time()
    documents = retriever.get_relevant_documents(query)
    
    # Log information about retrieved documents
    sources = {}
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        sources[source] = sources.get(source, 0) + 1
    
    logger.info(f"Retrieved {len(documents)} documents in {time.time() - start_time:.2f} seconds")
    for source, count in sources.items():
        logger.info(f"- {source}: {count} documents")
    
    return documents

def build_knowledge_graph(
    query: str, 
    documents: List[Document], 
    output_path: str,
    max_entities: int,
    max_keywords: int,
    min_relevance: float
) -> Dict[str, Any]:
    """
    Build and visualize a knowledge graph.
    
    Args:
        query: Search query
        documents: List of documents
        output_path: Path to save visualization
        max_entities: Maximum entities per document
        max_keywords: Maximum number of top keywords
        min_relevance: Minimum relevance score for documents
        
    Returns:
        Dictionary with graph summary and export paths
    """
    logger.info(f"Building knowledge graph for query: '{query}'")
    start_time = time.time()
    
    # Initialize the knowledge graph builder
    graph_builder = EnhancedKnowledgeGraphV2(
        max_entities_per_doc=max_entities,
        max_keywords=max_keywords,
        relevance_threshold=min_relevance
    )
    
    # Build the graph
    graph = graph_builder.build_graph(documents, query)
    
    # Get graph summary
    summary = graph_builder.get_graph_summary(graph)
    
    # Visualize the graph
    title = f"Knowledge Graph for: {query}"
    output_path = graph_builder.visualize(graph, output_path, title)
    
    # Export data
    export_dir = os.path.join(os.path.dirname(output_path), "knowledge_graph_data")
    export_paths = graph_builder.export_data(graph, export_dir)
    
    logger.info(f"Knowledge graph built in {time.time() - start_time:.2f} seconds")
    logger.info(f"- Nodes: {summary['node_count']}")
    logger.info(f"- Edges: {summary['edge_count']}")
    logger.info(f"- Visualization saved to: {output_path}")
    
    return {
        "summary": summary,
        "graph": graph,
        "visualization_path": output_path,
        "export_paths": export_paths
    }

def create_summary_visualizations(summary: Dict[str, Any], output_dir: str) -> Dict[str, str]:
    """
    Create summary visualizations from the graph data.
    
    Args:
        summary: Graph summary dictionary
        output_dir: Directory to save visualizations
        
    Returns:
        Dictionary with paths to visualization files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    visualization_paths = {}
    
    # 1. Create node types pie chart
    try:
        plt.figure(figsize=(10, 6))
        node_types = summary.get("node_types", {})
        if node_types:
            labels = list(node_types.keys())
            sizes = list(node_types.values())
            colors = sns.color_palette("pastel", len(labels))
            
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title("Distribution of Node Types")
            
            node_types_path = os.path.join(output_dir, "node_types_pie.png")
            plt.savefig(node_types_path, dpi=300, bbox_inches="tight")
            plt.close()
            
            visualization_paths["node_types_pie"] = node_types_path
    except Exception as e:
        logger.error(f"Error creating node types pie chart: {e}")
    
    # 2. Create entity types bar chart
    try:
        plt.figure(figsize=(12, 6))
        entity_types = summary.get("entity_types", {})
        if entity_types:
            types = list(entity_types.keys())
            counts = list(entity_types.values())
            
            bars = plt.bar(types, counts, color=sns.color_palette("husl", len(types)))
            plt.xticks(rotation=45, ha="right")
            plt.title("Entity Types Distribution")
            plt.xlabel("Entity Type")
            plt.ylabel("Count")
            plt.tight_layout()
            
            # Add count labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom')
            
            entity_types_path = os.path.join(output_dir, "entity_types_bar.png")
            plt.savefig(entity_types_path, dpi=300, bbox_inches="tight")
            plt.close()
            
            visualization_paths["entity_types_bar"] = entity_types_path
    except Exception as e:
        logger.error(f"Error creating entity types bar chart: {e}")
    
    # 3. Create top keywords bar chart
    try:
        plt.figure(figsize=(12, 6))
        top_keywords = summary.get("top_keywords", [])
        if top_keywords:
            keywords = [k for k, _ in top_keywords[:10]]
            scores = [s for _, s in top_keywords[:10]]
            
            # Create horizontal bar chart
            bars = plt.barh(keywords, scores, color=sns.color_palette("YlOrRd", len(keywords)))
            plt.title("Top Keywords (TF-IDF Score)")
            plt.xlabel("TF-IDF Score")
            plt.tight_layout()
            
            # Add score labels on bars
            for bar in bars:
                width = bar.get_width()
                plt.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                        f'{width:.4f}', ha='left', va='center')
            
            keywords_path = os.path.join(output_dir, "top_keywords_bar.png")
            plt.savefig(keywords_path, dpi=300, bbox_inches="tight")
            plt.close()
            
            visualization_paths["keywords_bar"] = keywords_path
    except Exception as e:
        logger.error(f"Error creating top keywords bar chart: {e}")
    
    return visualization_paths

def generate_html_report(
    query: str,
    documents: List[Document],
    graph_result: Dict[str, Any],
    visualization_paths: Dict[str, str],
    output_path: str
) -> str:
    """
    Generate an HTML report with visualizations and summary information.
    
    Args:
        query: Search query
        documents: List of documents
        graph_result: Result from build_knowledge_graph
        visualization_paths: Paths to visualization images
        output_path: Path to save the HTML report
        
    Returns:
        Path to the saved HTML report
    """
    try:
        summary = graph_result["summary"]
        
        # Prepare source statistics
        source_stats = {}
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            source_stats[source] = source_stats.get(source, 0) + 1
        
        # Prepare top entities
        top_entities = summary.get("top_entities", [])
        
        # Generate HTML
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{PROJECT_NAME} - Knowledge Graph Analysis</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    color: #333;
                }}
                header {{
                    display: flex;
                    align-items: center;
                    margin-bottom: 30px;
                    border-bottom: 1px solid #eee;
                    padding-bottom: 20px;
                }}
                .logo {{
                    height: 80px;
                    margin-right: 20px;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .card {{
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    padding: 20px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .stats {{
                    display: flex;
                    justify-content: space-between;
                    flex-wrap: wrap;
                }}
                .stat-item {{
                    flex: 1;
                    min-width: 120px;
                    text-align: center;
                    padding: 15px;
                    background-color: #f9f9f9;
                    border-radius: 8px;
                    margin: 5px;
                }}
                .stat-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #3498db;
                }}
                .stat-label {{
                    color: #7f8c8d;
                }}
                .visualization {{
                    margin: 20px 0;
                    text-align: center;
                }}
                .visualization img {{
                    max-width: 100%;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                th, td {{
                    padding: 10px;
                    border: 1px solid #ddd;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .button {{
                    display: inline-block;
                    padding: 10px 20px;
                    background-color: #3498db;
                    color: white;
                    text-decoration: none;
                    border-radius: 4px;
                    transition: background-color 0.3s;
                }}
                .button:hover {{
                    background-color: #2980b9;
                }}
                footer {{
                    margin-top: 30px;
                    padding-top: 20px;
                    border-top: 1px solid #eee;
                    text-align: center;
                    color: #7f8c8d;
                }}
            </style>
        </head>
        <body>
            <header>
                <img src="logo.png" alt="{PROJECT_NAME} Logo" class="logo">
                <div>
                    <h1>{PROJECT_NAME}</h1>
                    <p>Knowledge Graph Analysis Report</p>
                </div>
            </header>
            
            <div class="card">
                <h2>Query Information</h2>
                <p><strong>Query:</strong> {query}</p>
                <p><strong>Analysis Date:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Documents Retrieved:</strong> {len(documents)}</p>
                
                <div class="stats">
                    <div class="stat-item">
                        <div class="stat-value">{summary['node_count']}</div>
                        <div class="stat-label">Nodes</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{summary['edge_count']}</div>
                        <div class="stat-label">Edges</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{len(summary.get('entity_types', {}))}</div>
                        <div class="stat-label">Entity Types</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{len(summary.get('top_keywords', []))}</div>
                        <div class="stat-label">Top Keywords</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>Knowledge Graph Visualization</h2>
                <p>Interactive visualization available at: <a href="{graph_result['visualization_path']}" target="_blank">{graph_result['visualization_path']}</a></p>
                <div style="text-align: center; margin: 20px 0;">
                    <a href="{graph_result['visualization_path']}" class="button" target="_blank">View Interactive Knowledge Graph</a>
                </div>
            </div>
            
            <div class="card">
                <h2>Document Sources</h2>
                <table>
                    <tr>
                        <th>Source</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
        """
        
        # Add source statistics rows
        for source, count in source_stats.items():
            percentage = (count / len(documents)) * 100
            html += f"""
                    <tr>
                        <td>{source}</td>
                        <td>{count}</td>
                        <td>{percentage:.1f}%</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
            
            <div class="card">
                <h2>Top Keywords</h2>
        """
        
        # Add visualization if available
        if "keywords_bar" in visualization_paths:
            html += f"""
                <div class="visualization">
                    <img src="{visualization_paths['keywords_bar']}" alt="Top Keywords">
                </div>
            """
        
        # Add keywords table
        html += """
                <table>
                    <tr>
                        <th>#</th>
                        <th>Keyword</th>
                        <th>TF-IDF Score</th>
                    </tr>
        """
        
        for i, (keyword, score) in enumerate(summary.get("top_keywords", []), 1):
            html += f"""
                    <tr>
                        <td>{i}</td>
                        <td>{keyword}</td>
                        <td>{score:.4f}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
            
            <div class="card">
                <h2>Top Entities</h2>
                <table>
                    <tr>
                        <th>#</th>
                        <th>Entity</th>
                        <th>Type</th>
                        <th>Centrality</th>
                    </tr>
        """
        
        for i, entity in enumerate(top_entities, 1):
            html += f"""
                    <tr>
                        <td>{i}</td>
                        <td>{entity['text']}</td>
                        <td>{entity['type']}</td>
                        <td>{entity['centrality']:.4f}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
            
            <div class="card">
                <h2>Graph Statistics</h2>
                <div style="display: flex; flex-wrap: wrap;">
        """
        
        # Add visualizations
        for vis_name, vis_path in visualization_paths.items():
            if vis_name not in ["keywords_bar"]:  # Skip keywords visualization as it's shown above
                html += f"""
                    <div class="visualization" style="flex: 1; min-width: 300px;">
                        <img src="{vis_path}" alt="{vis_name}">
                    </div>
                """
        
        html += """
                </div>
            </div>
            
            <footer>
                <p>Generated by {PROJECT_NAME} - Created by {PROJECT_AUTHOR}</p>
            </footer>
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        
        logger.info(f"HTML report saved to: {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error generating HTML report: {e}")
        return ""

def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description=f"{PROJECT_NAME} - Enhanced Knowledge Graph Example")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--output", default="knowledge_graph.html", help="Output file path for visualization")
    parser.add_argument("--max_results", type=int, default=10, help="Maximum results per source")
    parser.add_argument("--min_score", type=float, default=0.5, help="Minimum relevance score")
    parser.add_argument("--max_entities", type=int, default=20, help="Maximum entities per document")
    parser.add_argument("--max_keywords", type=int, default=30, help="Maximum number of top keywords")
    parser.add_argument("--open_browser", action="store_true", help="Open browser with results after completion")
    args = parser.parse_args()
    
    # Print banner
    print(f"\n{'=' * 60}")
    print(f"{PROJECT_NAME} - Enhanced Knowledge Graph Example")
    print(f"Created by: {PROJECT_AUTHOR}")
    print(f"{'=' * 60}\n")
    
    # Retrieve documents
    documents = retrieve_documents(
        query=args.query,
        max_results=args.max_results,
        min_score=args.min_score
    )
    
    if not documents:
        logger.error("No documents retrieved. Cannot build knowledge graph.")
        return
    
    # Build knowledge graph
    graph_result = build_knowledge_graph(
        query=args.query,
        documents=documents,
        output_path=args.output,
        max_entities=args.max_entities,
        max_keywords=args.max_keywords,
        min_relevance=args.min_score
    )
    
    # Create summary visualizations
    output_dir = os.path.join(os.path.dirname(args.output), "visualizations")
    visualization_paths = create_summary_visualizations(
        summary=graph_result["summary"],
        output_dir=output_dir
    )
    
    # Generate HTML report
    report_path = os.path.join(os.path.dirname(args.output), "knowledge_graph_report.html")
    html_report = generate_html_report(
        query=args.query,
        documents=documents,
        graph_result=graph_result,
        visualization_paths=visualization_paths,
        output_path=report_path
    )
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("Knowledge Graph Analysis Complete!")
    print(f"{'=' * 60}")
    print(f"- Interactive visualization: {args.output}")
    print(f"- HTML report: {report_path}")
    print(f"- Data exports: {os.path.join(os.path.dirname(args.output), 'knowledge_graph_data')}")
    print(f"- Visualizations: {output_dir}")
    
    # Open browser if requested
    if args.open_browser and html_report:
        print("\nOpening report in browser...")
        webbrowser.open(f"file://{os.path.abspath(html_report)}")

if __name__ == "__main__":
    main() 