#!/usr/bin/env python3
"""
Enhanced RAG Demonstration Script

This script demonstrates the capabilities of the enhanced RAG system
with knowledge graph integration for improved document retrieval.
"""

import os
import sys
import logging
import time
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any

import networkx as nx
import matplotlib.pyplot as plt
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Import our enhanced implementations
from enhanced_rag_retriever import EnhancedRetriever
from enhanced_knowledge_graph_v2 import EnhancedKnowledgeGraphBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("rag_enhanced.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("demo_enhanced_rag")

def load_sample_documents() -> List[Dict[str, Any]]:
    """
    Load some sample documents for testing.
    
    Returns:
        List of document dictionaries
    """
    logger.info("Loading sample documents")
    
    documents = [
        {
            "content": """Artificial intelligence (AI) is intelligence demonstrated by machines, 
            as opposed to intelligence of humans and other animals. Example tasks in which 
            this is done include speech recognition, computer vision, translation between languages, 
            and decision making. The term "artificial intelligence" is often also used to refer 
            to the field of science and engineering focused on creating machines with the above capabilities.""",
            "metadata": {
                "source": "doc1",
                "title": "Introduction to Artificial Intelligence",
                "author": "AI Researcher",
                "date": "2023-01-15"
            }
        },
        {
            "content": """Machine learning (ML) is a field of inquiry devoted to understanding and 
            building methods that 'learn', that is, methods that leverage data to improve performance 
            on some set of tasks. It is seen as a sub-field of artificial intelligence. 
            Machine learning algorithms build a model based on sample data, known as training data, 
            in order to make predictions or decisions without being explicitly programmed to do so.""",
            "metadata": {
                "source": "doc2",
                "title": "Machine Learning Fundamentals",
                "author": "ML Expert",
                "date": "2023-02-20"
            }
        },
        {
            "content": """Deep learning is part of a broader family of machine learning methods based on 
            artificial neural networks with representation learning. Learning can be supervised, 
            semi-supervised or unsupervised. Deep-learning architectures such as deep neural networks, 
            deep belief networks, deep reinforcement learning, recurrent neural networks and convolutional 
            neural networks have been applied to fields including computer vision, speech recognition, 
            natural language processing, machine translation, bioinformatics, drug design, medical image analysis, 
            climate science and board game programs, where they have produced results comparable to and in some 
            cases surpassing human expert performance.""",
            "metadata": {
                "source": "doc3",
                "title": "Deep Learning Explained",
                "author": "Neural Network Specialist",
                "date": "2023-03-10"
            }
        },
        {
            "content": """Natural Language Processing (NLP) is a subfield of linguistics, computer science, 
            and artificial intelligence concerned with the interactions between computers and human language, 
            in particular how to program computers to process and analyze large amounts of natural language data. 
            The goal is a computer capable of "understanding" the contents of documents, including the contextual 
            nuances of the language within them.""",
            "metadata": {
                "source": "doc4",
                "title": "Natural Language Processing",
                "author": "NLP Researcher",
                "date": "2023-04-05"
            }
        },
        {
            "content": """Retrieval-Augmented Generation (RAG) is a technique that combines retrieval-based methods 
            with generative models to improve the quality and factuality of generated content. In RAG, a retrieval 
            system first finds relevant documents from a corpus, and then a generative model uses these documents 
            as additional context when generating a response. This approach helps ground the generation in 
            factual information and reduces hallucinations. RAG has applications in question answering, 
            chatbots, and content generation systems.""",
            "metadata": {
                "source": "doc5",
                "title": "Retrieval-Augmented Generation",
                "author": "RAG Expert",
                "date": "2023-05-18"
            }
        }
    ]
    
    logger.info(f"Loaded {len(documents)} sample documents")
    return documents

def setup_vector_store(documents: List[Dict[str, Any]]) -> Chroma:
    """
    Set up a vector store with the sample documents.
    
    Args:
        documents: List of document dictionaries
        
    Returns:
        Chroma vector store
    """
    logger.info("Setting up vector store")
    
    try:
        # Convert to LangChain Document objects
        langchain_docs = []
        for doc in documents:
            langchain_docs.append(
                Document(
                    page_content=doc["content"],
                    metadata=doc["metadata"]
                )
            )
        
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Create vector store
        vector_store = Chroma.from_documents(
            documents=langchain_docs,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        
        logger.info("Vector store setup complete")
        return vector_store
        
    except Exception as e:
        logger.error(f"Error setting up vector store: {e}")
        return None

def visualize_graph(graph: nx.DiGraph, output_path: str = "knowledge_graph.html"):
    """
    Visualize the knowledge graph using pyvis.
    
    Args:
        graph: NetworkX graph
        output_path: Path to save the visualization
    """
    logger.info("Visualizing knowledge graph")
    
    try:
        from pyvis.network import Network
        
        # Create a pyvis network
        net = Network(height="750px", width="100%", notebook=False, directed=True)
        
        # Add nodes
        for node, attrs in graph.nodes(data=True):
            label = attrs.get("label", str(node))
            title = f"{label}\nType: {attrs.get('type', 'unknown')}"
            color = attrs.get("color", "#95a5a6")  # Default gray
            
            net.add_node(node, label=label, title=title, color=color)
        
        # Add edges
        for source, target, attrs in graph.edges(data=True):
            weight = attrs.get("weight", 1.0)
            label = attrs.get("label", "")
            
            net.add_edge(source, target, value=weight, title=label, arrowStrikethrough=True)
        
        # Customize physics
        net.barnes_hut(spring_length=200, spring_strength=0.005, damping=0.09)
        
        # Save the visualization
        net.show(output_path)
        logger.info(f"Graph visualization saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error visualizing graph: {e}")

def demo_retrieval(vector_store: Chroma, knowledge_graph: nx.DiGraph):
    """
    Demonstrate enhanced retrieval using both vector store and knowledge graph.
    
    Args:
        vector_store: Chroma vector store
        knowledge_graph: Knowledge graph
    """
    logger.info("Demonstrating enhanced retrieval")
    
    # Initialize the enhanced retriever
    retriever = EnhancedRetriever(
        vector_store=vector_store,
        knowledge_graph=knowledge_graph,
        top_k_vector=3,
        top_k_graph=2,
        similarity_threshold=0.5
    )
    
    # Example queries
    queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What are applications of deep learning?",
        "Explain retrieval-augmented generation."
    ]
    
    for query in queries:
        logger.info(f"Query: {query}")
        
        # Retrieve documents
        results = retriever.retrieve(query)
        
        print(f"\n--- Results for query: '{query}' ---")
        print(f"Retrieved {len(results)} documents\n")
        
        for i, doc in enumerate(results):
            source = doc.get("metadata", {}).get("source", "Unknown")
            title = doc.get("metadata", {}).get("title", "Untitled")
            score = doc.get("score", 0.0)
            source_type = doc.get("source", "Unknown")
            
            print(f"Document {i+1}: {title} (Source: {source})")
            print(f"Score: {score:.4f} | Type: {source_type}")
            print(f"Content: {doc.get('content', '')[:150]}...\n")
        
        print("---\n")

def main():
    """Main demonstration function"""
    logger.info("Starting Enhanced RAG demonstration")
    
    # Load environment variables
    load_dotenv()
    
    # Load sample documents
    documents = load_sample_documents()
    
    # Setup vector store
    vector_store = setup_vector_store(documents)
    if not vector_store:
        logger.error("Failed to set up vector store. Exiting.")
        return
    
    # Build knowledge graph
    logger.info("Building knowledge graph")
    kg_builder = EnhancedKnowledgeGraphBuilder()
    knowledge_graph = kg_builder.build_from_documents(documents)
    logger.info(f"Knowledge graph built with {knowledge_graph.number_of_nodes()} nodes and {knowledge_graph.number_of_edges()} edges")
    
    # Visualize the graph
    visualize_graph(knowledge_graph)
    
    # Demonstrate retrieval
    demo_retrieval(vector_store, knowledge_graph)
    
    logger.info("Enhanced RAG demonstration completed")

if __name__ == "__main__":
    main() 