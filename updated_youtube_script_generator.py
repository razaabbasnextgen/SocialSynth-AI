import streamlit as st
import os
import sys
import logging
import time
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import networkx as nx
import json

from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from advanced_rag_v2 import EnhancedKnowledgeGraph, ContextualReranker, visualize_knowledge_graph
from enhanced_rag_retriever import EnhancedRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("script_generator.log")
    ]
)
logger = logging.getLogger("script_generator")

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize embeddings
def get_embeddings():
    """Initialize embeddings with Google Generative AI"""
    try:
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize Google embeddings: {e}")
        raise

# Set up vector store
def setup_vector_store(embedding_function):
    """Set up or load the vector store"""
    try:
        return Chroma(
            collection_name="script_rag_store", 
            embedding_function=embedding_function, 
            persist_directory="./chroma_db",
            collection_metadata={"hnsw:space": "cosine"}
        )
    except Exception as e:
        logger.error(f"Error setting up vector store: {e}")
        return Chroma(
            collection_name="script_rag_store_new", 
            embedding_function=embedding_function, 
            persist_directory="./chroma_db",
            collection_metadata={"hnsw:space": "cosine"}
        )

# Create LLM
def create_llm(model_name="gemini-1.5-flash"):
    """Create LLM instance using Gemini Flash"""
    try:
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.7,
            convert_system_message_to_human=True
        )
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        # Fallback to another model if Flash isn't available
        try:
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                google_api_key=GOOGLE_API_KEY,
                temperature=0.7,
                convert_system_message_to_human=True
            )
        except:
            raise

def generate_youtube_script(topic: str, tone: str = "educational", length: str = "medium", include_sources: bool = True) -> str:
    """
    Generate a YouTube script based on a given topic.
    
    Args:
        topic: The topic for the YouTube script
        tone: The tone of the script (educational, entertaining, professional, conversational, dramatic)
        length: Length of the script (short, medium, long)
        include_sources: Whether to include sources in the script
        
    Returns:
        The generated script as a string
    """
    try:
        logger.info(f"Generating script for topic: {topic}, tone: {tone}, length: {length}")
        
        # Initialize components
        embedding_function = get_embeddings()
        vector_store = setup_vector_store(embedding_function)
        llm = create_llm()
        
        # Create knowledge graph and reranker
        knowledge_graph = EnhancedKnowledgeGraph(vector_store=vector_store)
        reranker = ContextualReranker(llm=llm)
        
        # Generate the script
        result = generate_script(
            query=topic,
            tone=tone,
            length=length,
            knowledge_graph=knowledge_graph,
            reranker=reranker,
            llm=llm
        )
        
        script = result["script"]
        
        # Optionally add sources
        if include_sources and result["documents"]:
            sources_text = "\n\nSOURCES:\n"
            for i, doc in enumerate(result["documents"], 1):
                source = doc.metadata.get("source", "Unknown")
                title = doc.metadata.get("title", "Untitled")
                url = doc.metadata.get("url", "No URL available")
                sources_text += f"{i}. [{source}] {title} - {url}\n"
            
            script += sources_text
        
        return script
    
    except Exception as e:
        logger.error(f"Error in generate_youtube_script: {e}")
        return f"Error generating script: {str(e)}"

# Generate script
def generate_script(query, tone, length, knowledge_graph, reranker, llm):
    """Generate a YouTube script based on retrieved information"""
    try:
        # Get relevant documents from knowledge graph
        relevant_docs = knowledge_graph.retrieve(query=query, k=8, use_hybrid=True)
        
        # Rerank documents
        reranked_docs = reranker.rerank(query, relevant_docs, top_k=8)
        
        # Format context from documents
        docs_context = "\n\n---\n\n".join([
            f"SOURCE ({doc.metadata.get('source', 'unknown')}): {doc.metadata.get('title', '')}\n\n"
            f"URL: {doc.metadata.get('url', 'N/A')}\n\n"
            f"RELEVANCE: {doc.metadata.get('relevance_score', 0.0):.2f}\n\n"
            f"MATCHED ENTITIES: {', '.join(doc.metadata.get('matched_entities', []))}\n\n"
            f"CONTENT: {doc.page_content}"
            for doc in reranked_docs
        ])
        
        # Determine script length in words
        if length == "short":
            word_count = "400-600"
            minutes = "3-5"
        elif length == "medium":
            word_count = "800-1200"
            minutes = "7-10" 
        else:  # long
            word_count = "1500-2000"
            minutes = "12-15"
        
        # System prompt for script generation
        system_prompt = f"""You are a professional YouTube script writer.

Generate a {tone} script for a YouTube video on the topic: "{query}"

The script should be approximately {word_count} words (for a {minutes} minute video).

Use ONLY the factual information provided in the context sources below. If information isn't available in the sources, acknowledge the limitation rather than making up facts.

Organize the script with:
1. Attention-grabbing hook (first 10 seconds)
2. Brief intro with what the video will cover
3. Main content (3-4 key sections with informative points)
4. Conclusion summarizing key takeaways
5. Call-to-action asking viewers to like, subscribe, and comment

Make the script engaging, factual, and ready to record.

TONE GUIDELINES:
- If educational: Clear explanations, facts, statistics
- If entertaining: More casual language, humor, relatability 
- If professional: Formal language, industry terminology, focused information

CONTEXT INFORMATION:
{docs_context}"""

        # Generate script
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Write a {tone} YouTube script about: {query}")
        ]
        
        response = llm.invoke(messages)
        
        return {
            "script": response.content,
            "documents": reranked_docs,
            "query": query,
            "tone": tone,
            "length": length
        }
    except Exception as e:
        logger.error(f"Error generating script: {e}")
        return {
            "script": f"Error generating script: {str(e)}",
            "documents": [],
            "query": query,
            "tone": tone,
            "length": length
        }

# Format knowledge graph for visualization
def format_knowledge_graph_for_viz(graph):
    """Format knowledge graph for visualization with additional attributes"""
    # Create a copy of the graph to visualize
    viz_graph = nx.Graph()
    
    # Add nodes with attributes
    for node, attrs in graph.nodes(data=True):
        node_type = attrs.get("type", "unknown")
        size = 15  # Default size
        color = "#A9A9A9"  # Default gray
        title = node  # Default title
        
        if node_type == "document":
            # Document node (blue)
            size = 20
            color = "#6495ED"  # Blue
            content = attrs.get("content", "")
            title = content[:150] + "..." if len(content) > 150 else content
            source = attrs.get("metadata", {}).get("source", "unknown")
            
            # Adjust size based on relevance score
            relevance_score = attrs.get("metadata", {}).get("relevance_score", 0.0)
            if relevance_score > 0:
                size = 15 + (relevance_score * 10)  # Size from 15-25 based on relevance
            
        elif node_type == "entity":
            # Entity node
            entity_type = attrs.get("entity_type", "")
            entity_name = attrs.get("name", "")
            title = f"{entity_name} ({entity_type})"
            
            # Color based on entity type
            if entity_type in ["PERSON", "ORG", "GPE"]:
                color = "#9370DB"  # Purple
            elif entity_type in ["DATE", "TIME", "MONEY", "PERCENT"]:
                color = "#3CB371"  # Green
            else:
                color = "#FFA500"  # Orange
        
        # Add the node with enhanced attributes
        viz_graph.add_node(
            node,
            type=node_type,
            size=size,
            color=color,
            title=title
        )
    
    # Add edges with enhanced attributes
    for source, target, attrs in graph.edges(data=True):
        relationship = attrs.get("relationship", "")
        weight = attrs.get("weight", 1.0)
        width = max(1, min(10, weight * 10))  # Scale weight to width
        
        if relationship == "contains":
            # Entity contained in document
            viz_graph.add_edge(
                source,
                target,
                title="contains",
                width=width,
                color="#888888"  # Gray
            )
        elif relationship == "similar_to":
            # Similar documents
            viz_graph.add_edge(
                source,
                target,
                title=f"similarity: {weight:.2f}",
                width=width,
                dashes=True,
                color="#0000FF"  # Blue
            )
    
    return viz_graph

# Main Streamlit app
def main():
    # Set page config
    st.set_page_config(
        page_title="Enhanced YouTube Script Generator",
        page_icon="🎬",
        layout="wide"
    )
    
    st.title("🎬 Enhanced YouTube Script Generator")
    st.markdown("Generate professional YouTube scripts with diverse, relevant real-time information using advanced RAG")
    
    # Initialize components
    try:
        with st.spinner("Loading models and components..."):
            # Initialize embeddings, vector store, LLMs
            embedding_function = get_embeddings()
            vector_store = setup_vector_store(embedding_function)
            llm = create_llm()
            
            # Initialize knowledge graph and reranker
            knowledge_graph = EnhancedKnowledgeGraph(embedding_function)
            reranker = ContextualReranker()
            
            # Initialize enhanced retriever
            enhanced_retriever = EnhancedRetriever(embedding_function)
        
        # User inputs
        with st.form("script_form"):
            query = st.text_input("Topic or Keyword:", placeholder="e.g., AI trends in 2025")
            
            col1, col2 = st.columns(2)
            
            with col1:
                tone = st.selectbox(
                    "Script Tone:",
                    options=["educational", "entertaining", "professional", "conversational", "inspiring"],
                    index=0
                )
            
            with col2:
                length = st.selectbox(
                    "Script Length:",
                    options=["short", "medium", "long"],
                    index=1
                )
            
            # Advanced options
            with st.expander("Advanced Options"):
                use_trends = st.checkbox("Use Google Trends to expand query", value=True)
                min_relevance = st.slider("Minimum relevance score", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
                data_sources = st.multiselect(
                    "Data Sources:",
                    options=["YouTube", "Blogs", "News"],
                    default=["YouTube", "Blogs", "News"]
                )
            
            submit_button = st.form_submit_button("Generate Script")
        
        # Process form submission
        if submit_button and query:
            with st.spinner("Collecting diverse, relevant information..."):
                # Determine source configuration
                youtube_count = 5 if "YouTube" in data_sources else 0
                blog_count = 5 if "Blogs" in data_sources else 0
                news_count = 5 if "News" in data_sources else 0
                
                # Use enhanced retriever to get relevant documents
                documents = enhanced_retriever.retrieve(
                    query=query,
                    use_trends=use_trends,
                    min_relevance_score=min_relevance
                )
                
                if not documents:
                    st.warning("No relevant information found. Try adjusting your query or lowering the relevance threshold.")
                else:
                    st.success(f"Collected {len(documents)} relevant sources")
                    
                    # Add documents to vector store and knowledge graph
                    vector_store.add_documents(documents)
                    knowledge_graph.add_documents(documents)
            
            with st.spinner("Generating your YouTube script..."):
                # Generate script
                result = generate_script(query, tone, length, knowledge_graph, reranker, llm)
                
                # Store in session state for later use
                st.session_state.result = result
            
            # Display script
            st.markdown("### 📝 Your YouTube Script")
            st.text_area("Script", value=result["script"], height=400)
            
            # Download options
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "Download Script (.txt)",
                    result["script"],
                    file_name=f"youtube_script_{query.replace(' ', '_')}.txt",
                    mime="text/plain"
                )
            
            # Visualization
            st.markdown("### 🧠 Knowledge Graph Visualization")
            with st.spinner("Generating knowledge graph..."):
                try:
                    # Create enhanced visualization
                    if knowledge_graph.visualize("knowledge_graph.html"):
                        with open("knowledge_graph.html", "r", encoding="utf-8") as f:
                            html_content = f.read()
                        st.components.v1.html(html_content, height=600)
                except Exception as e:
                    st.error(f"Error generating visualization: {e}")
            
            # Show sources with relevance and entity matching information
            st.markdown("### 📚 Sources Used")
            with st.expander("View Sources and Relevance"):
                # Group docs by source type
                sources_by_type = {}
                for doc in result["documents"]:
                    source_type = doc.metadata.get("source", "unknown")
                    if source_type not in sources_by_type:
                        sources_by_type[source_type] = []
                    sources_by_type[source_type].append(doc)
                
                # Show sources by type with relevance information
                for source_type, docs in sources_by_type.items():
                    st.subheader(f"{source_type.capitalize()} Sources")
                    for i, doc in enumerate(docs):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{doc.metadata.get('title', 'Unknown')}**")
                            st.markdown(f"*URL*: {doc.metadata.get('url', 'N/A')}")
                        with col2:
                            st.markdown(f"Relevance: **{doc.metadata.get('relevance_score', 0.0):.2f}**")
                            
                        # Show matched entities if available
                        matched_entities = doc.metadata.get('matched_entities', [])
                        if matched_entities:
                            st.markdown(f"*Matched Entities*: {', '.join(matched_entities)}")
                        
                        # Show content preview
                        st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                        st.divider()
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        logger.error(f"Application error: {e}", exc_info=True)

if __name__ == "__main__":
    main() 