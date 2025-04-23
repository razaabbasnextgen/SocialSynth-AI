import streamlit as st
import os
import sys
import logging
from dotenv import load_dotenv
from typing import List, Dict, Any
import time

from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from advanced_rag_v2 import EnhancedKnowledgeGraph, ContextualReranker, visualize_knowledge_graph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("rag_advanced.log")
    ]
)
logger = logging.getLogger("advanced_rag")

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def get_embeddings():
    """Initialize embeddings with Google Generative AI"""
    try:
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize Google embeddings: {e}")
        raise

def setup_vector_store(embedding_function):
    """Set up or load the vector store"""
    try:
        return Chroma(
            collection_name="rag_store", 
            embedding_function=embedding_function, 
            persist_directory="./chroma_db",
            collection_metadata={"hnsw:space": "cosine"}
        )
    except Exception as e:
        logger.error(f"Error setting up vector store: {e}")
        # Create new collection if loading fails
        return Chroma(
            collection_name="rag_store_new", 
            embedding_function=embedding_function, 
            persist_directory="./chroma_db",
            collection_metadata={"hnsw:space": "cosine"}
        )

def get_sample_documents():
    """Get some sample documents if needed for testing"""
    return [
        Document(
            page_content="Google Gemini is a multimodal AI model that can understand and generate text, code, images, and more.",
            metadata={"source": "sample", "type": "ai"}
        ),
        Document(
            page_content="Retrieval Augmented Generation (RAG) enhances LLM responses by incorporating external knowledge.",
            metadata={"source": "sample", "type": "ai"}
        ),
        Document(
            page_content="Knowledge graphs represent information as a network of entities and relationships.",
            metadata={"source": "sample", "type": "database"}
        )
    ]

def create_llm(model_name="gemini-1.5-pro"):
    """Create LLM instance"""
    try:
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.2,
            convert_system_message_to_human=True
        )
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise

def generate_response(llm, query: str, relevant_docs: List[Document]):
    """Generate response using RAG approach"""
    try:
        # Format documents for context
        docs_context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(relevant_docs)])
        
        # System prompt with RAG instructions
        system_prompt = """You are an AI assistant powered by an advanced Retrieval-Augmented Generation system.
You have access to relevant documents that have been retrieved based on the user's query.
Use these documents to provide accurate and comprehensive answers.
If the documents don't contain relevant information, acknowledge this and provide the best answer you can based on your knowledge.
Always cite specific documents when using information from them."""
        
        # Generate response using RAG context
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""Query: {query}
            
Retrieved Context:
{docs_context}

Please answer the query based on the retrieved context and your knowledge.""")
        ]
        
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"I apologize, but I encountered an error while generating a response: {str(e)}"

def main():
    st.set_page_config(
        page_title="Advanced RAG System",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Advanced RAG with Knowledge Graph")
    st.markdown("Enhanced retrieval using knowledge graphs, hybrid search, and query expansion")
    
    # Initialize components
    try:
        with st.spinner("Loading embedding model..."):
            embedding_function = get_embeddings()
        
        with st.spinner("Setting up vector store..."):
            vector_store = setup_vector_store(embedding_function)
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        
        with st.spinner("Initializing knowledge graph..."):
            knowledge_graph = EnhancedKnowledgeGraph(embedding_function)
            reranker = ContextualReranker()
            
            # Get existing documents from vector store
            existing_docs = vector_store.get()
            
            if existing_docs["documents"]:
                docs = [
                    Document(page_content=content, metadata=metadata)
                    for content, metadata in zip(existing_docs["documents"], existing_docs["metadatas"])
                ]
                knowledge_graph.add_documents(docs)
                st.sidebar.success(f"Loaded {len(docs)} documents into knowledge graph")
            else:
                # Add sample documents for testing
                sample_docs = get_sample_documents()
                vector_store.add_documents(sample_docs)
                knowledge_graph.add_documents(sample_docs)
                st.sidebar.info("Added sample documents for testing")
        
        with st.spinner("Initializing LLM..."):
            llm = create_llm()
        
        # Left sidebar for data input
        with st.sidebar:
            st.header("Add Documents")
            
            uploaded_file = st.file_uploader("Upload Document", type=["txt", "pdf", "md"])
            if uploaded_file is not None:
                text_content = uploaded_file.read().decode("utf-8")
                metadata = {"source": uploaded_file.name, "type": "upload"}
                new_doc = Document(page_content=text_content, metadata=metadata)
                
                if st.button("Add Document"):
                    with st.spinner("Adding document..."):
                        vector_store.add_documents([new_doc])
                        knowledge_graph.add_documents([new_doc])
                        st.success("Document added!")
            
            st.divider()
            
            st.header("Manual Input")
            text_input = st.text_area("Document Text")
            source_input = st.text_input("Source")
            
            if st.button("Add Text"):
                if text_input:
                    with st.spinner("Adding text..."):
                        metadata = {"source": source_input or "manual input", "type": "manual"}
                        new_doc = Document(page_content=text_input, metadata=metadata)
                        vector_store.add_documents([new_doc])
                        knowledge_graph.add_documents([new_doc])
                        st.success("Text added!")
        
        # Main query interface
        query = st.text_input("Enter your query:", placeholder="What would you like to know?")
        
        col1, col2 = st.columns(2)
        with col1:
            use_hybrid = st.checkbox("Use Hybrid Search", value=True)
            use_query_expansion = st.checkbox("Use Query Expansion", value=True)
        
        with col2:
            k_slider = st.slider("Number of documents to retrieve", min_value=1, max_value=10, value=5)
            visualize = st.checkbox("Visualize Knowledge Graph", value=False)
        
        if query and st.button("Search"):
            with st.spinner("Retrieving information..."):
                start_time = time.time()
                
                # Retrieve documents using knowledge graph
                relevant_docs = knowledge_graph.retrieve(
                    query=query, 
                    k=k_slider,
                    use_hybrid=use_hybrid
                )
                
                # Rerank documents
                reranked_docs = reranker.rerank(query, relevant_docs, top_k=k_slider)
                
                # Generate response
                response = generate_response(llm, query, reranked_docs)
                
                retrieval_time = time.time() - start_time
                
            # Display results
            st.markdown("### Answer")
            st.write(response)
            
            st.markdown(f"*Retrieved in {retrieval_time:.2f} seconds*")
            
            # Display retrieved documents
            with st.expander("Retrieved Documents"):
                for i, doc in enumerate(reranked_docs):
                    st.markdown(f"**Document {i+1}**")
                    st.text(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
                    st.markdown(f"Source: *{doc.metadata.get('source', 'Unknown')}*")
                    st.divider()
            
            # Visualize knowledge graph if requested
            if visualize:
                with st.spinner("Generating knowledge graph visualization..."):
                    if knowledge_graph.visualize("knowledge_graph.html"):
                        with open("knowledge_graph.html", "r", encoding="utf-8") as f:
                            html_content = f.read()
                        st.components.v1.html(html_content, height=700)
    
    except Exception as e:
        st.error(f"Error initializing the application: {str(e)}")
        logger.error(f"Application error: {e}", exc_info=True)

if __name__ == "__main__":
    main() 