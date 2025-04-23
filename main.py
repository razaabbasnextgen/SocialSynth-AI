import streamlit as st
import os
import sys
import time
import logging
import asyncio
import requests
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any, Optional
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda
from langchain_community.embeddings.openai import OpenAIEmbeddings  # Fallback embeddings
from langgraph.graph import StateGraph, END
from langchain.chat_models import init_chat_model  # For Vertex AI

from agents import get_youtube_top_videos, get_blog_articles
from trends import get_trending_keywords
from advanced_rag import (
    KnowledgeGraph,
    CrossEncoderReranker,
    visualize_knowledge_graph
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("script_generator")

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Fallback API key
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
CX = os.getenv("CX")

# Circuit breaker state
service_status = {
    "google_embeddings": {"failures": 0, "last_failure": 0, "disabled": False},
    "news_api": {"failures": 0, "last_failure": 0, "disabled": False},
    "youtube_api": {"failures": 0, "last_failure": 0, "disabled": False},
    "blog_api": {"failures": 0, "last_failure": 0, "disabled": False},
    "trends_api": {"failures": 0, "last_failure": 0, "disabled": False},
}

# Constants
MAX_FAILURES = 5  # Number of failures before triggering circuit breaker
COOLDOWN_PERIOD = 300  # 5 minutes in seconds

# Initialize embeddings with fallback
def get_embeddings():
    """Get embeddings model with fallback support"""
    try:
        if not service_status["google_embeddings"]["disabled"]:
            # Try Vertex AI first
            try:
                return init_chat_model("gemini-2.0-flash-001", model_provider="google_vertexai")
            except Exception as vertex_error:
                logger.warning(f"Vertex AI initialization failed: {vertex_error}, falling back to Google Generative AI")
            return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    except Exception as e:
        logger.error(f"Failed to initialize Google embeddings: {e}")
        update_service_status("google_embeddings")
    
    # Fallback to OpenAI embeddings if available
    if OPENAI_API_KEY:
        logger.info("Using OpenAI embeddings as fallback")
        return OpenAIEmbeddings()
    
    # Last resort - use Google embeddings even if previously failed
    logger.warning("Forcing Google embeddings as last resort")
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize embeddings and vector store
embedding = get_embeddings()
vectorstore = Chroma(
    collection_name="rag_store", 
    embedding_function=embedding, 
    persist_directory="./chroma_db",
    collection_metadata={"hnsw:space": "cosine"}  # Add required metadata
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Initialize advanced RAG components
knowledge_graph = KnowledgeGraph(embedding)
reranker = CrossEncoderReranker()

# Add a session state to store pre-generated graph data
if 'graph_data' not in st.session_state:
    st.session_state.graph_data = None
    st.session_state.graph_generated = False

# Set up event loop for async operations
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# App state definition
class AppState(TypedDict, total=False):
    text: str
    source: str
    query: str
    docs: List[Document]
    script: str
    tone: str
    format_type: str
    voice_tone: str
    script_length: int
    errors: List[str]

# Circuit breaker pattern
def update_service_status(service_name: str, success: bool = False):
    """Update service status for circuit breaker pattern"""
    if service_name not in service_status:
        return
    
    current_time = time.time()
    
    # Reset on success
    if success:
        service_status[service_name]["failures"] = 0
        service_status[service_name]["disabled"] = False
        return
    
    # Update on failure
    service_status[service_name]["failures"] += 1
    service_status[service_name]["last_failure"] = current_time
    
    # Check if we should disable the service
    if service_status[service_name]["failures"] >= MAX_FAILURES:
        service_status[service_name]["disabled"] = True
        logger.warning(f"Circuit breaker triggered for {service_name}")
    
    # Check for cooldown expiration
    if (service_status[service_name]["disabled"] and 
        current_time - service_status[service_name]["last_failure"] > COOLDOWN_PERIOD):
        logger.info(f"Cooldown period expired for {service_name}, re-enabling")
        service_status[service_name]["failures"] = 0
        service_status[service_name]["disabled"] = False

# News fetching with improved error handling
def fetch_news_headlines_sync(query: str, api_key: str) -> List[Dict[str, Any]]:
    """Fetch news headlines with error handling and circuit breaker"""
    # Check if service is disabled by circuit breaker
    if service_status["news_api"]["disabled"]:
        logger.info("News API is currently disabled by circuit breaker")
        return []
    
    try:
        url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=5&apiKey={api_key}"
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            logger.warning(f"NewsAPI returned status code {response.status_code}")
            update_service_status("news_api")
            return []
        
        data = response.json()
        if "articles" not in data:
            logger.warning("NewsAPI response missing 'articles' key")
            return []
        
        update_service_status("news_api", True)  # Success
        return data.get("articles", [])
    except requests.exceptions.Timeout:
        logger.warning("NewsAPI request timed out")
        update_service_status("news_api")
        return []
    except Exception as e:
        logger.error(f"NewsAPI Error: {e}")
        update_service_status("news_api")
        return []

# Async parallel fetching with individual error handling
async def fetch_data_parallel(query: str, api_key: str):
    """Fetch data from multiple sources in parallel with individual error handling"""
    loop = asyncio.get_event_loop()
    results = [[], [], [], []]  # Initialize with empty defaults
    
    async def safe_execute(index, func, *args):
        try:
            result = await loop.run_in_executor(executor, func, *args)
            results[index] = result
        except Exception as e:
            logger.error(f"Error in parallel fetch index {index}: {e}")
            results[index] = []  # Use empty default
    
    with ThreadPoolExecutor() as executor:
        tasks = [
            safe_execute(0, fetch_news_headlines_sync, query, api_key),
            safe_execute(1, get_youtube_top_videos, query),
            safe_execute(2, get_blog_articles, query, GOOGLE_API_KEY, CX),
            safe_execute(3, get_trending_keywords, query)
        ]
        await asyncio.gather(*tasks)
    
    return results

# Safe document addition with advanced retry and exponential backoff
def safe_add_docs(docs: List[Document], max_retries: int = 3, initial_batch_size: int = 5):
    """Add documents to vector store with retry logic and dynamic batch sizing"""
    valid = [doc for doc in docs if doc.page_content.strip()]
    if not valid:
        logger.info("No valid documents to add")
        return
    
    # Check if embeddings service is disabled
    if service_status["google_embeddings"]["disabled"]:
        logger.warning("Google embeddings are currently disabled by circuit breaker")
        # Try to add docs anyway with potentially fallback embeddings
    
    batch_size = initial_batch_size
    i = 0
    added_count = 0
    
    while i < len(valid):
        batch = valid[i:i+batch_size]
        success = False
        
        for attempt in range(1, max_retries + 1):
            try:
                # Add exponential backoff delay on retries
                if attempt > 1:
                    wait_time = 2 ** (attempt - 1)  # 1, 2, 4, 8... seconds
                    logger.info(f"Waiting {wait_time}s before retry {attempt}")
                    time.sleep(wait_time)
                
                # Add to vector store
                vectorstore.add_documents(batch)
                
                # Also add to knowledge graph
                try:
                    logger.info(f"Adding {len(batch)} documents to knowledge graph")
                    knowledge_graph.add_documents(batch)
                except Exception as kg_error:
                    logger.error(f"Error adding to knowledge graph: {kg_error}")
                
                added_count += len(batch)
                i += batch_size
                success = True
                update_service_status("google_embeddings", True)  # Success
                break
            except Exception as e:
                logger.error(f"[Embedding Error] Batch {i//batch_size+1} attempt {attempt}: {e}")
                update_service_status("google_embeddings")
        
        if not success:
            logger.warning(f"🚫 Skipped batch after {max_retries} attempts")
            # Reduce batch size for future batches
            batch_size = max(1, batch_size // 2)
            logger.info(f"Reducing batch size to {batch_size}")
            i += batch_size  # Skip this problematic batch
    
    logger.info(f"Successfully added {added_count} out of {len(valid)} documents")

@lru_cache(maxsize=100)
def cached_rewrite_query(query):
    """Cache query rewrites to reduce API calls"""
    prompt = f"Make this YouTube topic prompt clearer and more specific: {query}"
    try:
        # Try Vertex AI first
        try:
            model = init_chat_model("gemini-2.0-flash-001", model_provider="google_vertexai")
            response = model.invoke(prompt).content.strip()
            return response
        except Exception as vertex_error:
            logger.warning(f"Vertex AI query rewrite failed: {vertex_error}, falling back to Google Generative AI")
        model = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", request_timeout=30)
        response = model.invoke(prompt).content.strip()
        return response
    except Exception as e:
        logger.error(f"Query rewrite failed: {e}")
        return query  # Return original query on failure

def graph_based_retrieve(state: AppState) -> AppState:
    """Graph-based retrieval function to replace or complement hybrid_retrieve"""
    query = state["query"]
    errors = state.get("errors", [])
    
    try:
        # Use the knowledge graph for retrieval
        graph_docs = knowledge_graph.retrieve(query, k=5)
        
        # Fall back to vector retrieval if graph retrieval returns nothing
        if not graph_docs:
            try:
                sem_docs = retriever.get_relevant_documents(query)
                return {"query": query, "docs": sem_docs, "errors": errors}
            except Exception as e:
                errors.append(f"Fallback retrieval error: {str(e)}")
                return {"query": query, "docs": [], "errors": errors}
        
        return {"query": query, "docs": graph_docs, "errors": errors}
    except Exception as e:
        errors.append(f"Graph retrieval error: {str(e)}")
        
        # Fall back to standard retrieval
        try:
            sem_docs = retriever.get_relevant_documents(query)
            return {"query": query, "docs": sem_docs, "errors": errors}
        except Exception as e2:
            errors.append(f"Fallback retrieval error: {str(e2)}")
            return {"query": query, "docs": [], "errors": errors}

def compress_docs(state: AppState) -> AppState:
    """Compress and extract relevant information from documents"""
    query = state["query"]
    docs = state.get("docs", [])
    errors = state.get("errors", [])
    
    if not docs:
        logger.warning("No documents to compress")
        return state
    
    try:
        model = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", request_timeout=45)
        # Limit total content to avoid token limits
        combined = "\n\n".join([doc.page_content[:800] for doc in docs[:10]])
        prompt = f"""Extract the most relevant information from these documents 
        for creating a YouTube script about: '{query}'
        
        Focus on:
        - Key facts and statistics
        - Interesting angles or perspectives
        - Current trends or developments
        
        Documents:
        {combined}
        """
        
        result = model.invoke(prompt).content.strip()
        return {
            "query": query, 
            "docs": [Document(page_content=result)], 
            "errors": errors
        }
    except Exception as e:
        logger.error(f"Document compression failed: {e}")
        errors.append(f"Compression error: {str(e)}")
        # Fall back to using original docs but limit total content
        fallback_docs = [
            Document(page_content=doc.page_content[:500], metadata=doc.metadata) 
            for doc in docs[:5]
        ]
        return {"query": query, "docs": fallback_docs, "errors": errors}

def generate_script(query, num_docs=5, length="medium", tone="balanced", model="claude-3-opus-20240229"):
    """Generate a script based on user query and parameters"""
    try:
        logger.info(f"Generating script with query: {query}, docs: {num_docs}, length: {length}, tone: {tone}")
        
        # Mark that we're generating a script (for session state tracking)
        st.session_state.script_generated = False
        st.session_state.graph_generated = False
        st.session_state.graph_data = None
        
        # Get context from documents
        docs = fetch_relevant_docs(query, num_docs)
        
        if not docs:
            logger.warning("No relevant documents found for query")
            return "I couldn't find relevant information to generate a script. Please try a different query or check your document database."
        
        # Generate the script using the context
        script = run_script_gen_chain(docs, query, length, tone, model)
        
        # Set session state to indicate script was generated
        st.session_state.script_generated = True
        
        # Pre-generate knowledge graph data to be ready for visualization
        try:
            # Update Streamlit session state with graph data
            graph_data = knowledge_graph.visualize_graph()
            if graph_data and graph_data.get('nodes') and len(graph_data.get('nodes', [])) > 0:
                st.session_state.graph_data = graph_data
                st.session_state.graph_generated = True
                logger.info(f"Knowledge graph data pre-generated successfully with {len(graph_data.get('nodes', []))} nodes")
            else:
                logger.warning("Knowledge graph data generated but contains no nodes")
                st.session_state.graph_generated = False
        except Exception as kg_error:
            logger.error(f"Error pre-generating knowledge graph: {kg_error}")
            st.session_state.graph_data = None
            st.session_state.graph_generated = False
        
        return script
    
    except Exception as e:
        logger.error(f"Error generating script: {e}")
        return f"An error occurred while generating the script: {str(e)}"

def add_document(state: AppState) -> AppState:
    """Process any documents added by the user"""
    return state

def rerank_docs(state: AppState) -> AppState:
    """Rerank documents by relevance"""
    query = state["query"]
    docs = state.get("docs", [])
    errors = state.get("errors", [])
    
    if not docs:
        return state
    
    try:
        # Simple reranking - move docs with query terms in them to the top
        def score_doc(doc):
            content = doc.page_content.lower()
            query_terms = query.lower().split()
            score = sum(10 if term in content else 0 for term in query_terms)
            
            # Add bonus for docs from certain sources
            source = doc.metadata.get("source", "").lower()
            if "youtube" in source:
                score += 5  # Prioritize YouTube content
            
            return score
        
        reranked = sorted(docs, key=score_doc, reverse=True)
        return {"query": query, "docs": reranked, "errors": errors}
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        return state  # Fall back to original order on failure

# Build the LangGraph processing pipeline
builder = StateGraph(AppState)
builder.add_node("REWRITE", RunnableLambda(lambda state: {
    "query": cached_rewrite_query(state["query"]), 
    **state,
    "errors": state.get("errors", [])
}))
builder.add_node("ADD", RunnableLambda(add_document))
builder.add_node("HYBRID", RunnableLambda(graph_based_retrieve))
builder.add_node("RERANK", RunnableLambda(rerank_docs))
builder.add_node("COMPRESS", RunnableLambda(compress_docs))
builder.add_node("GENERATE", RunnableLambda(generate_script))

# Set up the processing flow
builder.set_entry_point("REWRITE")
builder.add_edge("REWRITE", "ADD")
builder.add_edge("ADD", "HYBRID")
builder.add_edge("HYBRID", "RERANK")
builder.add_edge("RERANK", "COMPRESS")
builder.add_edge("COMPRESS", "GENERATE")
builder.add_edge("GENERATE", END)

# Compile the graph
app = builder.compile()

# Streamlit UI
def main():
    # Set page config with proper favicon
    st.set_page_config(
        page_title="SocialSynth-AI",
        page_icon="logo.png",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Create a dark mode professional UI
    st.markdown("""
    <style>
    /* Reset and base styles */
    body {
        font-family: 'Inter', sans-serif;
        background-color: #111827 !important;
        color: #e2e8f0;
        margin: 0;
        padding: 0;
    }
    
    /* Override Streamlit's default white background */
    .css-18e3th9 {
        background-color: #111827 !important;
    }
    
    .css-1d391kg {
        background-color: #111827 !important;
    }
    
    /* Main container styles */
    .main .block-container {
        padding-top: 1rem;
        max-width: 1200px;
        margin: 0 auto;
        background-color: #111827 !important;
    }
    
    /* Modern header with gradient */
    .header-container {
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -2px rgba(0, 0, 0, 0.15);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .logo-section {
        display: flex;
        align-items: center;
    }
    
    .logo-image {
        height: 50px;
        margin-right: 15px;
    }
    
    .header-title {
        color: white;
        font-weight: 800;
        font-size: 2rem;
        margin: 0;
        letter-spacing: -0.025em;
    }
    
    .header-subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        font-weight: 400;
    }
    
    /* Form styling */
    .stForm {
        background-color: #1f2937 !important;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2), 0 2px 4px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        border: 1px solid #374151;
    }
    
    .stForm [data-testid="stForm"] {
        border: none;
        padding: 0;
    }
    
    /* Input styling */
    [data-testid="stTextInput"] input {
        border-radius: 0.5rem;
        border: 1px solid #374151;
        padding: 0.75rem 1rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        background-color: #1e293b;
        color: #e2e8f0;
    }
    
    [data-testid="stTextInput"] input:focus {
        border-color: #4F46E5;
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.2);
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(90deg, #4F46E5 0%, #6366F1 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border-radius: 0.5rem;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(99, 102, 241, 0.4);
        transition: all 0.2s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 10px -1px rgba(99, 102, 241, 0.5);
    }
    
    /* Download button */
    .download-btn {
        background: linear-gradient(90deg, #7C3AED 0%, #9333EA 100%) !important;
        box-shadow: 0 4px 6px -1px rgba(124, 58, 237, 0.4) !important;
    }
    
    .download-btn:hover {
        box-shadow: 0 6px 10px -1px rgba(124, 58, 237, 0.5) !important;
    }
    
    /* Custom button */
    .custom-button {
        display: inline-block;
        background: linear-gradient(90deg, #4F46E5 0%, #6366F1 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border-radius: 0.5rem;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(99, 102, 241, 0.4);
        transition: all 0.2s ease;
        text-align: center;
        cursor: pointer;
        width: 100%;
    }
    
    .custom-button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 10px -1px rgba(99, 102, 241, 0.5);
    }
    
    /* Select box styling */
    [data-testid="stSelectbox"] > div > div {
        border-radius: 0.5rem;
        border: 1px solid #374151;
        background-color: #1e293b;
        color: #e2e8f0;
    }
    
    /* Radio button styling */
    .stRadio [data-testid="stRadio"] {
        padding: 1rem;
        background-color: #1f2937;
        border-radius: 0.5rem;
        color: #e2e8f0;
    }
    
    /* Textarea styling */
    textarea {
        border-radius: 0.5rem !important;
        border: 1px solid #374151 !important;
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #4F46E5 0%, #6366F1 100%);
        border-radius: 1rem;
    }
    
    /* Section headers */
    h2, h3 {
        color: #e2e8f0;
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #e2e8f0;
        background-color: #1f2937;
        border-radius: 0.5rem;
    }
    
    /* Success message */
    .success-message {
        background-color: #064e3b;
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
        color: #ecfdf5;
    }
    
    .success-icon {
        color: #10b981;
        font-size: 1.2rem;
        margin-right: 0.75rem;
    }
    
    /* Script output container */
    .script-container {
        background-color: #1f2937;
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        border: 1px solid #374151;
    }
    
    /* Slider styling */
    [data-testid="stSlider"] {
        padding: 1rem 0;
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        background-color: #1f2937;
        border-radius: 0.5rem;
        padding: 1rem;
        border: 1px dashed #4F46E5;
    }
    
    /* Checkbox styling */
    [data-testid="stCheckbox"] {
        color: #e2e8f0;
    }
    
    /* Info box */
    .stAlert {
        background-color: #172554;
        color: #e2e8f0;
    }
    
    /* Metric styling */
    [data-testid="stMetric"] {
        background-color: #1f2937;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #374151;
    }
    
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
    }
    
    /* Make the tab button more visible */
    button[data-baseweb="tab"] {
        background-color: #1f2937 !important;
        border: 1px solid #374151 !important;
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
        margin-right: 5px !important;
        border-radius: 8px 8px 0 0 !important;
    }
    
    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(90deg, #4F46E5 0%, #6366F1 100%) !important;
        color: white !important;
        border: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Custom header with actual logo file
    st.markdown("""
    <div class="header-container">
        <div class="logo-section">
            <img src="./logo.png" class="logo-image">
            <div>
                <h1 class="header-title">SocialSynth-AI</h1>
                <p class="header-subtitle">Professional Social Media Content Generator</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for content generator and knowledge base visualization
    tab1, tab2 = st.tabs(["Content Generator", "Knowledge Base Visualization"])
    
    # Content Generator Tab
    with tab1:
        # Check service status and display warnings if needed
        disabled_services = [svc for svc, status in service_status.items() if status["disabled"]]
        if disabled_services:
            st.warning(f"⚠️ Some services are currently unavailable: {', '.join(disabled_services)}. Functionality may be limited.")
        
        with st.form("generator"):
            col1, col2 = st.columns(2)
            
            with col1:
                query = st.text_input("Enter your content topic:", 
                                 placeholder="e.g., Latest AI advancements in healthcare")
                uploaded_files = st.file_uploader("Upload reference materials:", 
                                                 accept_multiple_files=True,
                                                 help="PDF, TXT, and DOCX files supported")
            
            with col2:
                tone = st.selectbox("Content Tone", 
                                  ["Cinematic", "Educational", "Dystopian", "Emotional", 
                                   "Satirical", "Neutral", "Enthusiastic", "Professional"])
                format_type = st.radio("Script Format", 
                                     ["Scene-by-scene", "Dialogue", "Monologue", "Tutorial"])
                voice_tone = st.selectbox("Voice Tone", 
                                        ["Calm", "Energetic", "Serious", "Casual", 
                                         "Narrative", "Conversational", "Authoritative"])
            
            # Data source options
            st.markdown("<h3>Data Sources</h3>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                use_news = st.checkbox("Include News", value=True)
            with col2:
                use_youtube = st.checkbox("Include YouTube Data", value=True)
            with col3:
                use_trends = st.checkbox("Include Trends", value=True)
            
            # Advanced options in expander
            with st.expander("Advanced Options"):
                include_citations = st.checkbox("Include Source Citations", value=False)
                script_length = st.slider("Target Script Length (words)", 
                                         min_value=300, max_value=2000, value=800, step=100)
            
            submitted = st.form_submit_button("Generate Professional Script")
        
        if submitted and query.strip():
            with st.spinner("Researching and generating your script..."):
                progress_bar = st.progress(0)
                
                # Process uploaded files
                if uploaded_files:
                    st.info(f"Processing {len(uploaded_files)} uploaded files...")
                    for file in uploaded_files:
                        try:
                            content = file.read().decode("utf-8")
                            safe_add_docs([Document(
                                page_content=content, 
                                metadata={"source": f"upload:{file.name}"}
                            )])
                        except UnicodeDecodeError:
                            st.error(f"Could not process file {file.name} - not a valid text file")
                        except Exception as e:
                            st.error(f"Error processing file {file.name}: {e}")
                
                progress_bar.progress(20)
                
                # Fetch external data with proper error handling
                try:
                    # Fix for asyncio on Windows
                    if sys.platform == 'win32':
                        # Use WindowsSelectorEventLoopPolicy explicitly
                        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
                        
                        # Create and set the event loop
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        # Fetch external data with the loop
                        st.info("Gathering data from external sources...")
                        news_docs, yt_docs, blog_docs, trend_keywords = loop.run_until_complete(
                            fetch_data_parallel(query, NEWSAPI_KEY)
                        )
                        
                        # Close the loop
                        loop.close()
                    else:
                        # For non-Windows platforms
                        st.info("Gathering data from external sources...")
                        news_docs, yt_docs, blog_docs, trend_keywords = asyncio.run(
                            fetch_data_parallel(query, NEWSAPI_KEY)
                        )
                except Exception as e:
                    st.error(f"Error fetching external data: {e}")
                    news_docs, yt_docs, blog_docs, trend_keywords = [], [], [], []
                
                # Add YouTube videos
                if use_youtube and yt_docs:
                    st.info(f"Adding {len(yt_docs)} YouTube videos to knowledge base...")
                    try:
                        safe_add_docs(yt_docs)
                    except Exception as e:
                        st.error(f"Error adding YouTube data: {e}")
                
                progress_bar.progress(40)
                
                # Add documents from various sources to vector store
                if use_news and news_docs:
                    st.info(f"Adding {len(news_docs)} news articles to knowledge base...")
                    try:
                        safe_add_docs([Document(
                            page_content=f"{a.get('title')}\n{a.get('description')}", 
                            metadata={"source": a.get("url")}
                        ) for a in news_docs])
                    except Exception as e:
                        st.error(f"Error adding news: {e}")
                
                # Add blog content
                try:
                    safe_add_docs(blog_docs)
                except Exception as e:
                    st.error(f"Error adding blog data: {e}")
                
                progress_bar.progress(60)
                
                # Add trend data
                if use_trends and trend_keywords:
                    st.info(f"Found related trending keywords: {', '.join(trend_keywords)}")
                    trend_text = "\nTrending Keywords:\n" + "\n".join(trend_keywords)
                    try:
                        safe_add_docs([Document(
                            page_content=trend_text, 
                            metadata={"source": "trends"}
                        )])
                    except Exception as e:
                        st.error(f"Error adding trends: {e}")
                
                progress_bar.progress(80)
                
                # Generate the script
                st.info("Generating your script...")
                try:
                    result = app.invoke({
                        "query": query, 
                        "tone": tone, 
                        "format_type": format_type, 
                        "voice_tone": voice_tone,
                        "script_length": script_length,
                        "errors": []
                    })
                except Exception as e:
                    st.error(f"Error generating script: {e}")
                    result = {"script": f"Failed to generate script: {e}", "errors": [str(e)]}
                
                progress_bar.progress(100)
                
                # Display results
                if result.get("errors"):
                    st.warning("Completed with some issues:")
                    for error in result["errors"]:
                        st.error(error)
                
                st.markdown("<h2>Your Generated Script</h2>", unsafe_allow_html=True)
                
                # Store the script in a variable for easier handling
                script_text = result.get("script", "Script generation failed.")
                
                # Use a styled container for the script
                st.markdown('<div class="script-container">', unsafe_allow_html=True)
                
                # Display the script in a text area
                script_area = st.text_area("", script_text, height=600)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Add buttons for copy and download
                col1, col2 = st.columns(2)
                
                with col1:
                    # Create a fixed copy to clipboard button with direct JavaScript
                    st.markdown("""
                    <div id="copy-success" class="success-message" style="display:none;">
                      <div class="success-icon">✓</div>
                      <div>Script copied to clipboard successfully!</div>
                    </div>
                    
                    <button id="copy-button" class="custom-button" onclick="copyToClipboard()">
                        📋 Copy to Clipboard
                    </button>
                    
                    <script>
                    function copyToClipboard() {
                        const textArea = document.querySelector('textarea');
                        textArea.select();
                        document.execCommand('copy');
                        
                        // Show success message
                        document.getElementById('copy-success').style.display = 'flex';
                        setTimeout(function() {
                            document.getElementById('copy-success').style.display = 'none';
                        }, 3000);
                    }
                    </script>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Use Streamlit's download button with improved styling
                    if script_text:
                        download_filename = f"SocialSynth_script_{query[:20].replace(' ', '_')}.txt" if query else "SocialSynth_script.txt"
                        st.download_button(
                            label="💾 Download Script",
                            data=script_text,
                            file_name=download_filename,
                            mime="text/plain",
                            key="download_btn"
                        )
                
                # Analytics and metadata
                with st.expander("Script Analytics"):
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    
                    word_count = len(script_text.split())
                    minutes = word_count // 150
                    seconds = (word_count % 150) * 60 // 150
                    
                    with metrics_col1:
                        st.metric("Word Count", f"{word_count}")
                    with metrics_col2:
                        st.metric("Estimated Duration", f"{minutes} min {seconds} sec")
                    with metrics_col3:
                        st.metric("Tone", tone)
                    
                    st.markdown(f"""
                    <div style="background-color: #1e293b; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem; border: 1px solid #374151;">
                        <h4 style="margin-top: 0; color: #e2e8f0;">Script Details</h4>
                        <p style="color: #e2e8f0;"><strong>Format:</strong> {format_type}</p>
                        <p style="color: #e2e8f0;"><strong>Voice:</strong> {voice_tone}</p>
                        <p style="color: #e2e8f0;"><strong>Target Length:</strong> {script_length} words</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Knowledge Base Visualization Tab
    with tab2:
        st.markdown("<h2>Knowledge Base Visualization</h2>", unsafe_allow_html=True)
        st.info("The knowledge graph shows connections between concepts in your content database.")
        
        # Display badge if graph data is already generated
        if ('graph_generated' in st.session_state and st.session_state.graph_generated and 
            'graph_data' in st.session_state and st.session_state.graph_data):
            st.success("✅ Knowledge graph data is ready to view")
        
        # Create a prominent button for visualization
        if st.button("View Knowledge Graph", type="primary"):
            try:
                # Use pre-generated graph data if available, otherwise generate it
                if ('graph_data' in st.session_state and st.session_state.graph_data and 
                    st.session_state.graph_data.get('nodes') and len(st.session_state.graph_data.get('nodes', [])) > 0):
                    graph_data = st.session_state.graph_data
                    st.info("Displaying pre-generated knowledge graph")
                else:
                    with st.spinner("Generating knowledge graph data..."):
                        graph_data = knowledge_graph.visualize_graph()
                        if graph_data and graph_data.get('nodes') and len(graph_data.get('nodes', [])) > 0:
                            st.session_state.graph_data = graph_data
                            st.session_state.graph_generated = True
                        else:
                            st.warning("No knowledge graph data available. Try generating a script first.")
                            st.stop()
                
                # Create metrics for the graph stats
                graph_col1, graph_col2 = st.columns(2)
                with graph_col1:
                    st.metric("Nodes", f"{len(graph_data['nodes'])}")
                with graph_col2:
                    st.metric("Connections", f"{len(graph_data['edges'])}")
                
                # Show top entities in a styled container
                entity_nodes = [n for n in graph_data['nodes'] if n['type'] == 'entity']
                if entity_nodes:
                    st.markdown("""
                    <div style="background-color: #1f2937; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; border: 1px solid #374151;">
                        <h4 style="margin-top: 0; color: #e2e8f0;">Top Entities</h4>
                        <div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">
                    """, unsafe_allow_html=True)
                    
                    # Display entities as pills/tags
                    entity_html = ""
                    for entity in entity_nodes[:15]:  # Show top 15
                        entity_html += f"""
                        <div style="background: linear-gradient(90deg, #4338CA 0%, #5B21B6 100%); 
                                   padding: 0.5rem 1rem; border-radius: 2rem; color: white; 
                                   font-size: 0.9rem; display: inline-block;">
                            {entity.get('label', '')}
                        </div>"""
                    
                    st.markdown(entity_html + "</div></div>", unsafe_allow_html=True)
                
                # Create a more advanced visualization using pyvis with improved error handling
                st.markdown("<h3>Interactive Knowledge Graph</h3>", unsafe_allow_html=True)
                with st.spinner("Generating interactive graph visualization..."):
                    try:
                        # Use our improved visualization function
                        graph_path = visualize_knowledge_graph(knowledge_graph)
                        
                        # Add a safety check to ensure the HTML exists
                        import os
                        if os.path.exists(graph_path):
                            with open(graph_path, 'r') as f:
                                html_content = f.read()
                                
                            # Use iframe instead of components.v1.html for better isolation
                            st.markdown(f"""
                            <iframe srcdoc='{html_content}' width='100%' height='700px' 
                            style='border: none; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);'></iframe>
                            """, unsafe_allow_html=True)
                        else:
                            st.error(f"Visualization file was not created properly: {graph_path}")
                    except Exception as viz_error:
                        st.error(f"Error displaying the visualization: {viz_error}")
                        st.warning("Falling back to simple entity display")
                        
                        # Create a simple fallback visualization
                        st.markdown("<h4>Entity Connections (Fallback View)</h4>", unsafe_allow_html=True)
                        for entity in entity_nodes[:10]:
                            st.markdown(f"**{entity.get('label', '')}** - Connected to {len([e for e in graph_data['edges'] if e['source'] == entity['id'] or e['target'] == entity['id']])} items")
                
                if len(graph_data['nodes']) < 5:
                    st.warning("Your knowledge graph is quite small. Generate scripts to populate it with more data.")
            except Exception as e:
                st.error(f"Error generating knowledge graph: {e}")
                st.info("The knowledge graph may be empty. Try generating a script first to populate it with data.")
                
        # Add some helpful information about the graph
        with st.expander("About Knowledge Graphs"):
            st.markdown("""
            A knowledge graph visualizes connections between concepts in your content. Nodes represent:
            
            - **Documents**: Content pieces in your database (blue)
            - **Entities**: Key concepts, people, places or things (orange)
            
            The connections show relationships between entities and documents.
            
            **How to use it**:
            1. Generate scripts to populate your knowledge base
            2. Click "View Knowledge Graph" to see connections
            3. Explore relationships to identify content patterns
            """)

# Ensure we have a function to generate a more reliable visualization
def visualize_knowledge_graph(knowledge_graph, height=700):
    """Generate a more reliable visualization of the knowledge graph"""
    try:
        graph_data = knowledge_graph.visualize_graph()
        
        # Create a NetworkX graph
        import networkx as nx
        from pyvis.network import Network
        import tempfile
        import os
        
        G = nx.Graph()
        
        # Add nodes
        for node in graph_data['nodes']:
            node_type = node.get('type', 'unknown')
            label = node.get('label', 'Unnamed')
            
            # Use different colors for different node types
            if node_type == 'entity':
                color = '#8B5CF6'  # Purple for entities
            elif node_type == 'document':
                color = '#3B82F6'  # Blue for documents
            else:
                color = '#9CA3AF'  # Gray for unknown
                
            G.add_node(node['id'], label=label, title=label, color=color, shape='dot', size=10)
        
        # Add edges
        for edge in graph_data['edges']:
            G.add_edge(edge['source'], edge['target'], title=edge.get('label', ''), width=1)
        
        # Create pyvis network
        net = Network(height=f"{height}px", width="100%", bgcolor="#FFFFFF", font_color="#121212")
        
        # Set options for better visualization
        net.set_options("""
        {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {
              "enabled": true,
              "iterations": 1000
            }
          },
          "interaction": {
            "navigationButtons": true,
            "keyboard": true
          }
        }
        """)
        
        # Add the NetworkX graph to the pyvis network
        net.from_nx(G)
        
        # Generate HTML file
        temp_dir = tempfile.gettempdir()
        html_path = os.path.join(temp_dir, "knowledge_graph.html")
        net.save_graph(html_path)
        
        # Add custom CSS to ensure the graph is visible
        with open(html_path, 'r') as f:
            html_content = f.read()
        
        # Add custom CSS to ensure better visibility
        custom_css = """
        <style>
        #mynetwork {
            width: 100% !important;
            height: 700px !important;
            border: 1px solid #ddd;
            background-color: #f8fafc;
        }
        </style>
        """
        
        html_content = html_content.replace('</head>', f'{custom_css}</head>')
        
        with open(html_path, 'w') as f:
            f.write(html_content)
            
        return html_path
    except Exception as e:
        import traceback
        st.error(f"Error generating visualization: {e}")
        st.code(traceback.format_exc())
        
        # Return a fallback HTML with error details
        fallback_html = f"""
        <html>
        <body>
            <div style="padding: 20px; background-color: #fee2e2; border: 1px solid #ef4444; border-radius: 8px;">
                <h3>Error Generating Knowledge Graph</h3>
                <p>{str(e)}</p>
                <pre>{traceback.format_exc()}</pre>
            </div>
        </body>
        </html>
        """
        
        temp_dir = tempfile.gettempdir()
        fallback_path = os.path.join(temp_dir, "knowledge_graph_error.html")
        with open(fallback_path, 'w') as f:
            f.write(fallback_html)
            
        return fallback_path

if __name__ == "__main__":
    main()
