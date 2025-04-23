"""
SocialSynth-AI Streamlit Application

This module contains the Streamlit UI implementation for SocialSynth-AI.
"""

import os
import time
import logging
import webbrowser
from typing import List, Dict, Any
import io
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Add custom run_in_executor to avoid the import error
async def run_in_executor(executor, func, *args, **kwargs):
    """
    Run a function in an executor (thread pool).
    This is a utility function to replace the missing run_in_executor from langchain.
    
    Args:
        executor: The executor to use
        func: The function to run
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        The result of the function
    """
    loop = asyncio.get_event_loop()
    if kwargs:
        return await loop.run_in_executor(
            executor, lambda: func(*args, **kwargs)
        )
    else:
        return await loop.run_in_executor(executor, func, *args)

# Third-party imports
import streamlit as st
import networkx as nx
import pandas as pd

# Setup for dotenv
from dotenv import load_dotenv
load_dotenv()  # Ensure .env is loaded at application start

# Define extract_text_from_file function at the very top level
def extract_text_from_file(file_path: str) -> str:
    """
    Extract text from various file formats.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Extracted text content
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
                
        elif file_extension == '.pdf':
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page_num in range(len(pdf_reader.pages)):
                        text += pdf_reader.pages[page_num].extract_text() + "\n"
                    return text
            except ImportError:
                logger.warning("PyPDF2 not installed. Cannot extract text from PDF.")
                return f"[PDF content from {os.path.basename(file_path)} - install PyPDF2 for full extraction]"
                
        elif file_extension == '.docx':
            try:
                import docx
                doc = docx.Document(file_path)
                return "\n".join([para.text for para in doc.paragraphs])
            except ImportError:
                logger.warning("python-docx not installed. Cannot extract text from DOCX.")
                return f"[DOCX content from {os.path.basename(file_path)} - install python-docx for full extraction]"
        else:
            return f"[Unsupported file format: {file_extension}]"
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {e}")
        return f"[Error extracting content from {os.path.basename(file_path)}]"

# LangChain imports
try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document

# Import from other modules - try both absolute and relative imports
try:
    # Try relative imports first (when used as part of package)
    try:
        from ..retrieval.enhanced_retriever import EnhancedRetriever
        from ..knowledge_graph.enhanced_builder import EnhancedKnowledgeGraphBuilder
        from ..generation.script_generator import ScriptGenerator, generate_content_script
        from ..utils.api_keys import load_api_keys
        from ..retrieval.enhanced_retriever_adapter import RetrieverAdapter
        from ..knowledge_graph.enhanced_knowledge_graph_adapter import KnowledgeGraphBuilderAdapter, EnhancedKnowledgeGraphAdapter
        from ..documentation.pdf_generator import create_documentation
        
        logger = logging.getLogger("socialsynth.ui")
        logger.info("Loaded modules using relative imports")
    except ImportError as e:
        logger.error(f"Error importing modules with relative paths: {str(e)}")
        raise
except:
    # Fall back to absolute imports (when script is run directly)
    try:
        try:
            from src.socialsynth.retrieval.enhanced_retriever import EnhancedRetriever
            from src.socialsynth.knowledge_graph.enhanced_builder import EnhancedKnowledgeGraphBuilder
            from src.socialsynth.generation.script_generator import ScriptGenerator, generate_content_script
            from src.socialsynth.utils.api_keys import load_api_keys
            from src.socialsynth.retrieval.enhanced_retriever_adapter import RetrieverAdapter
            from src.socialsynth.knowledge_graph.enhanced_knowledge_graph_adapter import KnowledgeGraphBuilderAdapter, EnhancedKnowledgeGraphAdapter
            from src.socialsynth.documentation.pdf_generator import create_documentation
            
            logger = logging.getLogger("socialsynth.ui")
            logger.info("Loaded modules using absolute imports")
        except ImportError as e:
            logger.error(f"Error importing modules with absolute paths: {str(e)}")
            raise
    except:
        # As a last resort, try to import assuming script is run from project root
        try:
            logger = logging.getLogger("socialsynth.ui")
            logger.info("Attempting to import modules by adjusting sys.path")
            
            import sys
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
            
            # Import core modules first
            try:
                from src.socialsynth.utils.api_keys import load_api_keys
                logger.info("API key loading module loaded successfully")
            except ImportError as e:
                logger.error(f"Error importing api_keys module: {str(e)}")
                
            # Try to import adapter modules
            try:
                from src.socialsynth.knowledge_graph.enhanced_knowledge_graph_adapter import KnowledgeGraphBuilderAdapter, EnhancedKnowledgeGraphAdapter
                logger.info("Adapter modules loaded successfully")
            except ImportError as e:
                logger.error(f"Error importing knowledge graph adapter: {str(e)}")
            
            # Try to import retrieval and knowledge graph modules separately
            try:
                from src.socialsynth.retrieval.enhanced_retriever import EnhancedRetriever
                from src.socialsynth.retrieval.enhanced_retriever_adapter import RetrieverAdapter
                from src.socialsynth.knowledge_graph.enhanced_builder import EnhancedKnowledgeGraphBuilder
                logger.info("Retrieval and knowledge graph modules loaded successfully")
            except ImportError as e:
                logger.error(f"Error importing retrieval modules: {str(e)}")
                
            # Try to import generation modules
            try:
                from src.socialsynth.generation.script_generator import ScriptGenerator, generate_content_script
                from src.socialsynth.documentation.pdf_generator import create_documentation
                logger.info("Google Generative AI (Gemini) module loaded successfully")
            except ImportError as e:
                logger.error(f"Error importing generation modules: {str(e)}")
                
            # Configure logging
            if 'logger' not in locals():
                logging.basicConfig(level=logging.INFO)
                logger = logging.getLogger("socialsynth.ui")
        except ImportError as e:
            logger.error(f"Error importing modules: {e}")
            st.error(f"Failed to import required modules: {e}")
            raise

# Try to import audio generation libraries
try:
    import gtts
    from gtts import gTTS
    import io
    AUDIO_SUPPORT = True
except ImportError:
    AUDIO_SUPPORT = False
    logger.warning("gTTS not found. Audio generation will be disabled.")

def setup_page():
    """Configure the Streamlit page settings"""
    st.set_page_config(
        page_title="SocialSynth-AI",
        page_icon="logo.png",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def display_sidebar():
    """Display application sidebar with logo and API key status"""
    try:
        st.sidebar.image("logo.png", width=250)
        st.sidebar.title("SocialSynth-AI")
        st.sidebar.markdown("Advanced Content Generation with Knowledge Graphs")
        
        st.sidebar.markdown("---")
        
        api_keys = load_api_keys()
        
        # Check which API keys are available
        has_youtube = bool(api_keys.get("youtube_api_key"))
        has_news = bool(api_keys.get("news_api_key"))
        has_blog_search = bool(api_keys.get("blog_search_api_key")) and bool(api_keys.get("blog_search_cx"))
        
        st.sidebar.subheader("API Keys Status:")
        
        if has_youtube:
            st.sidebar.success("✓ YouTube API Ready")
        else:
            st.sidebar.warning("⚠️ YouTube API Not Configured")
            
        if has_news:
            st.sidebar.success("✓ News API Ready")
        else:
            st.sidebar.warning("⚠️ News API Not Configured")
            
        if has_blog_search:
            st.sidebar.success("✓ Blog Search API Ready")
        else:
            st.sidebar.warning("⚠️ Blog Search API Not Configured")
        
        # Add helpful information about API keys
        if not (has_youtube or has_news or has_blog_search):
            st.sidebar.error("⚠️ No API keys configured. External data sources will be unavailable.")
            st.sidebar.info("To enable external sources, add API keys in the configuration or in the UI.")
        
        # Debug expander to show raw API keys status (masked)
        with st.sidebar.expander("Debug API Keys"):
            st.write("Environment API keys:")
            for key, value in api_keys.items():
                if value:
                    # Show only first and last two characters of the API key
                    masked_value = value[:2] + "..." + value[-2:] if len(value) > 4 else "****"
                    st.code(f"{key}: {masked_value}")
                else:
                    st.code(f"{key}: Not set")
            
            st.write("To add API keys, update your .env file or add them directly in the UI.")
            
        st.sidebar.markdown("---")
        st.sidebar.info("Created by: Raza Abbas")
    except Exception as e:
        logger.error(f"Error displaying sidebar: {e}")
        # Provide minimal sidebar if there's an error
        st.sidebar.title("SocialSynth-AI")

def check_required_packages():
    """Check if required packages for file extraction are installed and show installation instructions if not."""
    missing_packages = []
    
    try:
        import PyPDF2
    except ImportError:
        missing_packages.append("PyPDF2")
    
    try:
        import docx
    except ImportError:
        missing_packages.append("python-docx")
    
    if missing_packages:
        st.warning("Some document processing packages are missing.")
        
        with st.expander("Show Installation Instructions"):
            st.markdown("### Install required packages")
            st.markdown("The following packages are needed for full document processing functionality:")
            
            packages_str = " ".join(missing_packages)
            
            st.code(f"pip install {packages_str}")
            
            st.markdown("After installing, restart the application.")
            
            if "PyPDF2" in missing_packages:
                st.info("PyPDF2 is needed for PDF file processing.")
            
            if "python-docx" in missing_packages:
                st.info("python-docx is needed for DOCX file processing.")
    
    return len(missing_packages) == 0

def script_generator_page():
    """Main page for script generation with all input options"""
    # Import required modules at the start of the function
    from src.socialsynth.utils.api_keys import load_api_keys
    import os
    import time
    
    st.title("Social Media Script Generator")
    
    # Check for required packages
    packages_installed = check_required_packages()
    
    # Load API keys directly from environment
    api_keys = load_api_keys()
    
    # Add helpful debug info for API keys
    with st.expander("API Keys Debug Info"):
        st.info("This section shows what API keys are available from your environment")
        for key_name, key_value in api_keys.items():
            has_key = bool(key_value)
            if has_key:
                st.success(f"✓ {key_name} is configured")
            else:
                st.warning(f"⚠️ {key_name} is not configured")
    
    # Initial setup
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up columns for input options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Content Input")
        input_method = st.radio(
            "Select input method:",
            ["Topic/Keyword", "Upload Documents", "Both"],
            key="input_method_radio"
        )
        
        # Topic/Keyword input
        if input_method in ["Topic/Keyword", "Both"]:
            topic = st.text_input("Enter topic or keywords:", key="topic_input")
            
            # API key configuration for external data sources
            with st.expander("Configure External Data Sources (Optional)"):
                st.info("Add API keys to enable real-time data retrieval from external sources")
                youtube_api_key = st.text_input("YouTube API Key (optional):", type="password", key="youtube_api_key")
                news_api_key = st.text_input("News API Key (optional):", type="password", key="news_api_key")
                blog_search_api_key = st.text_input("Blog Search API Key (optional):", type="password", key="blog_search_api_key")
                
                # Save API keys to session state
                if youtube_api_key:
                    st.session_state['youtube_api_key'] = youtube_api_key
                if news_api_key:
                    st.session_state['news_api_key'] = news_api_key
                if blog_search_api_key:
                    st.session_state['blog_search_api_key'] = blog_search_api_key
                    
                external_sources = st.multiselect(
                    "Select external sources to retrieve from:",
                    ["YouTube", "News", "Blogs"],
                    default=["YouTube", "News"] if youtube_api_key or news_api_key else [],
                    key="external_sources"
                )
        
        # Document upload
        if input_method in ["Upload Documents", "Both"]:
            uploaded_files = st.file_uploader(
                "Upload documents (PDF, DOCX, TXT):", 
                accept_multiple_files=True,
                type=["pdf", "docx", "txt"],
                key="file_uploader"
            )
    
    with col2:
        st.subheader("Content Settings")
        
        # Content type selection
        content_type = st.selectbox(
            "Content type:",
            ["Post", "Thread", "Article", "Video Script", "Tutorial"],
            key="content_type"
        )
        
        # Tone selection
        tone = st.selectbox(
            "Tone:",
            ["Professional", "Casual", "Humorous", "Inspirational", "Educational"],
            key="tone_select"
        )
        
        # Target audience
        target_audience = st.selectbox(
            "Target audience:",
            ["General", "Professionals", "Students", "Enthusiasts", "Beginners"],
            key="audience_select"
        )
        
        # Additional options
        add_hashtags = st.checkbox("Include relevant hashtags", value=True, key="hashtags_checkbox")
        include_sources = st.checkbox("Cite sources", value=True, key="sources_checkbox")
        
        # Enable knowledge graph for better understanding
        use_enhanced_knowledge_graph = st.checkbox(
            "Enable enhanced knowledge graph (recommended)", 
            value=True,
            key="kg_checkbox"
        )
        
        # Max length setting
        max_length = st.slider(
            "Maximum length (words):", 
            min_value=100, 
            max_value=2000, 
            value=500,
            step=100,
            key="length_slider"
        )
    
    # Generate button
    if st.button("Generate Script", key="generate_button"):
        # Validate input
        if (input_method == "Topic/Keyword" and not topic) or \
           (input_method == "Upload Documents" and not uploaded_files) or \
           (input_method == "Both" and not (topic or uploaded_files)):
            st.error("Please provide input as per your selected method.")
            return
        
        # Set up progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Process input and retrieve documents
            status_text.text("Step 1/4: Processing input and retrieving documents...")
            documents = []
            
            # Process uploaded documents
            if input_method in ["Upload Documents", "Both"] and uploaded_files:
                for file in uploaded_files:
                    try:
                        # Save uploaded file temporarily
                        file_path = os.path.join(output_dir, file.name)
                        with open(file_path, "wb") as f:
                            f.write(file.getbuffer())
                        
                        # Extract text based on file type
                        text = extract_text_from_file(file_path)
                        
                        # Create document
                        if text:
                            document = {
                                "page_content": text,
                                "metadata": {
                                    "source": "upload",
                                    "filename": file.name,
                                    "filetype": file.type
                                }
                            }
                            documents.append(document)
                            logger.info(f"Processed uploaded file: {file.name}")
                    except Exception as e:
                        logger.error(f"Error processing file {file.name}: {e}")
                
                progress_bar.progress(25)
            
            # Retrieve from external sources if topic is provided
            if input_method in ["Topic/Keyword", "Both"] and topic:
                try:
                    # Get API keys from both session state and environment
                    env_api_keys = load_api_keys()
                    
                    # Prefer user-supplied API keys, fallback to environment variables
                    youtube_api_key = st.session_state.get('youtube_api_key', '') or env_api_keys.get('youtube_api_key', '')
                    news_api_key = st.session_state.get('news_api_key', '') or env_api_keys.get('news_api_key', '')
                    blog_search_api_key = st.session_state.get('blog_search_api_key', '') or env_api_keys.get('blog_search_api_key', '')
                    
                    # Get selected external sources (lowercase for matching)
                    selected_sources = st.session_state.get('external_sources', [])
                    
                    # Default to all sources if none explicitly selected but keys are available
                    if not selected_sources:
                        if youtube_api_key:
                            selected_sources.append("YouTube")
                        if news_api_key:
                            selected_sources.append("News")
                        if blog_search_api_key:
                            selected_sources.append("Blogs")
                    
                    sources = [source.lower() for source in selected_sources]
                    
                    # Log available API keys
                    logger.info(f"YouTube API key available: {bool(youtube_api_key)}")
                    logger.info(f"News API key available: {bool(news_api_key)}")
                    logger.info(f"Blog search API key available: {bool(blog_search_api_key)}")
                    logger.info(f"Selected sources: {sources}")
                    
                    # Verify API keys are available for selected sources
                    missing_keys = []
                    if 'youtube' in sources and not youtube_api_key:
                        missing_keys.append("YouTube")
                    if 'news' in sources and not news_api_key:
                        missing_keys.append("News")
                    if 'blogs' in sources and not blog_search_api_key:
                        missing_keys.append("Blogs")
                    
                    if missing_keys:
                        warning_msg = f"Missing API keys for: {', '.join(missing_keys)}. These sources will be skipped."
                        logger.warning(warning_msg)
                        st.warning(warning_msg)
                    
                    # Only proceed if we have valid sources with API keys
                    valid_sources = []
                    if 'youtube' in sources and youtube_api_key:
                        valid_sources.append('youtube')
                    if 'news' in sources and news_api_key:
                        valid_sources.append('news')
                    if 'blogs' in sources and blog_search_api_key:
                        valid_sources.append('blogs')
                        
                    if not valid_sources:
                        logger.warning("No valid external sources selected with API keys. Skipping external retrieval.")
                        st.warning("No valid external sources with API keys. Please add API keys or upload documents.")
                    else:
                        # Create retriever adapter with available keys
                        try:
                            retriever = RetrieverAdapter(
                                youtube_api_key=youtube_api_key,
                                news_api_key=news_api_key,
                                blog_search_api_key=blog_search_api_key,
                            )
                            
                            # Log retrieval attempt
                            logger.info(f"Attempting to retrieve documents for topic: {topic}")
                            logger.info(f"Using sources: {valid_sources}")
                            
                            # Retrieve documents
                            try:
                                retrieved_docs = retriever.retrieve(query=topic, sources=valid_sources)
                                
                                if retrieved_docs:
                                    # Convert Retrieved documents to our standard format
                                    for doc in retrieved_docs:
                                        # Check if doc is already a dictionary or Document object
                                        if isinstance(doc, dict):
                                            document = doc
                                        elif hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                                            document = {
                                                "page_content": doc.page_content,
                                                "metadata": doc.metadata
                                            }
                                        else:
                                            # Handle unexpected document format
                                            document = {
                                                "page_content": str(doc),
                                                "metadata": {"source": "external"}
                                            }
                                        documents.append(document)
                                    
                                    logger.info(f"Retrieved {len(retrieved_docs)} documents from external sources")
                                    st.info(f"Retrieved {len(retrieved_docs)} documents from external sources")
                                else:
                                    logger.warning("No documents retrieved from external sources")
                                    st.warning("No documents retrieved from external sources. Proceeding with uploaded documents only.")
                            except Exception as e:
                                logger.error(f"Error in document retrieval: {e}")
                                st.warning(f"Error retrieving documents: {str(e)}. Proceeding with uploaded documents only.")
                        except Exception as e:
                            logger.error(f"Error creating retriever: {e}")
                            st.warning(f"Error initializing retriever: {str(e)}. Proceeding with uploaded documents only.")
                except Exception as e:
                    logger.error(f"Error in external retrieval setup: {e}")
                    st.warning(f"Error setting up external retrieval: {str(e)}. Proceeding with uploaded documents only.")
            
            progress_bar.progress(50)
            
            # Check if we have any documents
            if not documents:
                st.error("No documents could be processed. Please ensure you've either uploaded files or provided a topic with valid API keys.")
                
                # Provide more specific troubleshooting guidance
                if input_method in ["Upload Documents", "Both"] and uploaded_files:
                    st.warning("⚠️ Files were uploaded but couldn't be processed. Common issues:")
                    st.markdown("""
                    - Make sure files are in supported formats (PDF, DOCX, TXT)
                    - Ensure required packages are installed (PyPDF2, python-docx)
                    - Check that the files contain readable text content
                    """)
                
                if input_method in ["Topic/Keyword", "Both"] and topic:
                    st.warning("⚠️ No documents were retrieved for your topic. Common issues:")
                    st.markdown("""
                    - API keys might be missing or invalid
                    - External services might be unavailable
                    - Try a more general or popular topic
                    - Check that you've selected appropriate external sources
                    """)
                    
                    # Show API key status
                    with st.expander("API Key Status"):
                        env_api_keys = load_api_keys()
                        for key_name, key_value in env_api_keys.items():
                            has_key = bool(key_value)
                            if has_key:
                                st.success(f"✓ {key_name} is configured in environment")
                            else:
                                st.warning(f"⚠️ {key_name} is not configured in environment")
                        
                        # Check session state keys
                        st.markdown("---")
                        st.markdown("##### Keys provided in UI:")
                        has_ui_keys = False
                        for key_name in ['youtube_api_key', 'news_api_key', 'blog_search_api_key']:
                            if key_name in st.session_state and st.session_state[key_name]:
                                st.success(f"✓ {key_name} is provided in UI")
                                has_ui_keys = True
                            else:
                                st.warning(f"⚠️ {key_name} is not provided in UI")
                        
                        if not has_ui_keys:
                            st.info("No API keys provided in the UI. Please expand the 'Configure External Data Sources' section and add your API keys.")
                
                return
            
            status_text.text(f"Step 2/4: Building knowledge graph from {len(documents)} documents...")
            
            # Step 2: Build knowledge graph
            try:
                # Initialize appropriate knowledge graph builder
                if use_enhanced_knowledge_graph:
                    kg_builder = EnhancedKnowledgeGraphAdapter(
                        relevance_threshold=0.3,
                        max_entities_per_doc=30,
                        max_keywords=20
                    )
                else:
                    kg_builder = KnowledgeGraphBuilderAdapter()
                
                # Build graph
                kg = kg_builder.build_graph(documents)
                
                if not kg.nodes:
                    st.warning("Knowledge graph is empty. This might indicate an issue with document processing.")
                    logger.warning("Generated knowledge graph has no nodes")
                
                # Visualize graph (if not empty)
                if kg.nodes:
                    output_viz_path = os.path.join(output_dir, "knowledge_graph.html")
                    kg_builder.visualize(kg, output_viz_path)
                    
                    # Get graph summary
                    graph_summary = kg_builder.get_graph_summary(kg)
                    logger.info(f"Knowledge graph summary: {graph_summary}")
                
            except Exception as e:
                logger.error(f"Error building knowledge graph: {e}")
                st.error(f"Error building knowledge graph: {str(e)}")
                return
            
            progress_bar.progress(75)
            status_text.text("Step 3/4: Generating script...")
            
            # Step 3: Generate script
            try:
                # Prepare input for generator
                generator_input = {
                    "topic": topic if input_method in ["Topic/Keyword", "Both"] else "",
                    "documents": documents,
                    "content_type": content_type,
                    "tone": tone,
                    "target_audience": target_audience,
                    "max_length": max_length,
                    "add_hashtags": add_hashtags,
                    "include_sources": include_sources,
                }
                
                # Include graph summary if available
                if 'graph_summary' in locals():
                    generator_input["graph_summary"] = graph_summary
                
                # Generate script
                script_generator = ScriptGenerator()
                script, metadata = script_generator.generate(generator_input)
                
                if not script:
                    st.error("Failed to generate script. Please try again with different input.")
                    return
                
                # Save script
                script_filename = f"script_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                script_path = os.path.join(output_dir, script_filename)
                
                with open(script_path, "w", encoding="utf-8") as f:
                    f.write(script)
                
            except Exception as e:
                logger.error(f"Error generating script: {e}")
                st.error(f"Error generating script: {str(e)}")
                return
            
            progress_bar.progress(100)
            status_text.text("Step 4/4: Script generated successfully!")
            
            # Display results
            st.subheader("Generated Script")
            st.write(script)
            
            # Display knowledge graph visualization if available
            if 'output_viz_path' in locals() and os.path.exists(output_viz_path):
                with st.expander("View Knowledge Graph", expanded=True):
                    st.components.v1.html(open(output_viz_path, 'r', encoding='utf-8').read(), height=600)
            
            # Download button for script
            with open(script_path, "rb") as file:
                st.download_button(
                    label="Download Script",
                    data=file,
                    file_name=script_filename,
                    mime="text/plain"
                )
                
        except Exception as e:
            logger.error(f"Unexpected error in script generation: {e}")
            st.error(f"An unexpected error occurred: {str(e)}")

def about_page():
    """Simple about page with documentation download"""
    st.title("About SocialSynth-AI")
    
    st.markdown("""
    ## SocialSynth-AI v1.0.0
    
    SocialSynth-AI is an advanced content generation tool that leverages AI and knowledge graphs
    to create high-quality content scripts for various platforms.
    
    Built with Streamlit, LangChain, and Google Gemini 2.0 Flash.
    """)
    
    # Add info about documentation
    st.info("Comprehensive documentation will be available in future updates.")
    
    # Add project info
    st.markdown("""
    ### Features:
    - Content script generation for multiple platforms
    - Knowledge graph-enhanced retrieval
    - Multi-angle content generation
    
    ### Coming Soon:
    - PDF documentation generation
    - Audio generation
    - Enhanced real-time data retrieval
    """)

def main():
    """Main application function that sets up the Streamlit UI"""
    try:
        # Ensure os is properly imported at the global level
        setup_page()
        display_sidebar()
        
        # Create output directory for results
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Create two tabs for the application
        tabs = st.tabs(["Script Generator", "About"])
        
        with tabs[0]:
            script_generator_page()
        
        with tabs[1]:
            about_page()
            
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        st.error("Application encountered an error")
        st.markdown(f"""
        ### Something went wrong with SocialSynth-AI
        
        Please try these steps:
        1. Refresh the page and try again
        2. Check that your .env file contains all required API keys
        3. Verify your internet connection
        4. If the problem persists, check the log file at socialsynth.log
        
        We're continuously improving the application. Some features may be limited in this version and will be enhanced in future updates.
        """)
        
        # Show technical error details in an expander for developers
        with st.expander("Technical Error Details (for developers)"):
            st.code(f"Error: {str(e)}")
            st.info("This is a beta version of SocialSynth-AI. Please report issues to improve the application.")

if __name__ == "__main__":
    main() 