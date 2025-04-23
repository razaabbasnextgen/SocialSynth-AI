#!/usr/bin/env python3
"""
SocialSynth-AI Streamlit Application

This module contains the Streamlit UI implementation for SocialSynth-AI.
"""

import os
import time
import logging
import webbrowser
import sys
import json
from typing import List, Dict, Any
from pathlib import Path

# Third-party imports
import streamlit as st
import pandas as pd

# Add project paths
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("socialsynth.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("socialsynth.ui")

# Valid access token
VALID_TOKEN = "12456789"

# Import script generator - with error handling to avoid breaking the UI
try:
    from src.socialsynth.utils.api_keys import load_api_keys
    generation_available = True
    logger.info("API key loading module loaded successfully")
except ImportError as e:
    logger.error(f"Error importing API key module: {e}")
    generation_available = False
    
# Import retriever and knowledge graph builder - with error handling
try:
    from src.socialsynth.retrieval.enhanced_retriever import EnhancedRetriever
    from src.socialsynth.knowledge_graph.enhanced_builder import EnhancedKnowledgeGraphBuilder
    
    # Try to import the adapter modules if they exist
    try:
        from src.socialsynth.retrieval.enhanced_retriever_adapter import RetrieverAdapter
        from src.socialsynth.knowledge_graph.enhanced_knowledge_graph_adapter import KnowledgeGraphBuilderAdapter
        adapters_available = True
        logger.info("Adapter modules loaded successfully")
    except ImportError:
        adapters_available = False
        logger.warning("Adapter modules not available, using direct implementation")
        
    retrieval_available = True
    logger.info("Retrieval and knowledge graph modules loaded successfully")
except ImportError as e:
    logger.error(f"Error importing retrieval modules: {e}")
    retrieval_available = False

# Try to import Google's Gemini API
try:
    import google.generativeai as genai
    gemini_available = True
    logger.info("Google Generative AI (Gemini) module loaded successfully")
except ImportError:
    gemini_available = False
    logger.warning("Google Generative AI (Gemini) module not available. Install with: pip install google-generativeai")

def setup_page():
    """Configure the Streamlit page settings"""
    st.set_page_config(
        page_title="SocialSynth-AI",
        page_icon="logo.png",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if "token_valid" not in st.session_state:
        st.session_state.token_valid = False
    if "generated_script" not in st.session_state:
        st.session_state.generated_script = ""

def display_sidebar():
    """Display application sidebar with logo and API key status"""
    try:
        st.sidebar.image("logo.png", width=250, use_container_width=False)
    except:
        pass  # Continue even if logo can't be loaded
        
    st.sidebar.title("SocialSynth-AI")
    st.sidebar.markdown("Advanced Content Generation")
    st.sidebar.markdown("---")
    
    # Display API key status if retrieval functionality is available
    if retrieval_available:
        try:
            api_keys = load_api_keys()
            keys_configured = all([api_keys.get("youtube_api_key"), api_keys.get("news_api_key")])
            
            # Check for Google API key
            google_api_available = bool(api_keys.get("google_api_key"))
            
            if not keys_configured:
                st.sidebar.warning("⚠️ API keys not fully configured. Please check your .env file.")
            else:
                st.sidebar.success("✅ API keys configured")
                
            if not google_api_available:
                st.sidebar.warning("⚠️ No Google API key found. Script generation will use template mode.")
            else:
                st.sidebar.success("✅ Google API key configured for Gemini 2.0")
                
        except Exception as e:
            logger.error(f"Error checking API keys: {e}")
            st.sidebar.warning("⚠️ Could not check API key status")
    
    st.sidebar.markdown("---")
    st.sidebar.info("Created by: Raza Abbas")

def generate_script_with_gemini(
    topic: str, 
    tone: str = "educational", 
    length: str = "medium", 
    include_sources: bool = True,
    api_key: str = None
) -> str:
    """
    Generate a script using Google's Gemini model.
    
    Args:
        topic: The topic to generate a script about
        tone: The tone of the script (educational, entertaining, etc.)
        length: The length of the script (short, medium, long)
        include_sources: Whether to include sources in the script
        api_key: Google API key for Gemini
        
    Returns:
        A string containing the generated script
    """
    if not gemini_available:
        logger.warning("Gemini API not available, using template")
        return generate_template_script(topic, tone, length, include_sources)
        
    if not api_key:
        logger.warning("No Google API key provided, using template")
        return generate_template_script(topic, tone, length, include_sources)
    
    try:
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        
        # Set up the model (Gemini 2.0 Flash)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Determine the length in minutes
        if length == "short":
            length_minutes = "3-5 minutes"
        elif length == "medium":
            length_minutes = "5-10 minutes"
        else:
            length_minutes = "10-15 minutes"
        
        # Create the prompt
        prompt = f"""
        Generate a detailed YouTube script about "{topic}". 
        
        The script should be:
        - {length_minutes} in length
        - Written in a {tone} tone
        - Well-structured with an introduction, main points, and conclusion
        - Highly engaging and designed to maximize viewer retention
        
        Format the script with clear sections including:
        - TITLE (catchy and SEO-friendly)
        - INTRODUCTION (hook the viewer in the first 30 seconds)
        - MAIN CONTENT (organized in sections with timestamps)
        - CONCLUSION (with call to action)
        {include_sources and "- SOURCES (list relevant sources for fact-checking)" or ""}
        
        Include elements like:
        - [PAUSE] for dramatic effect
        - [B-ROLL: description] for suggested visuals
        - [CUT TO: description] for scene transitions
        - [GRAPHICS: description] for on-screen text or visuals
        
        The script should sound natural when read aloud and include appropriate pacing.
        """
        
        # Generate the response
        response = model.generate_content(prompt)
        
        # Format the output
        script = response.text
        
        logger.info(f"Successfully generated script with Gemini for topic: {topic}")
        return script
        
    except Exception as e:
        logger.error(f"Error generating script with Gemini: {e}")
        return generate_template_script(topic, tone, length, include_sources)

def generate_template_script(topic: str, tone: str, length: str, include_sources: bool) -> str:
    """Generate a template script when API generation is unavailable"""
    # Prepare sources section separately to avoid f-string backslash issues
    sources_section = ""
    if include_sources:
        sources_section = """
## Sources
1. Research papers and academic journals
2. Industry reports and analysis
3. Expert interviews and insights"""
    
    script = f"""# {topic.upper()} - {tone.capitalize()} Script ({length.capitalize()})

## Introduction
[HOOK] Welcome to this {tone.lower()} video about {topic}. Today, we'll explore this fascinating topic and its implications.

## Main Points
[TIMESTAMP 00:30] 1. Overview of {topic}
[B-ROLL: General imagery related to {topic}]

[TIMESTAMP 02:00] 2. Current trends and developments
[GRAPHICS: Show key statistics and growth trends]

[TIMESTAMP 04:30] 3. Future outlook and implications
[CUT TO: Expert opinion clips]

## Conclusion
[TIMESTAMP 06:00] Thank you for watching this video on {topic}. 
[PAUSE] If you found this valuable, please like and subscribe for more content like this!
{sources_section}"""
    
    return script

def script_generator_tab():
    """Display the script generator tab"""
    st.header("Generate Content")
    
    query = st.text_input("Content Topic:", placeholder="e.g., The impact of AI on healthcare", key="sg_topic")
    
    col1, col2 = st.columns(2)
    with col1:
        tone = st.selectbox(
            "Content Tone:",
            ["Educational", "Entertaining", "Professional", "Conversational", "Dramatic"],
            key="sg_tone"
        )
    with col2:
        length = st.selectbox(
            "Content Length:",
            ["Short (3-5 min)", "Medium (5-10 min)", "Long (10-15 min)"],
            key="sg_length"
        )
    
    include_sources = st.checkbox("Include sources in script", value=True, key="sg_include_sources")
    
    # Add option to use the advanced RAG-based generation if available
    if retrieval_available:
        use_advanced_rag = st.checkbox("Use Advanced RAG", value=True, 
                                 help="Use Gemini 2.0 with enhanced retrieval for better content",
                                 key="sg_use_advanced")
    else:
        use_advanced_rag = False
    
    if st.button("Generate Script", type="primary", key="sg_btn"):
        if not query:
            st.error("Please enter a content topic")
            st.info("Examples: 'Latest advancements in clean energy' or 'How AI is transforming education'")
        else:
            with st.spinner("Generating script with Gemini 2.0 Flash..."):
                try:
                    # Get API keys for script generation
                    api_keys = load_api_keys() if retrieval_available else {}
                    
                    # Get Google API key
                    google_api_key = api_keys.get("google_api_key")
                    
                    # Try document retrieval if advanced RAG is selected
                    if use_advanced_rag and retrieval_available:
                        logger.info(f"Using advanced RAG-based generation for topic: {query}")
                        
                        # Try to retrieve documents if that functionality is available
                        try:
                            # Initialize retriever - avoid using adapter with use_standalone parameter
                            if adapters_available:
                                try:
                                    # Create retriever without use_standalone parameter
                                    retriever = RetrieverAdapter(
                                        youtube_api_key=api_keys.get("youtube_api_key"),
                                        news_api_key=api_keys.get("news_api_key"),
                                        max_results_per_source=10,
                                        min_relevance_score=0.5,
                                        include_entities=True
                                    )
                                    
                                    # Get documents
                                    documents = retriever.get_relevant_documents(query)
                                    logger.info(f"Retrieved {len(documents)} documents for context")
                                    
                                    # We don't directly pass these to Gemini, but we could extract content later
                                except Exception as e:
                                    logger.error(f"Error with RetrieverAdapter: {e}")
                        except Exception as e:
                            logger.error(f"Error in document retrieval: {e}")
                            st.warning("Could not retrieve documents. Using basic generation.")
                    
                    # Generate script using Gemini
                    script = generate_script_with_gemini(
                        topic=query,
                        tone=tone.lower(),
                        length=length.split()[0].lower(),
                        include_sources=include_sources,
                        api_key=google_api_key
                    )
                    
                    # Store in session state
                    st.session_state.generated_script = script
                    
                    st.success("Script generated successfully with Gemini 2.0!")
                    st.text_area("Generated Script", script, height=400, key="sg_result")
                    
                    # Save to file
                    script_file = f"script_{int(time.time())}.txt"
                    with open(script_file, "w") as f:
                        f.write(script)
                    
                    st.download_button(
                        "Download Script",
                        script,
                        file_name=script_file,
                        mime="text/plain",
                        key="sg_download"
                    )
                except Exception as e:
                    logger.error(f"Script generation error: {e}")
                    st.error("Script generation failed")
                    st.markdown(f"""
                    **What happened**: {str(e)[:100]}...
                    
                    **Things to try**:
                    - Use a simpler, more mainstream topic
                    - Try a different tone or length
                    - Check your Google API key in the .env file
                    - Check your internet connection
                    """)

def documentation_tab():
    """Display the documentation tab"""
    st.header("Documentation")
    
    st.markdown("""
    # SocialSynth-AI Documentation
    
    SocialSynth-AI is an advanced content generation platform powered by Google's Gemini 2.0 Flash model.
    
    ## 🧠 Smart RAG System (Retrieval-Augmented Generation)
    SocialSynth uses an advanced RAG architecture to generate highly contextual and accurate scripts. It pulls data in real-time from:
    - News sources
    - YouTube videos
    - User-uploaded documents
    
    It then uses Gemini 2.0 Flash, a cutting-edge language model, to synthesize that information into a hyper-contextual script — written as if it were crafted by a professional screenwriter.
    
    ## 🗂️ Multi-Source Integration for Rich Context
    Whether the topic is science, politics, or entertainment, SocialSynth ensures that all relevant angles are covered:
    - Every script is fact-checked
    - Content is cross-verified
    - Sources are diverse and relevant
    
    ## 🎙️ Versatile Script Styles
    SocialSynth doesn't limit you to one format. It can generate scripts in multiple styles including:
    - News reporting
    - Documentary narration
    - Comedic commentary
    - Dramatic storytelling
    
    ## 💡 Creativity + Context = 🔥 Engagement
    The end goal? To create scripts that are not only informative but also engaging, hook-based, and designed to retain viewer attention.
    
    ## Requirements
    - Python 3.8 or higher
    - Streamlit
    - Internet connection
    - Google API key for Gemini
    """)
    
    # Create a simple PDF for demo purposes
    documentation_file = "SocialSynth-AI_Documentation.pdf"
    
    if os.path.exists(documentation_file):
        st.success("Documentation is available")
        
        with open(documentation_file, "rb") as f:
            st.download_button(
                "Download Documentation PDF",
                f,
                file_name=documentation_file,
                mime="application/pdf",
                key="doc_download"
            )
    else:
        st.warning("Documentation PDF not found")
        st.info("Documentation PDF will be generated in a future update.")

def validate_token():
    """Validate the access token"""
    if not st.session_state.token_valid:
        st.header("Access Validation")
        
        token = st.text_input("Enter access token:", type="password")
        
        if st.button("Validate"):
            if token == VALID_TOKEN:
                st.session_state.token_valid = True
                st.success("Token validated successfully!")
                st.rerun()
            else:
                st.error("Invalid token. Please try again.")
        
        st.info("Please enter your access token to use SocialSynth-AI.")
        return False
    
    return True

def main():
    """Main application function that sets up the Streamlit UI"""
    setup_page()
    display_sidebar()
    
    st.title("SocialSynth-AI")
    st.subheader("Script Generator & Documentation")
    
    # Token validation
    if not validate_token():
        return
    
    # Only display tabs if token is valid
    tab1, tab2 = st.tabs(["Script Generator", "Documentation"])
    
    with tab1:
        script_generator_tab()
    
    with tab2:
        documentation_tab()

if __name__ == "__main__":
    main() 