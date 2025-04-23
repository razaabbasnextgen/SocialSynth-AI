import streamlit as st
import os
import sys
import logging
import time
import asyncio
import requests
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import json

from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from advanced_rag_v2 import EnhancedKnowledgeGraph, ContextualReranker, visualize_knowledge_graph

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
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY") 
BLOG_SEARCH_CX = os.getenv("BLOG_SEARCH_CX")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

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

# Create summarization LLM (lower temperature for more factual summaries)
def create_summary_llm(model_name="gemini-1.5-flash"):
    """Create LLM instance for summarization"""
    try:
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.2,
            convert_system_message_to_human=True
        )
    except Exception as e:
        logger.error(f"Failed to initialize summarization LLM: {e}")
        try:
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                google_api_key=GOOGLE_API_KEY,
                temperature=0.2,
                convert_system_message_to_human=True
            )
        except:
            raise

# Get YouTube data
def get_youtube_data(query, max_results=5):
    """Fetch YouTube videos related to the query"""
    try:
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        
        # Search for videos
        search_response = youtube.search().list(
            q=query,
            part='snippet',
            maxResults=max_results,
            type='video',
            relevanceLanguage='en',
            order='relevance'
        ).execute()
        
        video_data = []
        
        for item in search_response['items']:
            video_id = item['id']['videoId']
            
            # Get more detailed video information
            video_response = youtube.videos().list(
                part='snippet,statistics,contentDetails',
                id=video_id
            ).execute()
            
            if video_response['items']:
                video_info = video_response['items'][0]
                video = {
                    'id': video_id,
                    'title': video_info['snippet']['title'],
                    'description': video_info['snippet']['description'],
                    'channel': video_info['snippet']['channelTitle'],
                    'publish_date': video_info['snippet']['publishedAt'],
                    'view_count': video_info['statistics'].get('viewCount', '0'),
                    'like_count': video_info['statistics'].get('likeCount', '0'),
                    'url': f"https://www.youtube.com/watch?v={video_id}"
                }
                video_data.append(video)
        
        return video_data
    except HttpError as e:
        logger.error(f"YouTube API error: {e}")
        return []
    except Exception as e:
        logger.error(f"Error fetching YouTube data: {e}")
        return []

# Get blog data
def get_blog_articles(query, max_results=5):
    """Fetch blog articles related to the query using Google Custom Search"""
    try:
        url = f"https://www.googleapis.com/customsearch/v1"
        params = {
            'key': GOOGLE_API_KEY,
            'cx': BLOG_SEARCH_CX,
            'q': query,
            'num': max_results
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if 'items' not in data:
            logger.warning(f"No blog results found for query: {query}")
            return []
        
        articles = []
        for item in data['items']:
            article = {
                'title': item.get('title', ''),
                'link': item.get('link', ''),
                'snippet': item.get('snippet', ''),
                'source': item.get('displayLink', ''),
                'date': item.get('pagemap', {}).get('metatags', [{}])[0].get('article:published_time', '')
            }
            articles.append(article)
        
        return articles
    except Exception as e:
        logger.error(f"Error fetching blog articles: {e}")
        return []

# Get news data
def get_news_articles(query, max_results=5):
    """Fetch news articles related to the query using NewsAPI"""
    try:
        url = f"https://newsapi.org/v2/everything"
        params = {
            'apiKey': NEWS_API_KEY,
            'q': query,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': max_results
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if 'articles' not in data:
            logger.warning(f"No news results found for query: {query}")
            return []
        
        return data['articles']
    except Exception as e:
        logger.error(f"Error fetching news articles: {e}")
        return []

# Parallel data fetching
async def fetch_data_parallel(query, youtube_count=5, blog_count=5, news_count=5):
    """Fetch data from multiple sources in parallel"""
    loop = asyncio.get_event_loop()
    results = {"youtube": [], "blogs": [], "news": []}
    
    async def safe_execute(data_type, func, *args):
        try:
            result = await loop.run_in_executor(executor, func, *args)
            results[data_type] = result
        except Exception as e:
            logger.error(f"Error in parallel fetch for {data_type}: {e}")
            results[data_type] = []
    
    with ThreadPoolExecutor() as executor:
        tasks = [
            safe_execute("youtube", get_youtube_data, query, youtube_count),
            safe_execute("blogs", get_blog_articles, query, blog_count),
            safe_execute("news", get_news_articles, query, news_count)
        ]
        await asyncio.gather(*tasks)
    
    return results

# Process and summarize data
def preprocess_data(data, summary_llm):
    """Process and summarize collected data into document chunks"""
    documents = []
    
    # Process YouTube data
    for video in data.get("youtube", []):
        content = f"YOUTUBE VIDEO: {video['title']}\n\nChannel: {video['channel']}\n\nDescription: {video['description']}\n\nViews: {video['view_count']}"
        metadata = {
            "source": "youtube",
            "title": video["title"],
            "url": video["url"],
            "date": video["publish_date"]
        }
        documents.append(Document(page_content=content, metadata=metadata))
    
    # Process blog data
    for article in data.get("blogs", []):
        content = f"BLOG ARTICLE: {article['title']}\n\nSource: {article['source']}\n\nExcerpt: {article['snippet']}"
        metadata = {
            "source": "blog",
            "title": article["title"],
            "url": article["link"],
            "date": article.get("date", "")
        }
        documents.append(Document(page_content=content, metadata=metadata))
    
    # Process news data
    for article in data.get("news", []):
        content = f"NEWS ARTICLE: {article['title']}\n\nSource: {article['source']['name']}\n\nExcerpt: {article['description']}"
        metadata = {
            "source": "news",
            "title": article["title"],
            "url": article["url"],
            "date": article.get("publishedAt", "")
        }
        documents.append(Document(page_content=content, metadata=metadata))
    
    # Summarize and enhance documents if we have many of them
    if len(documents) > 5:
        enhanced_docs = summarize_documents(documents, summary_llm)
        return enhanced_docs
    
    return documents

def summarize_documents(documents, llm):
    """Summarize and enhance documents to extract key points"""
    enhanced_docs = []
    
    for doc in documents:
        try:
            # System prompt for summarization
            system_prompt = """You are an AI that extracts and summarizes key information from content.
Extract the most important insights, facts, and trends from the provided content.
Focus on information that would be valuable for creating an informative YouTube script.
Format your response as concise bullet points of the key information."""

            # Generate summary
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Extract key information from this content:\n\n{doc.page_content}")
            ]
            
            response = llm.invoke(messages)
            
            # Create enhanced document with original + summary
            enhanced_content = f"{doc.page_content}\n\nKEY POINTS:\n{response.content}"
            enhanced_docs.append(Document(
                page_content=enhanced_content,
                metadata=doc.metadata
            ))
            
        except Exception as e:
            logger.error(f"Error summarizing document: {e}")
            enhanced_docs.append(doc)  # Fall back to original document
    
    return enhanced_docs

# Generate script
def generate_script(query, tone, length, knowledge_graph, reranker, llm):
    """Generate a YouTube script based on retrieved information"""
    try:
        # Get relevant documents from knowledge graph
        relevant_docs = knowledge_graph.retrieve(query=query, k=5, use_hybrid=True)
        
        # Rerank documents
        reranked_docs = reranker.rerank(query, relevant_docs)
        
        # Format context from documents
        docs_context = "\n\n---\n\n".join([f"SOURCE ({doc.metadata.get('source', 'unknown')}):\n{doc.page_content}" for doc in reranked_docs])
        
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

Use the real-world information provided below and organize the script with:
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

# Main Streamlit app
def main():
    # Set up event loop for async operations
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    st.set_page_config(
        page_title="YouTube Script Generator",
        page_icon="🎬",
        layout="wide"
    )
    
    st.title("🎬 YouTube Script Generator with RAG")
    st.markdown("Generate professional YouTube scripts with the latest information using Gemini 2.0 Flash and knowledge graphs")
    
    # Initialize components
    try:
        with st.spinner("Loading models and components..."):
            # Initialize embeddings, vector store, LLMs
            embedding_function = get_embeddings()
            vector_store = setup_vector_store(embedding_function)
            llm = create_llm()
            summary_llm = create_summary_llm()
            
            # Initialize knowledge graph and reranker
            knowledge_graph = EnhancedKnowledgeGraph(embedding_function)
            reranker = ContextualReranker()
        
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
            
            data_sources = st.multiselect(
                "Data Sources:",
                options=["YouTube", "Blogs", "News"],
                default=["YouTube", "Blogs", "News"]
            )
            
            submit_button = st.form_submit_button("Generate Script")
        
        # Process form submission
        if submit_button and query:
            with st.spinner("Collecting latest information..."):
                # Determine which sources to use
                youtube_count = 5 if "YouTube" in data_sources else 0
                blog_count = 5 if "Blogs" in data_sources else 0
                news_count = 5 if "News" in data_sources else 0
                
                # Run async data collection
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                data = loop.run_until_complete(fetch_data_parallel(
                    query,
                    youtube_count=youtube_count,
                    blog_count=blog_count,
                    news_count=news_count
                ))
                loop.close()
            
            with st.spinner("Processing and summarizing information..."):
                # Process and summarize data
                documents = preprocess_data(data, summary_llm)
                
                # Add documents to vector store and knowledge graph
                vector_store.add_documents(documents)
                knowledge_graph.add_documents(documents)
                
                st.success(f"Collected {len(documents)} sources of information")
            
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
                if knowledge_graph.visualize("knowledge_graph.html"):
                    with open("knowledge_graph.html", "r", encoding="utf-8") as f:
                        html_content = f.read()
                    st.components.v1.html(html_content, height=600)
            
            # Show sources
            st.markdown("### 📚 Sources Used")
            with st.expander("View Sources"):
                for i, doc in enumerate(result["documents"]):
                    st.markdown(f"**Source {i+1}**: {doc.metadata.get('title', 'Unknown')}")
                    st.markdown(f"*Type*: {doc.metadata.get('source', 'Unknown')} | *URL*: {doc.metadata.get('url', 'N/A')}")
                    st.text(doc.page_content[:300] + "...")
                    st.divider()
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        logger.error(f"Application error: {e}", exc_info=True)

if __name__ == "__main__":
    main() 