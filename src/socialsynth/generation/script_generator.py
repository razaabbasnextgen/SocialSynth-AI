#!/usr/bin/env python3
"""
Script Generator

This module provides functionality to generate content scripts based on knowledge graphs and documents.
It creates well-structured content for different formats and tones.
"""

import logging
import random
from typing import List, Dict, Any, Optional, Tuple
import networkx as nx
import json
import os
import time
from datetime import datetime

# Configure logging
logger = logging.getLogger("socialsynth.generation.script")

# Import langchain components
try:
    from langchain.prompts import PromptTemplate
except ImportError:
    from langchain_core.prompts import PromptTemplate

class ScriptGenerator:
    """
    Script Generator for creating content from knowledge graphs and documents.
    
    This class provides methods to generate various types of content scripts
    based on the provided documents, knowledge graph, and user preferences.
    """
    
    def __init__(self, model_name: str = "gemini"):
        """
        Initialize the script generator.
        
        Args:
            model_name: The AI model to use for generation
        """
        self.model_name = model_name
        self.max_retries = 3
        logger.info("Initializing ScriptGenerator")
        
        # Set up the AI model
        try:
            if model_name.lower() == "gemini":
                import google.generativeai as genai
                
                # Configure API key
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    logger.warning("GOOGLE_API_KEY not found in environment variables")
                    
                genai.configure(api_key=api_key)
                
                # Get available models
                try:
                    self.models = genai.list_models()
                    self.gemini_models = [m for m in self.models if "gemini" in m.name.lower()]
                    self.model = genai.GenerativeModel(model_name="gemini-pro")
                    logger.info(f"Successfully initialized Gemini model: gemini-pro")
                except Exception as e:
                    logger.error(f"Error initializing Gemini model: {e}")
                    self.model = None
            else:
                logger.warning(f"Unsupported model: {model_name}")
                self.model = None
        except ImportError:
            logger.warning(f"Required package for {model_name} not installed")
            self.model = None
            
    def _extract_graph_insights(self, graph_summary: Dict[str, Any]) -> str:
        """
        Extract key insights from knowledge graph to enhance script generation.
        
        Args:
            graph_summary: Summary data from a knowledge graph
            
        Returns:
            String of formatted insights
        """
        insights = []
        
        # Get node types and counts
        if "node_types" in graph_summary:
            node_types = graph_summary["node_types"]
            insights.append(f"The knowledge graph contains {graph_summary.get('num_nodes', 0)} nodes and {graph_summary.get('num_edges', 0)} edges.")
            
            # Add node type breakdown
            node_breakdown = [f"{count} {node_type}s" for node_type, count in node_types.items() if count > 0]
            if node_breakdown:
                insights.append(f"Node types include: {', '.join(node_breakdown)}.")
        
        # Add key entities if available
        if "top_central_entities" in graph_summary:
            top_entities = [entity["label"] for entity in graph_summary["top_central_entities"][:5]]
            if top_entities:
                insights.append(f"Key entities: {', '.join(top_entities)}.")
        elif "top_central_nodes" in graph_summary:
            top_nodes = [node["label"] for node in graph_summary["top_central_nodes"][:5]]
            if top_nodes:
                insights.append(f"Key nodes: {', '.join(top_nodes)}.")
                
        # Add key concepts/keywords
        if "top_keywords" in graph_summary:
            if isinstance(graph_summary["top_keywords"], list):
                if len(graph_summary["top_keywords"]) > 0:
                    if isinstance(graph_summary["top_keywords"][0], tuple):
                        keywords = [kw[0] for kw in graph_summary["top_keywords"][:5]]
                    else:
                        keywords = [kw for kw in graph_summary["top_keywords"][:5]]
                    if keywords:
                        insights.append(f"Key concepts: {', '.join(keywords)}.")
        
        # Add relationship info if available
        if "relation_types" in graph_summary:
            relation_types = graph_summary["relation_types"]
            if relation_types and isinstance(relation_types, dict):
                rel_types = [f"{relation}" for relation, count in relation_types.items() 
                            if relation != "unknown" and count > 0]
                if rel_types:
                    insights.append(f"Relationship types: {', '.join(rel_types)}.")
        
        # Add community info if available
        if "communities" in graph_summary and graph_summary["communities"] > 1:
            insights.append(f"The graph contains {graph_summary['communities']} distinct topic clusters.")
            
        return "\n".join(insights)
        
    def _extract_document_content(self, documents: List[Dict[str, Any]], max_chars: int = 1000) -> str:
        """
        Extract content from documents for use in prompt creation.
        
        Args:
            documents: List of documents
            max_chars: Maximum characters to include
            
        Returns:
            String of formatted document summaries
        """
        doc_summaries = []
        char_count = 0
        
        for i, doc in enumerate(documents[:5]):  # Limit to 5 documents
            # Get content and metadata
            content = ""
            if hasattr(doc, 'page_content'):
                content = doc.page_content
            elif isinstance(doc, dict):
                content = doc.get('content', doc.get('page_content', ''))
                
            # Get source
            source = "Unknown"
            if hasattr(doc, 'metadata') and hasattr(doc.metadata, 'get'):
                source = doc.metadata.get('source', 'Unknown')
            elif isinstance(doc, dict) and 'metadata' in doc:
                source = doc['metadata'].get('source', 'Unknown')
                
            # Truncate content to avoid overly long prompts
            if content:
                truncated = content[:200] + "..." if len(content) > 200 else content
                doc_summary = f"Document {i+1} (Source: {source}): {truncated}"
                
                if char_count + len(doc_summary) <= max_chars:
                    doc_summaries.append(doc_summary)
                    char_count += len(doc_summary)
                else:
                    break
                    
        return "\n\n".join(doc_summaries)
    
    def _create_hashtags(self, topic: str, graph_summary: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Create relevant hashtags based on topic and graph data.
        
        Args:
            topic: Main topic of content
            graph_summary: Knowledge graph summary data
            
        Returns:
            List of hashtags
        """
        hashtags = []
        
        # Add basic topic hashtags
        topic_words = topic.lower().split()
        for word in topic_words:
            if len(word) > 3 and word not in ["with", "that", "this", "from", "have", "some", "what"]:
                hashtags.append(word)
                
        # Add hashtags from graph data if available
        if graph_summary:
            if "top_central_nodes" in graph_summary:
                for node in graph_summary["top_central_nodes"][:3]:
                    label = node["label"]
                    if ' ' in label:
                        # Handle multi-word terms by removing spaces
                        label = label.replace(' ', '')
                    hashtags.append(label.lower())
            
            if "top_keywords" in graph_summary:
                if isinstance(graph_summary["top_keywords"], list):
                    if len(graph_summary["top_keywords"]) > 0:
                        if isinstance(graph_summary["top_keywords"][0], tuple):
                            for keyword, _ in graph_summary["top_keywords"][:3]:
                                if ' ' in keyword:
                                    # Handle multi-word terms by removing spaces
                                    keyword = keyword.replace(' ', '')
                                hashtags.append(keyword.lower())
        
        # Remove duplicates and format
        unique_hashtags = list(set(hashtags))
        formatted_hashtags = ["#" + tag.strip("#") for tag in unique_hashtags if tag and len(tag.strip("#")) > 2]
        
        return formatted_hashtags[:10]  # Limit to 10 hashtags
    
    def _generate_with_gemini(self, prompt: str, max_retries: int = 3) -> str:
        """
        Generate content using the Gemini model.
        
        Args:
            prompt: Generation prompt
            max_retries: Maximum number of retry attempts
            
        Returns:
            Generated text
        """
        if not self.model:
            logger.error("Gemini model not initialized")
            return ""
            
        retries = 0
        while retries < max_retries:
            try:
                response = self.model.generate_content(prompt)
                if hasattr(response, 'text'):
                    return response.text
                else:
                    logger.warning("Unexpected response format from Gemini API")
                    return str(response)
            except Exception as e:
                logger.error(f"Error generating with Gemini (attempt {retries+1}/{max_retries}): {e}")
                retries += 1
                time.sleep(2)  # Wait before retrying
                
        return "Error generating content. Please try again."
        
    def generate(self, input_data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a content script based on input data.
        
        Args:
            input_data: Dictionary containing generation parameters
            
        Returns:
            Tuple of (generated script, metadata)
        """
        # Extract parameters
        topic = input_data.get("topic", "")
        documents = input_data.get("documents", [])
        content_type = input_data.get("content_type", "Post")
        tone = input_data.get("tone", "Professional")
        target_audience = input_data.get("target_audience", "General")
        max_length = input_data.get("max_length", 500)
        add_hashtags = input_data.get("add_hashtags", True)
        include_sources = input_data.get("include_sources", True)
        graph_summary = input_data.get("graph_summary", {})
        
        logger.info(f"Generating script for topic: {topic}")
        
        # Create metadata
        metadata = {
            "topic": topic,
            "content_type": content_type,
            "tone": tone,
            "target_audience": target_audience,
            "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sources": []
        }
        
        # Extract document sources for citation
        if include_sources and documents:
            for doc in documents:
                source = None
                if hasattr(doc, 'metadata') and hasattr(doc.metadata, 'get'):
                    source = doc.metadata.get('source')
                    url = doc.metadata.get('url', '')
                elif isinstance(doc, dict) and 'metadata' in doc:
                    source = doc['metadata'].get('source')
                    url = doc['metadata'].get('url', '')
                
                if source and source not in [s["name"] for s in metadata["sources"]]:
                    metadata["sources"].append({"name": source, "url": url or ""})
        
        # Build the prompt
        prompt_parts = []
        
        # 1. Basic instruction
        prompt_parts.append(f"""
        Generate a high-quality {content_type} script about "{topic}" in a {tone} tone for a {target_audience} audience.
        The script should be well-structured, engaging, and factually accurate. 
        Maximum length: {max_length} words.
        """)
        
        # 2. Add knowledge graph insights if available
        if graph_summary:
            graph_insights = self._extract_graph_insights(graph_summary)
            if graph_insights:
                prompt_parts.append(f"""
                KNOWLEDGE GRAPH INSIGHTS:
                {graph_insights}
                
                Use these insights to structure your content around the most important entities and concepts.
                """)
        
        # 3. Add document content if available
        if documents:
            doc_content = self._extract_document_content(documents)
            if doc_content:
                prompt_parts.append(f"""
                DOCUMENT CONTENT:
                {doc_content}
                
                Use this information to make your content accurate and comprehensive.
                """)
        
        # 4. Add content type-specific instructions
        content_type_instructions = {
            "Post": "Create a concise social media post that captures attention quickly. Include a hook, main point, and call to action.",
            "Thread": "Create a thread of 3-5 connected points that build on each other. Each point should be substantial but concise.",
            "Article": "Create a well-structured article with an introduction, 2-4 main sections with subheadings, and a conclusion.",
            "Video Script": "Create a video script with clear sections for Intro Hook, Main Points, and Call To Action. Include natural transitions between sections.",
            "Tutorial": "Create a step-by-step tutorial with a brief introduction, clear numbered steps, and a conclusion summarizing the benefits."
        }
        
        prompt_parts.append(f"""
        CONTENT TYPE REQUIREMENTS:
        {content_type_instructions.get(content_type, "Create well-structured content with a clear beginning, middle, and end.")}
        """)
        
        # 5. Add tone-specific instructions
        tone_instructions = {
            "Professional": "Use formal language, cite facts, and maintain an authoritative voice throughout.",
            "Casual": "Use conversational language, personal anecdotes, and a friendly, approachable tone.",
            "Humorous": "Incorporate appropriate humor, wit, and a light-hearted approach while still delivering value.",
            "Inspirational": "Use motivational language, emotional appeals, and uplifting examples.",
            "Educational": "Focus on clear explanations, examples, and takeaway lessons the audience can apply."
        }
        
        prompt_parts.append(f"""
        TONE REQUIREMENTS:
        {tone_instructions.get(tone, "Maintain a balanced, clear, and engaging voice throughout.")}
        """)
        
        # 6. Format instructions
        if content_type == "Video Script":
            prompt_parts.append("""
            FORMAT REQUIREMENTS:
            Organize the script with clear section headers:
            
            Intro Hook: A compelling opening that grabs attention
            Main Points: The core content divided into clear sections
            Call To Action: What you want viewers to do next
            """)
        elif content_type == "Thread":
            prompt_parts.append("""
            FORMAT REQUIREMENTS:
            Organize the thread with clear numbered parts (Tweet 1, Tweet 2, etc.)
            Make each section concise but substantial
            Ensure a logical flow between sections
            End with a clear conclusion or call to action
            """)
        
        # 7. Citation instructions
        if include_sources and metadata["sources"]:
            sources_list = "\n".join([f"- {source['name']}" + (f" - {source['url']}" if source['url'] else "") 
                                    for source in metadata["sources"]])
            prompt_parts.append(f"""
            CITATION REQUIREMENTS:
            Include a "Sources:" section at the end listing these sources:
            {sources_list}
            """)
        
        # 8. Hashtag instructions
        if add_hashtags:
            hashtags = self._create_hashtags(topic, graph_summary)
            hashtag_str = " ".join(hashtags)
            prompt_parts.append(f"""
            HASHTAG REQUIREMENTS:
            Include these relevant hashtags after the main content: {hashtag_str}
            """)
        
        # Final prompt assembly
        full_prompt = "\n".join(prompt_parts)
        
        # Generate the script based on the selected model
        logger.info(f"Generating {content_type} script for topic: {topic}")
        
        if self.model_name.lower() == "gemini":
            script = self._generate_with_gemini(full_prompt, self.max_retries)
        else:
            logger.error(f"Unsupported model: {self.model_name}")
            script = f"Error: Unsupported model {self.model_name}"
            
        if script:
            logger.info(f"Successfully generated {content_type} script ({len(script)} characters)")
            # Count words
            word_count = len(script.split())
            logger.info(f"Successfully generated script with {word_count} words")
        else:
            logger.error("Failed to generate script")
            
        return script, metadata

def generate_content_script(topic: str, content_type: str = "Post", tone: str = "Professional", 
                          target_audience: str = "General", max_length: int = 500) -> str:
    """
    Generate a content script with minimal parameters.
    
    Args:
        topic: The topic to generate content about
        content_type: Type of content to generate
        tone: Tone of the content
        target_audience: Target audience for the content
        max_length: Maximum length in words
        
    Returns:
        Generated content script
    """
    try:
        generator = ScriptGenerator()
        input_data = {
            "topic": topic,
            "content_type": content_type,
            "tone": tone,
            "target_audience": target_audience,
            "max_length": max_length,
        }
        
        script, _ = generator.generate(input_data)
        logger.info(f"Successfully generated {content_type} script with {len(script.split())} words")
        return script
    except Exception as e:
        logger.error(f"Error generating content script: {e}")
        return f"Error generating content: {str(e)}"

def generate_multi_angle_script(
    topic: str,
    graph: nx.DiGraph = None,
    documents: List[Dict[str, Any]] = None,
    tone: str = "Conversational",
    format_style: str = "Educational",
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a multi-angle script with multiple content approaches and a final full script.
    This approach combines knowledge graph analysis with a structured prompt technique.
    
    Args:
        topic: The main topic for the content
        graph: The knowledge graph (optional)
        documents: List of document dictionaries (optional)
        tone: Tone of the content (Conversational, Entertaining, etc.)
        format_style: Format style (Educational, Vlog, etc.)
        api_key: API key for external model (optional)
        
    Returns:
        Dictionary containing angles, script snippets, evaluation, and full script
    """
    try:
        logger.info(f"Generating multi-angle script for topic: {topic}")
        
        # Extract insights from knowledge graph if available
        kg_insights = {}
        if graph and documents:
            # Extract key entities and concepts from the graph
            entities = extract_key_entities(graph, limit=15)
            concepts = extract_key_concepts(graph, limit=12)
            excerpts = extract_document_excerpts(documents, limit=8)
            
            # Structure the insights
            kg_insights = {
                "key_entities": [e["name"] for e in entities],
                "key_concepts": [c["name"] for c in concepts],
                "key_excerpts": [e["text"] for e in excerpts]
            }
        
        # Craft the prompt with knowledge graph insights if available
        prompt = craft_multi_angle_prompt(topic, tone, format_style, kg_insights)
        
        # Generate the multi-angle script using an external model
        result = generate_with_external_model(prompt, api_key)
        
        # Process the result
        output = process_multi_angle_result(result)
        
        # Add the knowledge graph insights to the output
        if kg_insights:
            output["knowledge_graph_insights"] = kg_insights
        
        logger.info(f"Successfully generated multi-angle script for topic: {topic}")
        return output
    
    except Exception as e:
        logger.error(f"Error generating multi-angle script: {e}")
        return {
            "error": str(e),
            "angles": [],
            "scripts": [],
            "evaluation": "Error occurred during generation",
            "full_script": ""
        }

def craft_multi_angle_prompt(
    topic: str, 
    tone: str, 
    format_style: str, 
    kg_insights: Dict[str, List[str]] = None
) -> str:
    """
    Craft a prompt for multi-angle script generation, incorporating knowledge graph insights if available.
    
    Args:
        topic: The main topic
        tone: The tone for the content
        format_style: The format style
        kg_insights: Insights from the knowledge graph (optional)
        
    Returns:
        Prompt string for the external model
    """
    # Base prompt
    prompt = f"""
You are an expert YouTube content strategist and scriptwriter.

The user wants to create a video with the following preferences:

Topic: "{topic}"  
Preferred Tone: "{tone}"  
Format Style: "{format_style}"  
Audience: General public (YouTube viewers, short attention span, high curiosity)
"""

    # Add knowledge graph insights if available
    if kg_insights and kg_insights.get("key_entities") and kg_insights.get("key_concepts"):
        prompt += f"""
Based on analysis of relevant documents, these are the key entities and concepts to consider:

Key Entities: {', '.join(kg_insights.get('key_entities', [])[:10])}
Key Concepts: {', '.join(kg_insights.get('key_concepts', [])[:10])}
"""

        # Add excerpts if available
        if kg_insights.get("key_excerpts"):
            prompt += f"""
Some relevant excerpts from source materials:
- {kg_insights.get('key_excerpts', [])[0] if kg_insights.get('key_excerpts') else ''}
- {kg_insights.get('key_excerpts', [])[1] if len(kg_insights.get('key_excerpts', [])) > 1 else ''}
"""

    # Continue with structured tasks
    prompt += """
Your task is as follows:

---

**Step 1: Generate 10 unique content angles**  
Come up with 10 different storylines or perspectives to explore this topic. Each angle should be creative, distinct, and suitable for a YouTube video in the given tone and format.

Return them as a numbered list, e.g.:
1. ...
2. ...
...

---

**Step 2: Write a short intro (2–3 paragraphs) for each angle**  
For each angle, write a compelling video intro using the provided tone and format.  
It should include a hook, context, and lead into what the video would explain.

Label each one clearly like:
### Script 1:
...  
### Script 2:
...

---

**Step 3: Evaluate all 10 intros**  
Compare all 10 generated intros and pick the best one based on:
- Creativity
- Emotional or intellectual hook
- Relevance to current events or audience interest
- Retention potential on YouTube

State clearly:
✅ Best Script: `#N`
Reason: ...

---

**Step 4: Expand the best intro into a full 3–4 minute YouTube script**  
Structure it like this:
- Hook (first 10 seconds)
- Intro (set the stage)
- Body (key points with transitions)
- Outro (conclusion and takeaway)
- Call-to-action (like, comment, subscribe)

Add cues if needed like [pause], [clip starts], [zoom in].

Make sure it's emotionally engaging, accurate, and flows naturally.

---

Output everything in clean sections:
- 🔢 Angles List
- ✍️ All 10 Short Scripts
- 🏆 Evaluation Result
- 🧠 Final Full Script
"""

    return prompt

def generate_with_external_model(prompt: str, api_key: Optional[str] = None) -> str:
    """
    Generate content using an external model (e.g., Gemini, GPT).
    
    Args:
        prompt: The prompt to send to the model
        api_key: API key for the external model (optional)
        
    Returns:
        Generated text
    """
    try:
        # Check if Gemini is available
        try:
            import google.generativeai as genai
            
            # Configure the API
            if api_key:
                genai.configure(api_key=api_key)
            
            # Setup the model
            generation_config = {
                "temperature": 0.8,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
            }
            
            model = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                generation_config=generation_config
            )
            
            # Generate content
            response = model.generate_content(prompt)
            return response.text
        
        except ImportError:
            logger.warning("Google GenerativeAI not available, trying fallback...")
            
            # Try OpenAI fallback
            try:
                import openai
                
                # Configure the API
                if api_key:
                    openai.api_key = api_key
                
                # Generate content
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an expert YouTube content strategist."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.8,
                    max_tokens=4000
                )
                
                return response.choices[0].message.content
            
            except (ImportError, Exception) as e:
                logger.error(f"OpenAI fallback failed: {e}")
                
                # Return a simple template as last resort
                return f"""
🔢 Angles List
1. The History and Evolution of {prompt.split('"')[1]}
2. Common Misconceptions About {prompt.split('"')[1]}
3. How {prompt.split('"')[1]} Impacts Daily Life
4. Future Trends in {prompt.split('"')[1]}
5. The Science Behind {prompt.split('"')[1]}
6. Notable Figures in {prompt.split('"')[1]}
7. {prompt.split('"')[1]} Around the World
8. Ethical Considerations of {prompt.split('"')[1]}
9. How to Get Started with {prompt.split('"')[1]}
10. The Economic Impact of {prompt.split('"')[1]}

✍️ All 10 Short Scripts
### Script 1:
[Sample introduction text would go here]

🏆 Evaluation Result
✅ Best Script: #3
Reason: This angle has the strongest hook and relates most directly to the viewer's life.

🧠 Final Full Script
[Hook]
Have you ever wondered how {prompt.split('"')[1]} affects your daily routine? Let's explore this fascinating topic.

[Content would continue...]
"""
    
    except Exception as e:
        logger.error(f"Error generating with external model: {e}")
        return f"Error generating content: {str(e)}"

def process_multi_angle_result(result: str) -> Dict[str, Any]:
    """
    Process the result from the external model into a structured format.
    
    Args:
        result: The text result from the external model
        
    Returns:
        Dictionary with structured content
    """
    try:
        # Extract the angles list
        angles = []
        if "🔢 Angles List" in result:
            angles_section = result.split("🔢 Angles List")[1].split("✍️")[0].strip()
            for line in angles_section.split("\n"):
                if line.strip() and line.strip()[0].isdigit():
                    angles.append(line.strip())
        
        # Extract the scripts
        scripts = []
        if "✍️ All 10 Short Scripts" in result:
            scripts_section = result.split("✍️ All 10 Short Scripts")[1].split("🏆")[0].strip()
            current_script = ""
            current_script_num = None
            
            for line in scripts_section.split("\n"):
                if line.strip().startswith("### Script"):
                    if current_script and current_script_num:
                        scripts.append({"number": current_script_num, "content": current_script.strip()})
                    current_script_num = line.strip().replace("### Script ", "").replace(":", "")
                    current_script = ""
                else:
                    current_script += line + "\n"
            
            # Add the last script
            if current_script and current_script_num:
                scripts.append({"number": current_script_num, "content": current_script.strip()})
        
        # Extract the evaluation
        evaluation = ""
        if "🏆 Evaluation Result" in result:
            evaluation = result.split("🏆 Evaluation Result")[1].split("🧠")[0].strip()
        
        # Extract the full script
        full_script = ""
        if "🧠 Final Full Script" in result:
            full_script = result.split("🧠 Final Full Script")[1].strip()
        
        return {
            "angles": angles,
            "scripts": scripts,
            "evaluation": evaluation,
            "full_script": full_script
        }
    
    except Exception as e:
        logger.error(f"Error processing multi-angle result: {e}")
        return {
            "angles": [],
            "scripts": [],
            "evaluation": "Error occurred while processing the result",
            "full_script": result  # Return the full result as is
        }

def extract_key_entities(graph: nx.DiGraph, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Extract key entities from the knowledge graph.
    
    Args:
        graph: Knowledge graph
        limit: Maximum number of entities to extract
        
    Returns:
        List of entity dictionaries
    """
    entities = []
    
    # Filter nodes that are entities (not documents or concepts)
    entity_nodes = [
        (node_id, node_data) 
        for node_id, node_data in graph.nodes(data=True) 
        if node_data.get('type', '') in ['entity', 'person', 'organization', 'location', 'product']
    ]
    
    # Sort by degree centrality (number of connections)
    entity_nodes.sort(key=lambda x: graph.degree(x[0]), reverse=True)
    
    # Take top entities
    for node_id, node_data in entity_nodes[:limit]:
        entity = {
            'id': node_id,
            'name': node_data.get('name', str(node_id)),
            'type': node_data.get('type', 'entity'),
            'degree': graph.degree(node_id),
            'connections': list(graph.neighbors(node_id))
        }
        entities.append(entity)
    
    return entities

def extract_key_concepts(graph: nx.DiGraph, limit: int = 8) -> List[Dict[str, Any]]:
    """
    Extract key concepts from the knowledge graph.
    
    Args:
        graph: Knowledge graph
        limit: Maximum number of concepts to extract
        
    Returns:
        List of concept dictionaries
    """
    concepts = []
    
    # Filter nodes that are concepts
    concept_nodes = [
        (node_id, node_data) 
        for node_id, node_data in graph.nodes(data=True) 
        if node_data.get('type', '') in ['concept', 'keyword', 'topic', 'subject']
    ]
    
    # Sort by degree centrality (number of connections)
    concept_nodes.sort(key=lambda x: graph.degree(x[0]), reverse=True)
    
    # Take top concepts
    for node_id, node_data in concept_nodes[:limit]:
        concept = {
            'id': node_id,
            'name': node_data.get('name', str(node_id)),
            'type': node_data.get('type', 'concept'),
            'degree': graph.degree(node_id),
            'connections': list(graph.neighbors(node_id))
        }
        concepts.append(concept)
    
    return concepts

def extract_document_excerpts(documents: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    """
    Extract key excerpts from documents.
    
    Args:
        documents: List of document dictionaries
        limit: Maximum number of excerpts to extract
        
    Returns:
        List of excerpt dictionaries
    """
    excerpts = []
    
    # Process each document
    for doc in documents[:limit]:
        # Get document content
        content = doc.get('content', doc.get('page_content', ''))
        
        if not content:
            continue
            
        # Get document metadata
        metadata = doc.get('metadata', {})
        source = metadata.get('source', 'unknown')
        
        # Extract a relevant excerpt (simplified version)
        # In a real system, this would use more sophisticated extraction techniques
        sentences = content.split('.')
        excerpt = '.'.join(sentences[:2]) + '.' if sentences else content[:200]
        
        # Add excerpt
        excerpts.append({
            'text': excerpt,
            'source': source,
            'doc_id': metadata.get('id', '')
        })
    
    return excerpts

def determine_script_structure(content_type: str, length: str) -> List[str]:
    """
    Determine the structure of the script based on content type and length.
    
    Args:
        content_type: Type of content
        length: Content length
        
    Returns:
        List of script parts
    """
    # Default structure
    structure = ["introduction", "main_points", "conclusion"]
    
    if content_type == "Blog Post":
        if length in ["Long", "Very Long"]:
            structure = ["introduction", "background", "main_points", "examples", "analysis", "conclusion"]
        elif length == "Medium":
            structure = ["introduction", "background", "main_points", "analysis", "conclusion"]
        else:  # Short or Very Short
            structure = ["introduction", "main_points", "conclusion"]
    
    elif content_type == "Social Media Post":
        if length in ["Very Short", "Short"]:
            structure = ["hook", "main_point", "call_to_action"]
        else:
            structure = ["hook", "context", "main_points", "call_to_action"]
    
    elif content_type == "Video Script":
        if length in ["Long", "Very Long"]:
            structure = ["intro_hook", "greeting", "topic_introduction", "main_points", "examples", "summary", "call_to_action", "outro"]
        elif length == "Medium":
            structure = ["intro_hook", "greeting", "topic_introduction", "main_points", "summary", "call_to_action"]
        else:  # Short or Very Short
            structure = ["intro_hook", "main_points", "call_to_action"]
    
    elif content_type == "Educational Content":
        if length in ["Long", "Very Long"]:
            structure = ["learning_objectives", "introduction", "background", "key_concepts", "examples", "practice", "summary", "further_reading"]
        elif length == "Medium":
            structure = ["learning_objectives", "introduction", "key_concepts", "examples", "summary"]
        else:  # Short or Very Short
            structure = ["introduction", "key_concepts", "summary"]
    
    elif content_type == "Newsletter":
        if length in ["Long", "Very Long"]:
            structure = ["greeting", "headline", "main_story", "secondary_stories", "industry_updates", "tips", "upcoming_events", "conclusion"]
        elif length == "Medium":
            structure = ["greeting", "headline", "main_story", "secondary_stories", "tips", "conclusion"]
        else:  # Short or Very Short
            structure = ["greeting", "headline", "main_story", "conclusion"]
    
    return structure

def generate_title(topic: str, content_type: str, tone: str) -> str:
    """
    Generate a title for the content script.
    
    Args:
        topic: Main topic
        content_type: Type of content
        tone: Content tone
        
    Returns:
        Generated title
    """
    # Title templates based on content type and tone
    templates = {
        "Blog Post": {
            "Informative": [
                f"Understanding {topic}: A Comprehensive Guide",
                f"The Complete Guide to {topic}",
                f"Everything You Need to Know About {topic}"
            ],
            "Persuasive": [
                f"Why {topic} Matters More Than Ever",
                f"How {topic} Can Transform Your Perspective",
                f"The Undeniable Impact of {topic}"
            ],
            "Conversational": [
                f"Let's Talk About {topic}",
                f"{topic}: What's the Big Deal?",
                f"Exploring {topic} Together"
            ],
            "Professional": [
                f"{topic}: A Strategic Analysis",
                f"The Business Implications of {topic}",
                f"A Professional's Guide to {topic}"
            ],
            "Entertaining": [
                f"The Fascinating World of {topic}",
                f"Surprising Facts About {topic} You Never Knew",
                f"{topic}: An Unexpected Journey"
            ]
        },
        "Social Media Post": {
            "Informative": [f"Key Insights on {topic}", f"What to Know About {topic}"],
            "Persuasive": [f"Why You Should Care About {topic}", f"{topic} - Act Now!"],
            "Conversational": [f"Thoughts on {topic}?", f"Let's Discuss {topic}"],
            "Professional": [f"{topic}: Industry Perspectives", f"Professional Take on {topic}"],
            "Entertaining": [f"Fun Facts About {topic}", f"{topic} - You Won't Believe This!"]
        },
        "Video Script": {
            "Informative": [f"{topic} Explained", f"Understanding {topic} in Minutes"],
            "Persuasive": [f"Why {topic} Should Matter to You", f"The Power of {topic}"],
            "Conversational": [f"Let's Chat About {topic}", f"{topic} - My Thoughts"],
            "Professional": [f"{topic} - A Deep Dive", f"Professional Analysis: {topic}"],
            "Entertaining": [f"{topic}: Surprising Revelations", f"The Amazing World of {topic}"]
        }
    }
    
    # Use default templates if specific ones not found
    default_templates = [
        f"The Ultimate Guide to {topic}",
        f"Exploring {topic} in Depth",
        f"{topic}: What You Need to Know"
    ]
    
    # Get templates for the content type and tone
    content_templates = templates.get(content_type, {})
    tone_templates = content_templates.get(tone, default_templates)
    
    # Return a random template
    return random.choice(tone_templates)

def generate_script_part(
    part: str,
    topic: str,
    entities: List[Dict[str, Any]],
    concepts: List[Dict[str, Any]],
    excerpts: List[Dict[str, Any]],
    tone: str,
    target_audience: str
) -> str:
    """
    Generate a part of the script.
    
    Args:
        part: Part of the script to generate
        topic: Main topic
        entities: Key entities
        concepts: Key concepts
        excerpts: Document excerpts
        tone: Content tone
        target_audience: Target audience
        
    Returns:
        Generated script part
    """
    # Simplified generator for demonstration purposes
    # In a real implementation, this would use more sophisticated NLG techniques
    
    part_title = part.replace('_', ' ').title()
    content = f"## {part_title}\n\n"
    
    if part == "introduction":
        content += f"Welcome to this {tone.lower()} exploration of {topic}. "
        
        if concepts:
            content += f"We'll be covering key concepts like {', '.join([c['name'] for c in concepts[:3]])}, "
            content += f"and examining how they relate to our understanding of {topic}."
        
        if excerpts:
            content += f" As {excerpts[0]['source']} notes, \"{excerpts[0]['text']}\""
    
    elif part == "main_points":
        content += f"Let's explore the key aspects of {topic}:\n\n"
        
        for i, concept in enumerate(concepts[:5], 1):
            content += f"### {i}. {concept['name']}\n"
            
            related_entities = [e['name'] for e in entities if random.random() > 0.5][:2]
            if related_entities:
                content += f"{concept['name']} is closely connected to {' and '.join(related_entities)}. "
            
            # Add an excerpt if available
            if i < len(excerpts):
                content += f"According to {excerpts[i]['source']}, \"{excerpts[i]['text']}\"\n\n"
            else:
                content += "\n\n"
    
    elif part == "conclusion":
        content += f"In conclusion, {topic} represents a fascinating area with many important implications. "
        
        if concepts:
            content += f"We've explored key concepts including {', '.join([c['name'] for c in concepts[:3]])}, "
            content += f"and seen how they interconnect to form our understanding of this subject."
        
        content += f" As our knowledge continues to evolve, {topic} will undoubtedly remain an important area of focus."
    
    elif part == "call_to_action":
        if tone == "Persuasive":
            content += f"Now is the time to act on {topic}. Share your thoughts in the comments below and "
            content += f"join the conversation about this important subject."
        else:
            content += f"What are your thoughts on {topic}? Let me know in the comments section below "
            content += f"and don't forget to subscribe for more content like this."
    
    else:
        # Generic content for other parts
        content += f"This section explores {part.replace('_', ' ')} related to {topic}. "
        
        if entities:
            content += f"Key elements include {', '.join([e['name'] for e in entities[:3]])}. "
        
        if excerpts and len(excerpts) > 2:
            content += f"As noted by {excerpts[2]['source']}, \"{excerpts[2]['text']}\""
    
    return content 