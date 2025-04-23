"""
SocialSynth-AI: Advanced RAG System with Knowledge Graph
=====================================================

Created by: Raza Abbas
Version: 1.0
License: MIT

Project Overview
----------------
SocialSynth-AI is a state-of-the-art content generation system that utilizes advanced Retrieval Augmented Generation (RAG) 
techniques with knowledge graph-based semantic reasoning to produce high-quality, factual, and engaging content. The system 
can generate scripts for various formats including documentaries, news reports, dramas, comedies, and educational content.

This document provides a technical explanation of the system architecture, components, and data flow, along with
implementation details and rationale behind technical decisions.

System Architecture
------------------
The system follows a modular architecture with the following main components:

1. Enhanced Retriever Module:
   - Multi-source data collection (YouTube, News, Blogs)
   - Relevance scoring and filtering
   - Entity extraction and metadata enrichment

2. Knowledge Graph Builder Module:
   - Entity recognition and relationship extraction
   - Graph construction and semantic network building
   - Visualization and graph analytics

3. RAG Processing Pipeline:
   - Query expansion and hybrid retrieval
   - Context selection and reranking
   - LLM prompting with structured context

4. Content Generation Module:
   - Script formatting with customizable parameters
   - Tone and style adaptation
   - Source attribution and context integration

Data Flow Architecture
---------------------
1. User Input → Query Processing
2. Enhanced Retrieval → Document Collection
3. Knowledge Graph Construction
4. Context Selection and Reranking
5. LLM-based Content Generation
6. Visualization and Results Presentation

Technical Implementation
-----------------------

Enhanced Retriever (enhanced_rag_retriever.py)
----------------------------------------------
The Enhanced Retriever is responsible for collecting diverse, relevant data from multiple sources and 
preprocessing it for the RAG system.

Key Features:
- Multi-source retrieval (YouTube, News, Blogs)
- Semantic relevance scoring using embeddings
- Entity extraction and metadata enrichment
- Customizable relevance thresholds and result limits

Implementation Rationale:
- Why multiple sources? Content diversity improves factual coverage and perspective balance
- Why embed relevance scoring? To filter out irrelevant content before the expensive LLM processing
- Why entity extraction? To enable semantic connections in the knowledge graph

Usage Example:
```python
from enhanced_rag_retriever import EnhancedRetriever

retriever = EnhancedRetriever(
    youtube_api_key="YOUR_KEY",
    news_api_key="YOUR_KEY",
    max_results_per_source=5,
    min_relevance_score=0.6
)

documents = retriever.get_relevant_documents("AI advancements in healthcare")
```

Knowledge Graph Builder (knowledge_graph_builder.py)
---------------------------------------------------
The Knowledge Graph Builder transforms retrieved documents into a semantic network that captures
entities, relationships, and contextual connections between information sources.

Key Features:
- Named entity recognition using SpaCy
- Relationship extraction using LLM
- Graph construction with NetworkX
- Interactive visualization with PyVis
- Graph analytics and statistics

Implementation Rationale:
- Why knowledge graphs? They provide semantic structure beyond vector embeddings
- Why use LLMs for relationship extraction? To capture complex semantic relationships
- Why interactive visualization? To enable intuitive exploration of information connections

Usage Example:
```python
from knowledge_graph_builder import KnowledgeGraphBuilder

graph_builder = KnowledgeGraphBuilder()
graph = graph_builder.build_from_documents(documents, query="AI healthcare")
graph_builder.visualize("knowledge_graph.html")
```

Advanced RAG System (advanced_rag_v2.py)
----------------------------------------
The core RAG system enhances traditional retrieval with graph-based context expansion and
hybrid retrieval methods.

Key Features:
- Query expansion for better recall
- Hybrid retrieval (dense + sparse)
- Graph traversal for context enrichment
- Entity-based retrieval enhancement
- Contextual reranking

Implementation Rationale:
- Why hybrid retrieval? Different retrieval methods have complementary strengths
- Why query expansion? To capture semantic variations of the user's intent
- Why graph-based context? To find related information beyond lexical similarity

Usage Example:
```python
from advanced_rag_v2 import EnhancedKnowledgeGraph, ContextualReranker

knowledge_graph = EnhancedKnowledgeGraph(embedding_function)
results = knowledge_graph.retrieve(query="medical AI applications", k=8, use_hybrid=True)
```

Script Generator (updated_youtube_script_generator.py)
-----------------------------------------------------
The Script Generator takes the enriched context and transforms it into coherent, engaging content
with specified parameters.

Key Features:
- Tone customization (educational, entertaining, professional, etc.)
- Length adjustment (short, medium, long)
- Source attribution and fact grounding
- Template-based script structure
- Visualization of knowledge sources

Implementation Rationale:
- Why use structured prompts? To guide the LLM toward specific content formats
- Why source attribution? To ensure factual accuracy and traceability
- Why customizable parameters? To adapt to different content needs and audience preferences

Usage Example:
```python
from youtube_script_generator import generate_script

result = generate_script(
    query="AI in medicine",
    tone="educational",
    length="medium",
    knowledge_graph=knowledge_graph,
    reranker=reranker,
    llm=llm
)
```

Enhanced Knowledge Graph (enhanced_knowledge_graph.py)
------------------------------------------------------
An alternative knowledge graph implementation focused on visual exploration and analytics.

Key Features:
- Entity type categorization and coloring
- Relationship extraction and visualization
- Graph metrics and analytics
- Data export in multiple formats

Implementation Rationale:
- Why visual encoding? To make complex relationships more intuitive
- Why graph analytics? To identify key entities and information patterns
- Why multiple export formats? To enable integration with other tools and systems

Usage Example:
```python
from enhanced_knowledge_graph import EnhancedKnowledgeGraph

graph_builder = EnhancedKnowledgeGraph()
html_path = graph_builder.generate_knowledge_graph(documents, query)
```

Technical Design Decisions
-------------------------

1. Embedding Models:
   - Using sentence transformers for dense retrieval
   - Why? Better semantic understanding compared to traditional methods
   - Implementation: GoogleGenerativeAIEmbeddings for production quality

2. LLM Selection:
   - Using Google's Gemini models (with fallback options)
   - Why? Balance of performance, cost, and capability
   - Implementation: Dynamic model selection based on availability

3. Hybrid Retrieval:
   - Combining BM25 (sparse) with embeddings (dense)
   - Why? Complementary strengths (lexical precision + semantic understanding)
   - Implementation: Weighted combination with tunable parameters

4. Entity Extraction:
   - Using SpaCy for NER + custom processors
   - Why? Balance of speed and accuracy for real-time processing
   - Implementation: Entity type categorization and filtering

5. Knowledge Graph Construction:
   - NetworkX for graph processing + PyVis for visualization
   - Why? Flexible graph operations with interactive visualization
   - Implementation: Multi-layered graph with entity-relationship modeling

Performance Optimization
-----------------------
1. Batched Processing:
   - Embedding generation in batches to prevent memory issues
   - Document processing parallelization where possible

2. Caching:
   - Document and embedding caching to prevent redundant processing
   - Session state management for interactive applications

3. Resource Management:
   - Text truncation for LLM processing to manage token usage
   - Selective relationship extraction to balance detail and performance

Installation and Setup
---------------------
1. Install required packages:
   ```
   pip install -r requirements.txt
   ```

2. Set up environment variables (create .env file):
   ```
   GOOGLE_API_KEY=your_key_here
   YOUTUBE_API_KEY=your_key_here
   NEWS_API_KEY=your_key_here
   ```

3. Download required models:
   ```
   python -m spacy download en_core_web_sm
   ```

4. Run the application:
   ```
   python main.py
   ```

Advanced Usage Scenarios
-----------------------
1. Content Marketing:
   - Generate factual, engaging marketing content with source attribution
   - Create scripts for product demonstrations and educational content

2. Education:
   - Develop educational scripts with verified information
   - Create explanatory content with diverse source integration

3. Media Production:
   - Generate initial scripts for documentary production
   - Create news summaries with comprehensive context and sources

4. Research:
   - Explore topic connections through knowledge graph visualization
   - Identify key entities and relationships in complex domains

Development Roadmap
------------------
1. Integration with additional data sources
2. Advanced relationship extraction with domain-specific models
3. Temporal analysis and trend detection
4. Multi-language support
5. User feedback integration for continuous improvement

Conclusion
----------
SocialSynth-AI represents a significant advancement in content generation through its integration of
knowledge graph-based semantic reasoning with advanced RAG techniques. The system demonstrates how
structured information representation can enhance generative AI applications, producing content that
is not only engaging but also factually grounded and contextually rich.

By Raza Abbas
""" 