import networkx as nx
import spacy
from sentence_transformers import SentenceTransformer, util
from pyvis.network import Network
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from langchain.schema import Document
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
import logging
from collections import defaultdict
import numpy as np
import os
from dotenv import load_dotenv
import json
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import time

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Load SpaCy for entity extraction
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class QueryExpansion:
    """Class for expanding queries to improve retrieval performance"""
    
    def __init__(self):
        """Initialize query expansion model"""
        try:
            self.model_name = "google/flan-t5-base"  # Smaller but effective model
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            logger.info(f"Query expansion model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Error loading query expansion model: {e}")
            self.model = None
            self.tokenizer = None
    
    def expand_query(self, query: str) -> List[str]:
        """Generate alternative phrasings of the query"""
        if not self.model or not self.tokenizer:
            logger.warning("Query expansion model not available, using original query")
            return [query]
            
        try:
            input_text = f"rephrase: {query}"
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            
            # Generate 3 alternative formulations
            outputs = []
            for _ in range(3):
                output = self.model.generate(
                    input_ids, 
                    max_length=50,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True
                )
                decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
                if decoded not in outputs and decoded != query:
                    outputs.append(decoded)
            
            # Always include original query
            if query not in outputs:
                outputs.append(query)
                
            return outputs
        except Exception as e:
            logger.error(f"Error in query expansion: {e}")
            return [query]  # Fall back to original query

class HybridRetriever:
    """Hybrid retrieval combining dense and sparse methods"""
    
    def __init__(self, embedding_function, documents: Optional[List[Document]] = None):
        """Initialize hybrid retriever"""
        self.embedding_function = embedding_function
        self.documents = documents or []
        self.document_embeddings = []
        self.document_texts = []
        self.bm25 = None
        self.tokenized_corpus = []
        self.stop_words = set(stopwords.words('english'))
        
        if documents:
            self.index_documents(documents)
    
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for BM25 indexing"""
        # Lowercase, tokenize and remove stopwords
        tokens = word_tokenize(text.lower())
        return [token for token in tokens if token.isalnum() and token not in self.stop_words]
    
    def index_documents(self, documents: List[Document]):
        """Index documents for both dense and sparse retrieval"""
        self.documents = documents
        self.document_texts = [doc.page_content for doc in documents]
        
        # Sparse indexing (BM25)
        self.tokenized_corpus = [self.preprocess_text(text) for text in self.document_texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # Dense indexing (embeddings)
        logger.info(f"Creating dense embeddings for {len(documents)} documents")
        self.document_embeddings = []
        
        # Process in batches to avoid memory issues
        batch_size = 32
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            batch_texts = [doc.page_content for doc in batch]
            try:
                batch_embeddings = [self.embedding_function.embed_query(text) for text in batch_texts]
                self.document_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error creating embeddings for batch {i//batch_size}: {e}")
                # Fill with zeros for failed embeddings
                self.document_embeddings.extend([np.zeros(768) for _ in range(len(batch))])
    
    def retrieve(self, query: str, k: int = 5, alpha: float = 0.7) -> List[Tuple[Document, float]]:
        """Hybrid retrieval combining dense and sparse scores"""
        if not self.documents:
            return []
            
        try:
            # Dense retrieval
            query_embedding = self.embedding_function.embed_query(query)
            dense_scores = []
            
            for i, doc_embedding in enumerate(self.document_embeddings):
                similarity = util.cos_sim(
                    np.array(query_embedding).reshape(1, -1),
                    np.array(doc_embedding).reshape(1, -1)
                )[0][0]
                dense_scores.append((i, float(similarity)))
                
            # Sparse retrieval (BM25)
            tokenized_query = self.preprocess_text(query)
            if not tokenized_query:  # Handle empty query after preprocessing
                tokenized_query = [""]  # Dummy token to prevent errors
                
            sparse_scores = [(i, score) for i, score in enumerate(self.bm25.get_scores(tokenized_query))]
            
            # Normalize scores
            max_dense = max(score for _, score in dense_scores) if dense_scores else 1.0
            max_sparse = max(score for _, score in sparse_scores) if sparse_scores else 1.0
            
            normalized_dense = [(i, score/max_dense) for i, score in dense_scores]
            normalized_sparse = [(i, score/max_sparse) for i, score in sparse_scores]
            
            # Combine scores with weighting factor alpha
            combined_scores = []
            for i in range(len(self.documents)):
                dense_score = next((score for idx, score in normalized_dense if idx == i), 0)
                sparse_score = next((score for idx, score in normalized_sparse if idx == i), 0)
                
                # Weighted combination
                hybrid_score = (alpha * dense_score) + ((1 - alpha) * sparse_score)
                combined_scores.append((i, hybrid_score))
            
            # Get top-k results
            top_results = []
            for i, score in sorted(combined_scores, key=lambda x: x[1], reverse=True)[:k]:
                top_results.append((self.documents[i], score))
                
            return top_results
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {e}")
            return []

class EnhancedKnowledgeGraph(nx.Graph):
    """Enhanced knowledge graph for advanced RAG retrieval"""
    
    def __init__(self, embedding_function):
        super().__init__()
        self.embedding_function = embedding_function
        self.document_ids = {}  # Map document text hash to graph ID
        self.entity_ids = {}    # Map entity name to graph ID
        self.entity_embeddings = {}  # Cache entity embeddings
        
        # Add hybrid retriever
        self.hybrid_retriever = HybridRetriever(embedding_function)
        
        # Add query expansion
        self.query_expander = QueryExpansion()
        
        # Document metadata storage
        self.doc_metadata = {}
    
    def add_documents(self, documents: List[Document]):
        """Process documents and add to knowledge graph with metadata preservation"""
        if not documents:
            return
            
        # Update hybrid retriever
        self.hybrid_retriever.index_documents(documents)
        
        # Add documents to graph
        for doc in documents:
            # Skip empty documents
            if not doc.page_content.strip():
                continue
                
            # Create unique document ID based on content hash
            doc_id = f"doc_{hash(doc.page_content)}"
            
            # Add document node if not exists
            if doc_id not in self.document_ids:
                # Get document embedding
                try:
                    doc_embedding = self.embedding_function.embed_query(doc.page_content)
                    
                    # Add document node with properties
                    self.add_node(
                        doc_id, 
                        type="document",
                        content=doc.page_content,
                        metadata=doc.metadata,
                        embedding=doc_embedding
                    )
                    self.document_ids[doc_id] = doc.page_content
                    self.doc_metadata[doc_id] = doc.metadata
                    
                    # Extract entities and create connections
                    entities = self._extract_entities(doc.page_content)
                    
                    # Connect document to entities
                    for entity, entity_type in entities:
                        entity_id = f"entity_{entity.lower()}"
                        
                        # Add entity node if not exists
                        if entity_id not in self.entity_ids:
                            # Get entity embedding
                            entity_embedding = self.embedding_function.embed_query(entity)
                            self.entity_embeddings[entity_id] = entity_embedding
                            
                            # Add entity node
                            self.add_node(
                                entity_id,
                                type="entity",
                                name=entity,
                                entity_type=entity_type,
                                embedding=entity_embedding
                            )
                            self.entity_ids[entity_id] = entity
                        
                        # Connect document to entity
                        self.add_edge(
                            doc_id, 
                            entity_id, 
                            weight=1.0,
                            relationship="contains"
                        )
                    
                    # Connect to similar documents
                    self._connect_similar_documents(doc_id, doc_embedding)
                    
                except Exception as e:
                    logger.error(f"Error adding document to graph: {e}")
    
    def _extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """Extract named entities from text"""
        doc = nlp(text[:5000])  # Limit text length for processing
        entities = []
        
        # Extract named entities
        for ent in doc.ents:
            if ent.text.strip():
                entities.append((ent.text, ent.label_))
        
        # Extract key noun phrases if few entities found
        if len(entities) < 3:
            for chunk in doc.noun_chunks:
                if chunk.text.strip() and len(chunk.text.split()) <= 4:
                    entities.append((chunk.text, "NOUN_PHRASE"))
        
        return entities
    
    def _connect_similar_documents(self, doc_id: str, doc_embedding: List[float]):
        """Connect document to similar existing documents"""
        # Skip if no other documents
        if len(self.document_ids) <= 1:
            return
            
        # Find similar documents
        similarities = []
        for other_id in self.document_ids:
            if other_id != doc_id and other_id in self:
                other_embedding = self.nodes[other_id].get("embedding")
                if other_embedding:
                    # Calculate cosine similarity
                    similarity = util.cos_sim(
                        np.array(doc_embedding).reshape(1, -1),
                        np.array(other_embedding).reshape(1, -1)
                    )[0][0]
                    similarities.append((other_id, float(similarity)))
        
        # Connect to top 3 most similar documents
        for other_id, similarity in sorted(similarities, key=lambda x: x[1], reverse=True)[:3]:
            if similarity > 0.7:  # Only connect if similarity is high
                self.add_edge(
                    doc_id,
                    other_id,
                    weight=similarity,
                    relationship="similar_to"
                )
    
    def retrieve(self, query: str, k: int = 5, use_hybrid: bool = True) -> List[Document]:
        """Advanced graph-based retrieval with multiple strategies"""
        start_time = time.time()
        results = []
        try:
            # 1. Query expansion
            expanded_queries = self.query_expander.expand_query(query)
            logger.info(f"Expanded query into {len(expanded_queries)} variations")
            
            # 2. Hybrid retrieval (combining dense and sparse)
            all_candidates = []
            if use_hybrid:
                # Retrieve using hybrid for all expanded queries
                for expanded_query in expanded_queries:
                    hybrid_results = self.hybrid_retriever.retrieve(expanded_query, k=k)
                    all_candidates.extend(hybrid_results)
            
            # 3. Graph-based retrieval
            query_embedding = self.embedding_function.embed_query(query)
            
            # Extract entities from query for graph traversal
            query_entities = self._extract_entities(query)
            query_entity_ids = [f"entity_{entity[0].lower()}" for entity in query_entities]
            
            # Find documents connected to query entities
            entity_matches = set()
            for entity_id in query_entity_ids:
                if entity_id in self.entity_ids:
                    neighbors = list(nx.neighbors(self, entity_id))
                    for node in neighbors:
                        if node.startswith("doc_"):
                            entity_matches.add(node)
            
            # Get documents from entity matches
            for doc_id in entity_matches:
                if doc_id in self:
                    doc_content = self.nodes[doc_id].get("content", "")
                    doc_metadata = self.doc_metadata.get(doc_id, {})
                    # Add graph context to metadata
                    graph_context = self._get_graph_context(doc_id)
                    if graph_context:
                        doc_metadata = doc_metadata.copy()  # Make a copy to avoid modifying original
                        doc_metadata["graph_context"] = graph_context
                    # Add to candidates with high score (1.0)
                    all_candidates.append((Document(page_content=doc_content, metadata=doc_metadata), 1.0))
            
            # 4. Deduplication and aggregation
            seen_content = set()
            unique_candidates = []
            for doc, score in all_candidates:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_candidates.append((doc, score))
            
            # 5. Reranking using contextual relevance
            for doc, score in sorted(unique_candidates, key=lambda x: x[1], reverse=True)[:k]:
                # Further boost score by adding LLM-assisted contextual relevance
                results.append(doc)
            
            logger.info(f"Retrieved {len(results)} documents in {time.time() - start_time:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Error in graph retrieval: {e}")
            return []
    
    def _get_graph_context(self, doc_id: str) -> Dict:
        """Get contextual information from the graph for this document"""
        if doc_id not in self:
            return {}
        
        context = {
            "related_entities": [],
            "similar_documents": []
        }
        
        # Get connected entities
        for neighbor in nx.neighbors(self, doc_id):
            if neighbor.startswith("entity_"):
                entity_data = {
                    "name": self.nodes[neighbor].get("name", ""),
                    "type": self.nodes[neighbor].get("entity_type", "")
                }
                context["related_entities"].append(entity_data)
        
        # Get similar documents
        for neighbor in nx.neighbors(self, doc_id):
            if neighbor.startswith("doc_") and neighbor != doc_id:
                edge_data = self.edges[doc_id, neighbor]
                if edge_data.get("relationship") == "similar_to":
                    doc_data = {
                        "content_preview": self.nodes[neighbor].get("content", "")[:100] + "...",
                        "similarity": edge_data.get("weight", 0.0)
                    }
                    context["similar_documents"].append(doc_data)
        
        return context
    
    def visualize(self, output_path: str = "knowledge_graph.html"):
        """Generate interactive visualization of the knowledge graph"""
        try:
            # Create network
            net = Network(height="750px", width="100%", notebook=True, directed=False)
            
            # Add nodes
            for node, attrs in self.nodes(data=True):
                node_type = attrs.get("type", "unknown")
                
                if node_type == "document":
                    # Document node
                    title = attrs.get("content", "")[:150] + "..."
                    net.add_node(
                        node,
                        title=title,
                        color="#6495ED",  # Blue
                        size=20
                    )
                elif node_type == "entity":
                    # Entity node
                    entity_type = attrs.get("entity_type", "")
                    title = f"{attrs.get('name', '')} ({entity_type})"
                    
                    # Different colors based on entity type
                    color = "#FFA500"  # Default orange
                    if entity_type in ["PERSON", "ORG", "GPE"]:
                        color = "#9370DB"  # Purple
                    elif entity_type in ["DATE", "TIME", "MONEY", "PERCENT"]:
                        color = "#3CB371"  # Green
                    
                    net.add_node(
                        node,
                        title=title,
                        color=color,
                        size=15
                    )
            
            # Add edges
            for source, target, attrs in self.edges(data=True):
                relationship = attrs.get("relationship", "")
                weight = attrs.get("weight", 1.0)
                
                # Edge style based on relationship
                if relationship == "contains":
                    net.add_edge(source, target, title="contains", width=2)
                elif relationship == "similar_to":
                    # Width based on similarity strength
                    width = max(1, min(10, weight * 10))
                    net.add_edge(
                        source, 
                        target, 
                        title=f"similarity: {weight:.2f}",
                        width=width,
                        dashes=True
                    )
            
            # Configure physics
            net.repulsion(node_distance=150, spring_length=200)
            
            # Save
            net.save_graph(output_path)
            logger.info(f"Knowledge graph visualization saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error visualizing graph: {e}")
            return False

class ContextualReranker:
    """Advanced reranker that considers multiple factors for document relevance"""
    
    def __init__(self):
        """Initialize reranker with cross-encoder model"""
        try:
            model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Efficient cross-encoder
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            logger.info(f"Reranker model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Error loading reranker model: {e}")
            self.model = None
            self.tokenizer = None
    
    def rerank(self, query: str, documents: List[Document], top_k: Optional[int] = None) -> List[Document]:
        """Rerank documents using contextual relevance and metadata signals"""
        if not documents:
            return []
            
        if not self.model or not self.tokenizer:
            logger.warning("Reranker model not available, skipping reranking")
            return documents[:top_k] if top_k else documents
        
        try:
            # Prepare pairs for cross-encoder
            pairs = [(query, doc.page_content) for doc in documents]
            
            features = self.tokenizer.batch_encode_plus(
                pairs,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Score all pairs
            with torch.no_grad():
                scores = self.model(**features).logits.flatten().cpu().numpy()
            
            # Enhance scores with metadata signals
            enhanced_scores = []
            for i, (score, doc) in enumerate(zip(scores, documents)):
                # Start with cross-encoder score
                final_score = float(score)
                
                # Factor in metadata signals if available
                if hasattr(doc, 'metadata') and doc.metadata:
                    # Boost score for documents with graph context
                    if "graph_context" in doc.metadata:
                        graph_boost = 0.2
                        related_entities = doc.metadata["graph_context"].get("related_entities", [])
                        # More entities = more relevant
                        entity_boost = min(0.3, len(related_entities) * 0.05)
                        final_score += graph_boost + entity_boost
                    
                    # Recency boost for documents with dates
                    if "date" in doc.metadata:
                        # Add logic for recency boosting here
                        pass
                
                enhanced_scores.append((i, final_score))
            
            # Sort by enhanced score and return top-k
            reranked_indices = [i for i, _ in sorted(enhanced_scores, key=lambda x: x[1], reverse=True)]
            reranked_docs = [documents[i] for i in reranked_indices]
            
            if top_k:
                return reranked_docs[:top_k]
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            return documents[:top_k] if top_k else documents

def visualize_knowledge_graph(graph: EnhancedKnowledgeGraph, output_path: str = "knowledge_graph.html"):
    """Wrapper for graph visualization"""
    return graph.visualize(output_path)
