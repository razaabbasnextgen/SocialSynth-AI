import networkx as nx
import spacy
from sentence_transformers import SentenceTransformer
from pyvis.network import Network
import re
from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging
from collections import defaultdict
import numpy as np
from sentence_transformers import util

logger = logging.getLogger(__name__)

# Load SpaCy for entity extraction
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class KnowledgeGraph:
    """Knowledge graph for enhanced RAG retrieval"""
    
    def __init__(self, embedding_function):
        self.graph = nx.Graph()
        self.embedding_function = embedding_function
        self.document_nodes = {}  # Track document nodes by ID
        self.entity_nodes = {}    # Track entity nodes
        self.entity_embeddings = {}  # Cache entity embeddings
    
    def add_documents(self, documents: List[Document]):
        """Process documents and add to knowledge graph"""
        for doc in documents:
            # Skip empty documents
            if not doc.page_content.strip():
                continue
                
            # Create unique document ID
            doc_id = f"doc_{hash(doc.page_content)}"
            
            # Add document node if not exists
            if doc_id not in self.document_nodes:
                # Get document embedding
                try:
                    doc_embedding = self.embedding_function.embed_query(doc.page_content)
                    # Add document node with properties
                    self.graph.add_node(
                        doc_id, 
                        type="document",
                        content=doc.page_content,
                        metadata=doc.metadata,
                        embedding=doc_embedding
                    )
                    self.document_nodes[doc_id] = doc.page_content
                    
                    # Extract entities and create connections
                    entities = self._extract_entities(doc.page_content)
                    
                    # Connect document to entities
                    for entity, entity_type in entities:
                        entity_id = f"entity_{entity.lower()}"
                        
                        # Add entity node if not exists
                        if entity_id not in self.entity_nodes:
                            # Get entity embedding
                            entity_embedding = self.embedding_function.embed_query(entity)
                            self.entity_embeddings[entity_id] = entity_embedding
                            
                            # Add entity node
                            self.graph.add_node(
                                entity_id,
                                type="entity",
                                name=entity,
                                entity_type=entity_type,
                                embedding=entity_embedding
                            )
                            self.entity_nodes[entity_id] = entity
                        
                        # Connect document to entity
                        self.graph.add_edge(
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
        if len(self.document_nodes) <= 1:
            return
            
        # Find similar documents
        similarities = []
        for other_id in self.document_nodes:
            if other_id != doc_id and other_id in self.graph:
                other_embedding = self.graph.nodes[other_id].get("embedding")
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
                self.graph.add_edge(
                    doc_id,
                    other_id,
                    weight=similarity,
                    relationship="similar_to"
                )
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve relevant documents using graph-based retrieval"""
        try:
            # Get query embedding
            query_embedding = self.embedding_function.embed_query(query)
            
            # Extract entities from query
            query_entities = self._extract_entities(query)
            query_entity_ids = [f"entity_{entity[0].lower()}" for entity in query_entities]
            
            # Find relevant documents through direct entity matches
            entity_matches = set()
            for entity_id in query_entity_ids:
                if entity_id in self.entity_nodes:
                    neighbors = list(nx.neighbors(self.graph, entity_id))
                    for node in neighbors:
                        if node.startswith("doc_"):
                            entity_matches.add(node)
            
            # Find documents by embedding similarity
            sim_scores = []
            for doc_id in self.document_nodes:
                if doc_id in self.graph:
                    doc_embedding = self.graph.nodes[doc_id].get("embedding")
                    if doc_embedding:
                        similarity = util.cos_sim(
                            np.array(query_embedding).reshape(1, -1),
                            np.array(doc_embedding).reshape(1, -1)
                        )[0][0]
                        sim_scores.append((doc_id, float(similarity)))
            
            # Get top k similar documents
            similar_docs = [doc_id for doc_id, _ in sorted(sim_scores, key=lambda x: x[1], reverse=True)[:k]]
            
            # Multi-hop expansion: Explore neighbors of high-similarity docs
            if entity_matches:
                expanded_matches = set(entity_matches)
                for doc_id in entity_matches:
                    # Get similar documents (second hop)
                    for neighbor in nx.neighbors(self.graph, doc_id):
                        if neighbor.startswith("doc_"):
                            expanded_matches.add(neighbor)
                
                all_candidates = list(expanded_matches) + [d for d in similar_docs if d not in expanded_matches]
            else:
                all_candidates = similar_docs
            
            # Rank final candidates
            final_scores = []
            for doc_id in all_candidates:
                if doc_id in self.graph:
                    # Base score from embedding similarity
                    base_score = 0
                    for d, score in sim_scores:
                        if d == doc_id:
                            base_score = score
                            break
                    
                    # Boost score if connected to query entities
                    entity_boost = 0
                    if doc_id in entity_matches:
                        entity_boost = 0.2
                    
                    # Boost score based on path centrality
                    centrality = nx.degree_centrality(self.graph).get(doc_id, 0)
                    centrality_boost = centrality * 0.1
                    
                    final_scores.append((doc_id, base_score + entity_boost + centrality_boost))
            
            # Get top k final documents
            final_docs = []
            for doc_id, _ in sorted(final_scores, key=lambda x: x[1], reverse=True)[:k]:
                doc_content = self.graph.nodes[doc_id].get("content", "")
                doc_metadata = self.graph.nodes[doc_id].get("metadata", {})
                
                # Add graph context to metadata
                graph_context = self._get_graph_context(doc_id)
                if graph_context:
                    if not doc_metadata:
                        doc_metadata = {}
                    doc_metadata["graph_context"] = graph_context
                
                final_docs.append(Document(page_content=doc_content, metadata=doc_metadata))
            
            return final_docs
            
        except Exception as e:
            logger.error(f"Error in graph retrieval: {e}")
            return []
    
    def _get_graph_context(self, doc_id: str) -> Dict:
        """Get contextual information from the graph for this document"""
        if doc_id not in self.graph:
            return {}
        
        context = {
            "related_entities": [],
            "similar_documents": []
        }
        
        # Get connected entities
        for neighbor in nx.neighbors(self.graph, doc_id):
            if neighbor.startswith("entity_"):
                entity_data = {
                    "name": self.graph.nodes[neighbor].get("name", ""),
                    "type": self.graph.nodes[neighbor].get("entity_type", "")
                }
                context["related_entities"].append(entity_data)
        
        # Get similar documents
        for neighbor in nx.neighbors(self.graph, doc_id):
            if neighbor.startswith("doc_") and neighbor != doc_id:
                edge_data = self.graph.get_edge_data(doc_id, neighbor)
                if edge_data and edge_data.get("relationship") == "similar_to":
                    doc_data = {
                        "id": neighbor,
                        "similarity": edge_data.get("weight", 0),
                        "preview": self.graph.nodes[neighbor].get("content", "")[:100] + "..."
                    }
                    context["similar_documents"].append(doc_data)
        
        return context

    def visualize_graph(self):
        """Create visualization data for the graph"""
        nodes_data = []
        for node, attrs in self.graph.nodes(data=True):
            node_type = attrs.get('type', 'unknown')
            if node_type == 'document':
                label = attrs.get('content', '')[:30] + '...'
                color = '#6baed6'  # Blue for documents
            else:
                label = attrs.get('name', '')
                color = '#fd8d3c'  # Orange for entities
            
            nodes_data.append({
                'id': node,
                'label': label,
                'type': node_type,
                'color': color
            })
        
        edges_data = []
        for source, target, attrs in self.graph.edges(data=True):
            relationship = attrs.get('relationship', '')
            weight = attrs.get('weight', 1.0)
            
            edges_data.append({
                'source': source,
                'target': target,
                'relationship': relationship,
                'weight': weight
            })
        
        return {
            'nodes': nodes_data,
            'edges': edges_data
        }

class CrossEncoderReranker:
    def __init__(self):
        self.model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def rerank(self, query: str, documents: List[Document], top_k: Optional[int] = None) -> List[Document]:
        if not documents:
            return []
        
        # Prepare pairs for cross-encoder
        pairs = [(query, doc.page_content) for doc in documents]
        
        # Tokenize
        features = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.device)
        
        # Get scores
        with torch.no_grad():
            scores = self.model(**features).logits.squeeze(-1).cpu().tolist()
        
        # Sort documents by score
        scored_docs = [(doc, score) for doc, score in zip(documents, scores)]
        ranked_docs = [doc for doc, _ in sorted(scored_docs, key=lambda x: x[1], reverse=True)]
        
        if top_k:
            return ranked_docs[:top_k]
        return ranked_docs

def create_hierarchical_chunks(text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
    """Create hierarchical document chunks"""
    # Create document-level chunk
    doc_chunk = Document(
        page_content=text[:10000],  # First 10K chars
        metadata={"level": "document", "parent": None, **(metadata or {})}
    )
    
    # Create section chunks (using headers or paragraphs)
    section_chunks = []
    sections = re.split(r'\n#{1,3}\s+', text)
    
    for i, section in enumerate(sections):
        if not section.strip():
            continue
        section_chunks.append(Document(
            page_content=section.strip(),
            metadata={"level": "section", "parent": id(doc_chunk), "index": i, **(metadata or {})}
        ))
    
    # Create paragraph chunks
    paragraph_chunks = []
    for section_chunk in section_chunks:
        paragraphs = re.split(r'\n\n+', section_chunk.page_content)
        
        for j, para in enumerate(paragraphs):
            if len(para.strip()) < 50:  # Skip very short paragraphs
                continue
            paragraph_chunks.append(Document(
                page_content=para.strip(),
                metadata={
                    "level": "paragraph", 
                    "parent": id(section_chunk),
                    "section_index": section_chunk.metadata["index"],
                    "index": j,
                    **(metadata or {})
                }
            ))
    
    return [doc_chunk] + section_chunks + paragraph_chunks

def decompose_query(query: str) -> List[str]:
    """Break complex query into sub-queries"""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        prompt = f"""Break this complex query into 2-3 simpler sub-questions that together would help answer the original question.
        
        Original query: "{query}"
        
        Format: Return only the numbered sub-questions, one per line.
        """
        
        model = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash")
        response = model.invoke(prompt).content.strip()
        
        # Extract sub-queries
        sub_queries = [line.strip() for line in response.split('\n') 
                      if line.strip() and not line.strip().isdigit()]
        
        # Clean up numbering if present
        sub_queries = [re.sub(r'^\d+[\.\)]\s*', '', sq) for sq in sub_queries]
        
        return sub_queries if sub_queries else [query]
    except Exception as e:
        logger.error(f"Query decomposition failed: {e}")
        return [query]  # Fall back to original query

def visualize_knowledge_graph(graph: KnowledgeGraph, output_path: str = "knowledge_graph.html"):
    """Visualize the knowledge graph using pyvis"""
    net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black")
    
    # Add nodes
    for node, data in graph.graph.nodes(data=True):
        if data.get('type') == 'document':
            net.add_node(node, label=data['content'][:50], color='#97c2fc')
        elif data.get('type') == 'entity':
            net.add_node(node, label=data['name'], color='#ffb3b3')
        else:
            net.add_node(node, label=data.get('content', '')[:30], color='#b3ffb3')
    
    # Add edges
    for source, target, data in graph.graph.edges(data=True):
        net.add_edge(source, target, title=data.get('relationship', ''))
    
    # Save the graph
    net.save_graph(output_path)
    return output_path 