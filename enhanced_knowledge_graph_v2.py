#!/usr/bin/env python3
"""
Enhanced Knowledge Graph Builder v2
This module provides an improved implementation for building knowledge graphs from documents.
"""

import logging
import re
import networkx as nx
from typing import List, Dict, Any, Optional, Set, Tuple
import spacy
try:
    from spacy.language import Language
except ImportError:
    Language = Any  # Type fallback if spacy not available

# Configure logging
logger = logging.getLogger(__name__)

class EnhancedKnowledgeGraphBuilder:
    """
    An enhanced knowledge graph builder that creates more meaningful connections
    between entities and concepts extracted from documents.
    """
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize the knowledge graph builder.
        
        Args:
            use_gpu: Whether to use GPU acceleration for NLP processing
        """
        self.use_gpu = use_gpu
        self.nlp = self._load_nlp_model()
        self.graph = nx.DiGraph()
        logger.info("Enhanced Knowledge Graph Builder initialized")
        
    def _load_nlp_model(self) -> Optional[Language]:
        """Load and configure the spaCy NLP model."""
        try:
            # Load a more comprehensive model for better entity recognition
            model_name = "en_core_web_lg"
            try:
                nlp = spacy.load(model_name)
                logger.info(f"Loaded spaCy model: {model_name}")
            except OSError:
                logger.warning(f"Model {model_name} not found. Downloading...")
                spacy.cli.download(model_name)
                nlp = spacy.load(model_name)
                
            # Configure pipeline components
            if self.use_gpu:
                spacy.prefer_gpu()
                logger.info("Using GPU acceleration for NLP processing")
                
            return nlp
        except Exception as e:
            logger.error(f"Error loading NLP model: {str(e)}")
            logger.warning("Continuing without NLP model - functionality will be limited")
            return None
            
    def build_from_documents(self, documents: List[Dict[str, Any]]) -> nx.DiGraph:
        """
        Build a knowledge graph from a list of documents.
        
        Args:
            documents: List of document dictionaries with at least 'content' and 'metadata' keys
            
        Returns:
            A directed graph (NetworkX DiGraph) representing the knowledge graph
        """
        if not documents:
            logger.warning("No documents provided for knowledge graph building")
            return nx.DiGraph()
            
        logger.info(f"Building knowledge graph from {len(documents)} documents")
        self.graph = nx.DiGraph()  # Reset graph
        
        # Process each document
        for doc_idx, doc in enumerate(documents):
            try:
                content = doc.get('content', doc.get('page_content', ''))
                metadata = doc.get('metadata', {})
                
                if not content:
                    logger.warning(f"Empty content in document {doc_idx}")
                    continue
                    
                # Extract document info for node attributes
                doc_id = metadata.get('source', f"doc_{doc_idx}")
                doc_title = metadata.get('title', f"Document {doc_idx}")
                
                # Add document node
                self.graph.add_node(
                    doc_id,
                    type='document',
                    title=doc_title,
                    metadata=metadata
                )
                
                # Extract entities and concepts
                self._process_document_content(content, doc_id)
                
                logger.debug(f"Processed document {doc_idx}: {doc_title}")
            except Exception as e:
                logger.error(f"Error processing document {doc_idx}: {str(e)}")
                
        # Build relationships between entities across documents
        self._build_cross_document_relationships()
        
        logger.info(f"Knowledge graph built with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
        return self.graph
        
    def _process_document_content(self, content: str, doc_id: str) -> None:
        """
        Process document content to extract entities and concepts.
        
        Args:
            content: The document content text
            doc_id: The document identifier
        """
        if not self.nlp:
            # Fallback to simple keyword extraction if NLP model is unavailable
            self._simple_keyword_extraction(content, doc_id)
            return
            
        # Process text with spaCy
        try:
            # Split long content into manageable chunks if needed
            max_length = self.nlp.max_length
            if len(content) > max_length:
                chunks = [content[i:i+max_length] for i in range(0, len(content), max_length)]
                logger.debug(f"Split content into {len(chunks)} chunks for processing")
            else:
                chunks = [content]
                
            entities = []
            for chunk in chunks:
                doc = self.nlp(chunk)
                chunk_entities = self._extract_entities(doc)
                entities.extend(chunk_entities)
                
            # Add entities to graph and connect to document
            self._add_entities_to_graph(entities, doc_id)
            
        except Exception as e:
            logger.error(f"Error in NLP processing: {str(e)}")
            # Fall back to simple extraction
            self._simple_keyword_extraction(content, doc_id)
            
    def _extract_entities(self, doc) -> List[Dict[str, Any]]:
        """
        Extract entities from a spaCy processed document.
        
        Args:
            doc: A spaCy Doc object
            
        Returns:
            List of entity dictionaries with text, label, and other attributes
        """
        entities = []
        
        # Extract named entities
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'type': 'entity',
                'start': ent.start_char,
                'end': ent.end_char
            })
            
        # Extract noun chunks as concepts
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) > 1:  # Only multi-word chunks
                entities.append({
                    'text': chunk.text,
                    'label': 'CONCEPT',
                    'type': 'concept',
                    'start': chunk.start_char,
                    'end': chunk.end_char
                })
                
        # Extract key phrases based on dependency parsing
        for token in doc:
            if token.dep_ in ('nsubj', 'dobj') and token.head.pos_ == 'VERB':
                phrase = f"{token.text} {token.head.text}"
                entities.append({
                    'text': phrase,
                    'label': 'ACTION',
                    'type': 'action',
                    'start': token.idx,
                    'end': token.head.idx + len(token.head.text)
                })
                
        return entities
        
    def _simple_keyword_extraction(self, content: str, doc_id: str) -> None:
        """
        Simple keyword extraction fallback when NLP is unavailable.
        
        Args:
            content: Document content
            doc_id: Document identifier
        """
        logger.info("Using simple keyword extraction")
        
        # Extract potential entities using regex patterns
        # Names (capitalized words)
        name_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        names = re.findall(name_pattern, content)
        
        # Technical terms (common in technical documents)
        tech_pattern = r'\b[a-z]+(?:[A-Z][a-z]*)+\b'  # camelCase
        tech_terms = re.findall(tech_pattern, content)
        
        # Add these as entities
        entities = []
        for name in set(names):
            if len(name) > 3:  # Filter out short names
                entities.append({
                    'text': name,
                    'label': 'PERSON',
                    'type': 'entity'
                })
                
        for term in set(tech_terms):
            entities.append({
                'text': term,
                'label': 'TECH',
                'type': 'concept'
            })
            
        self._add_entities_to_graph(entities, doc_id)
        
    def _add_entities_to_graph(self, entities: List[Dict[str, Any]], doc_id: str) -> None:
        """
        Add extracted entities to the knowledge graph.
        
        Args:
            entities: List of entity dictionaries
            doc_id: Document identifier
        """
        # Add nodes for each unique entity
        entity_nodes = {}
        
        for entity in entities:
            # Create a unique ID for the entity
            entity_text = entity['text'].lower().strip()
            if not entity_text or len(entity_text) < 2:
                continue
                
            entity_id = f"{entity['type']}_{entity_text.replace(' ', '_')}"
            
            # Add entity node if it doesn't exist
            if entity_id not in self.graph:
                self.graph.add_node(
                    entity_id,
                    label=entity_text,
                    type=entity['type'],
                    category=entity.get('label', 'UNKNOWN')
                )
                
            # Connect entity to document
            self.graph.add_edge(
                doc_id,
                entity_id,
                relation='contains',
                weight=1.0
            )
            
            # Store for later relationship building
            entity_nodes[entity_id] = entity
            
        # Build relationships between entities in the same document
        self._build_entity_relationships(list(entity_nodes.keys()), doc_id)
        
    def _build_entity_relationships(self, entity_ids: List[str], doc_id: str) -> None:
        """
        Build relationships between entities in the same document.
        
        Args:
            entity_ids: List of entity IDs from the same document
            doc_id: Document identifier
        """
        # Connect entities that appear in the same document
        for i, entity1 in enumerate(entity_ids):
            for entity2 in entity_ids[i+1:]:
                # Skip self-connections
                if entity1 == entity2:
                    continue
                    
                # Get entity types
                type1 = self.graph.nodes[entity1]['type']
                type2 = self.graph.nodes[entity2]['type']
                
                # Different relationship types based on entity types
                if type1 == 'entity' and type2 == 'entity':
                    relation = 'co_occurs_with'
                elif type1 == 'concept' and type2 == 'concept':
                    relation = 'related_to'
                elif type1 == 'action' or type2 == 'action':
                    relation = 'acts_on'
                else:
                    relation = 'associated_with'
                
                # Check if edge already exists
                if self.graph.has_edge(entity1, entity2):
                    # Update weight of existing edge
                    self.graph[entity1][entity2]['weight'] += 0.5
                else:
                    # Add new edge
                    self.graph.add_edge(
                        entity1,
                        entity2,
                        relation=relation,
                        source_doc=doc_id,
                        weight=1.0
                    )
                    
    def _build_cross_document_relationships(self) -> None:
        """Build relationships between entities across different documents."""
        # Find all entity nodes
        entity_nodes = [node for node, attrs in self.graph.nodes(data=True) 
                       if attrs.get('type') in ('entity', 'concept', 'action')]
                       
        # Connect similar entities across documents
        for i, entity1 in enumerate(entity_nodes):
            for entity2 in entity_nodes[i+1:]:
                # Skip if same entity
                if entity1 == entity2:
                    continue
                    
                # Get labels (text representation)
                label1 = self.graph.nodes[entity1].get('label', '')
                label2 = self.graph.nodes[entity2].get('label', '')
                
                # Connect if highly similar
                if self._calculate_similarity(label1, label2) > 0.8:
                    self.graph.add_edge(
                        entity1, 
                        entity2,
                        relation='similar_to',
                        weight=0.8
                    )
                    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score between 0 and 1
        """
        # If NLP model available, use vector similarity
        if self.nlp:
            try:
                doc1 = self.nlp(text1)
                doc2 = self.nlp(text2)
                return doc1.similarity(doc2)
            except Exception:
                pass
                
        # Fallback to simple string comparison
        # Jaccard similarity of words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if not union:
            return 0.0
            
        return len(intersection) / len(union) 