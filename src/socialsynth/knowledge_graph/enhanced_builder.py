#!/usr/bin/env python3
"""
Enhanced Knowledge Graph Builder

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
logger = logging.getLogger("socialsynth.knowledge_graph.builder")

class EnhancedKnowledgeGraphBuilder:
    """
    An enhanced knowledge graph builder that creates more meaningful connections
    between entities and concepts extracted from documents.
    """
    
    def __init__(
        self,
        relevance_threshold: float = 0.5,
        max_entities_per_doc: int = 20,
        max_keywords: int = 30,
        height: str = "800px",
        width: str = "100%",
        use_gpu: bool = False
    ):
        """
        Initialize the knowledge graph builder.
        
        Args:
            relevance_threshold: Minimum relevance score for entities
            max_entities_per_doc: Maximum entities to extract per document
            max_keywords: Maximum keywords to extract
            height: Height for visualization
            width: Width for visualization
            use_gpu: Whether to use GPU acceleration for NLP processing
        """
        self.relevance_threshold = relevance_threshold
        self.max_entities_per_doc = max_entities_per_doc
        self.max_keywords = max_keywords
        self.height = height
        self.width = width
        self.use_gpu = use_gpu
        self.nlp = self._load_nlp_model()
        self.graph = nx.DiGraph()
        self.top_keywords = []
        
        logger.info("Enhanced Knowledge Graph Builder initialized")
        
    def _load_nlp_model(self) -> Optional['spacy.language.Language']:
        """
        Load the NLP model for entity extraction.
        
        Returns:
            Loaded spaCy NLP model or None if loading fails
        """
        try:
            # First check what models are available
            available_models = spacy.util.get_installed_models()
            logger.info(f"Available spaCy models: {', '.join(available_models)}")
            
            # Configure GPU usage if requested
            if self.use_gpu:
                gpu_available = spacy.prefer_gpu()
                if gpu_available:
                    logger.info("GPU acceleration enabled for NLP processing")
                else:
                    logger.warning("GPU requested but not available, falling back to CPU")
            
            # Try to load the best available model
            model_name = None
            if "en_core_web_lg" in available_models:
                model_name = "en_core_web_lg"
            elif "en_core_web_md" in available_models:
                model_name = "en_core_web_md"
            elif "en_core_web_sm" in available_models:
                model_name = "en_core_web_sm"
            else:
                logger.warning("No English spaCy models found")
                return None
                
            logger.info(f"Loading spaCy model: {model_name}")
            nlp = spacy.load(model_name)
            logger.info(f"Successfully loaded spaCy model: {model_name}")
            return nlp
        except (OSError, ImportError) as e:
            logger.error(f"Failed to load spaCy model: {e}")
            logger.warning("Proceeding without NLP capabilities, using keyword extraction only")
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
        self.top_keywords = []  # Reset keywords
        
        # Process each document
        for doc_idx, doc in enumerate(documents):
            try:
                # Get content and metadata
                content = None
                metadata = {}
                
                if isinstance(doc, dict):
                    content = doc.get('content', doc.get('page_content', ''))
                    metadata = doc.get('metadata', {})
                else:
                    logger.warning(f"Document {doc_idx} is not a dictionary, skipping")
                    continue
                
                if not content or not isinstance(content, str):
                    logger.warning(f"Empty or invalid content in document {doc_idx}")
                    continue
                    
                # Extract document info for node attributes
                doc_id = f"doc_{doc_idx}"
                doc_title = metadata.get('title', doc.get('title', f"Document {doc_idx+1}"))
                doc_source = metadata.get('source', 'Unknown')
                doc_url = metadata.get('url', '')
                
                # Add document node
                self.graph.add_node(
                    doc_id,
                    label=doc_title,
                    type='document',
                    source=doc_source,
                    url=doc_url,
                    color='#3498db'  # Blue for documents
                )
                
                # Extract entities and concepts
                self._process_document_content(content, doc_id)
                
                logger.info(f"Processed document {doc_idx}: {doc_title}")
            except Exception as e:
                logger.error(f"Error processing document {doc_idx}: {str(e)}")
        
        # Build relationships between entities across documents
        self._build_cross_document_relationships()
        
        # Extract top keywords
        self._extract_top_keywords()
        
        node_count = len(self.graph.nodes)
        edge_count = len(self.graph.edges)
        
        if node_count == 0:
            logger.warning("Knowledge graph is empty")
        else:
            logger.info(f"Knowledge graph built with {node_count} nodes and {edge_count} edges")
            
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
            # Verify text quality before adding
            if len(ent.text.strip()) > 1 and not ent.text.isspace():
                entities.append({
                    'text': ent.text.strip(),
                    'label': ent.label_,
                    'type': self._map_entity_type(ent.label_),
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'relevance': 0.9,  # Higher relevance for named entities
                    'color': '#e67e22'  # Orange for entities
                })
            
        # Extract noun chunks as concepts
        seen_texts = set(e['text'].lower() for e in entities)
        for chunk in doc.noun_chunks:
            # Only include meaningful noun chunks
            chunk_text = chunk.text.strip()
            if (len(chunk_text.split()) > 1 and  # Multi-word
                len(chunk_text) > 3 and          # Reasonable length
                not chunk_text.lower() in seen_texts and  # Not duplicate
                not chunk_text.isspace()):       # Not just whitespace
                
                entities.append({
                    'text': chunk_text,
                    'label': 'CONCEPT',
                    'type': 'concept',
                    'start': chunk.start_char,
                    'end': chunk.end_char,
                    'relevance': 0.7,  # Medium relevance for concepts
                    'color': '#95a5a6'  # Gray for concepts
                })
                seen_texts.add(chunk_text.lower())
                
        # Extract key action phrases
        action_count = 0
        for token in doc:
            if token.dep_ in ('nsubj', 'dobj') and token.head.pos_ == 'VERB':
                # Build action phrase
                phrase = f"{token.text} {token.head.text}"
                phrase = phrase.strip()
                
                if (len(phrase) > 3 and 
                    not phrase.lower() in seen_texts and 
                    action_count < 10):  # Limit action phrases
                    
                    entities.append({
                        'text': phrase,
                        'label': 'ACTION',
                        'type': 'action',
                        'start': token.idx,
                        'end': token.head.idx + len(token.head.text),
                        'relevance': 0.6,  # Lower relevance for actions
                        'color': '#2ecc71'  # Green for actions
                    })
                    seen_texts.add(phrase.lower())
                    action_count += 1
                
        # Add extra filter for relevance based on document frequency
        word_counts = {}
        for token in doc:
            if not token.is_stop and not token.is_punct and len(token.text) > 2:
                word_counts[token.text.lower()] = word_counts.get(token.text.lower(), 0) + 1
        
        # Boost relevance for entities containing frequent terms
        for entity in entities:
            for word in entity['text'].lower().split():
                if word in word_counts and word_counts[word] > 2:
                    entity['relevance'] = min(1.0, entity['relevance'] + 0.1)
        
        # Return filtered to top most relevant entities
        entities.sort(key=lambda x: x.get('relevance', 0), reverse=True)
        return entities[:self.max_entities_per_doc] if self.max_entities_per_doc > 0 else entities
    
    def _map_entity_type(self, spacy_label: str) -> str:
        """Map spaCy entity labels to our entity types"""
        if spacy_label in ('PERSON', 'PER'):
            return 'person'
        elif spacy_label in ('ORG', 'ORGANIZATION'):
            return 'organization'
        elif spacy_label in ('GPE', 'LOC', 'LOCATION'):
            return 'location'
        elif spacy_label in ('DATE', 'TIME'):
            return 'date'
        elif spacy_label in ('EVENT'):
            return 'event'
        else:
            return 'entity'
        
    def _simple_keyword_extraction(self, content: str, doc_id: str) -> None:
        """
        Simple keyword extraction fallback when NLP is unavailable.
        
        Args:
            content: Document content
            doc_id: Document identifier
        """
        logger.info("Using simple keyword extraction")
        
        # Make sure we have content to process
        if not content or not isinstance(content, str):
            logger.warning(f"Invalid content for document {doc_id}")
            return
            
        # Clean content - remove extra whitespace and normalize
        content = ' '.join(content.split())
        
        # Extract potential entities using regex patterns
        # Names (capitalized words)
        name_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        names = re.findall(name_pattern, content)
        
        # Technical terms (common in technical documents)
        tech_pattern = r'\b[a-z]+(?:[A-Z][a-z]*)+\b'  # camelCase
        tech_terms = re.findall(tech_pattern, content)
        
        # Concept phrases (multi-word lowercase terms)
        concept_pattern = r'\b[a-z][a-z]+(?: [a-z][a-z]+){1,2}\b'
        concepts = re.findall(concept_pattern, content)
        
        # URLs and references (often important in web content)
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        urls = re.findall(url_pattern, content)
        
        # Add these as entities
        entities = []
        
        # Add named entities with higher relevance
        for name in set(names):
            if len(name) > 3:  # Filter out short names
                entities.append({
                    'text': name,
                    'label': 'PERSON',
                    'type': 'entity',
                    'relevance': 0.8,
                    'color': '#e67e22'  # Orange for entities
                })
                
        # Add technical terms as concepts        
        for term in set(tech_terms):
            if len(term) > 3:
                entities.append({
                    'text': term,
                    'label': 'TECH',
                    'type': 'concept',
                    'relevance': 0.7,
                    'color': '#95a5a6'  # Gray for concepts
                })
        
        # Add general concepts
        for concept in set(concepts):
            if len(concept.split()) > 1:  # Only multi-word concepts
                entities.append({
                    'text': concept,
                    'label': 'CONCEPT',
                    'type': 'concept',
                    'relevance': 0.6,
                    'color': '#95a5a6'  # Gray for concepts
                })
        
        # Add URLs as sources (useful for web content)
        for url in set(urls):
            entities.append({
                'text': url,
                'label': 'URL',
                'type': 'source',
                'relevance': 0.9,  # High relevance for sources
                'color': '#3498db'  # Blue for sources
            })
            
        # Add to graph with frequency-based filtering
        word_counts = {}
        for entity in entities:
            words = entity['text'].lower().split()
            for word in words:
                if len(word) > 2:  # Only count significant words
                    word_counts[word] = word_counts.get(word, 0) + 1
        
        # Boost relevance for entities with frequent terms
        for entity in entities:
            for word in entity['text'].lower().split():
                if word in word_counts and word_counts[word] > 1:
                    entity['relevance'] = min(1.0, entity.get('relevance', 0.5) + 0.1)
        
        # Filter by relevance threshold and add to graph
        filtered_entities = [e for e in entities if e.get('relevance', 0) >= self.relevance_threshold]
        
        # Ensure we have some entities even if relevance filtering is strict
        if len(filtered_entities) < 5 and entities:
            # Take at least 5 entities or all if fewer
            filtered_entities = sorted(entities, key=lambda x: x.get('relevance', 0), reverse=True)
            filtered_entities = filtered_entities[:min(5, len(entities))]
        
        self._add_entities_to_graph(filtered_entities, doc_id)
        
        # Store some keywords for the summary
        all_words = [e['text'] for e in filtered_entities]
        self.top_keywords = list(set([word for word in all_words if len(word) > 3]))[:self.max_keywords]
        
    def _add_entities_to_graph(self, entities: List[Dict[str, Any]], doc_id: str) -> None:
        """
        Add extracted entities to the knowledge graph.
        
        Args:
            entities: List of entity dictionaries
            doc_id: Document identifier
        """
        # Add nodes for each unique entity
        entity_nodes = {}
        
        # Limit entities to max_entities_per_doc
        filtered_entities = entities[:self.max_entities_per_doc] if self.max_entities_per_doc > 0 else entities
        
        for idx, entity in enumerate(filtered_entities):
            # Create a unique ID for the entity
            entity_text = entity['text'].lower().strip()
            if not entity_text or len(entity_text) < 2:
                continue
                
            entity_type = entity.get('type', 'entity')
            entity_id = f"{entity_type}_{len(entity_nodes)}"
            
            # Skip if relevance threshold not met
            relevance = entity.get('relevance', 0.8)  # Default relevance
            if relevance < self.relevance_threshold:
                continue
                
            # Add to graph if not already present
            if entity_id not in self.graph:
                self.graph.add_node(
                    entity_id,
                    label=entity['text'],
                    type=entity_type,
                    relevance=relevance,
                    color=entity.get('color', '#95a5a6')
                )
                
            # Connect entity to document
            self.graph.add_edge(
                doc_id,
                entity_id,
                weight=relevance,
                label='contains' if entity_type == 'entity' else 'discusses'
            )
            
            entity_nodes[entity_id] = entity
            
        # Build relationships between entities in the same document
        self._build_entity_relationships(list(entity_nodes.keys()), doc_id)
        
    def _build_entity_relationships(self, entity_ids: List[str], doc_id: str) -> None:
        """
        Build relationships between entities in the same document.
        
        Args:
            entity_ids: List of entity IDs
            doc_id: Document identifier
        """
        # Create connections between entities in the same document
        for i, entity_id1 in enumerate(entity_ids):
            entity1_type = self.graph.nodes[entity_id1].get('type', '')
            
            for entity_id2 in entity_ids[i+1:]:
                entity2_type = self.graph.nodes[entity_id2].get('type', '')
                
                # Skip if both are the same type (avoid concept-concept links)
                if entity1_type == entity2_type and entity1_type != 'entity':
                    continue
                    
                # Determine relationship type based on entity types
                if entity1_type == 'concept' and entity2_type == 'entity':
                    rel_type = 'describes'
                    weight = 0.7
                elif entity1_type == 'entity' and entity2_type == 'concept':
                    rel_type = 'has_property'
                    weight = 0.7
                elif entity1_type == 'action' and entity2_type in ('entity', 'concept'):
                    rel_type = 'acts_on'
                    weight = 0.8
                elif entity2_type == 'action' and entity1_type in ('entity', 'concept'):
                    rel_type = 'subject_of'
                    weight = 0.8
                else:
                    rel_type = 'related_to'
                    weight = 0.6
                
                # Add the relationship if it doesn't exist
                if not self.graph.has_edge(entity_id1, entity_id2):
                    self.graph.add_edge(
                        entity_id1,
                        entity_id2,
                        weight=weight,
                        label=rel_type,
                        doc_id=doc_id
                    )
        
    def _build_cross_document_relationships(self) -> None:
        """Build relationships between entities across different documents."""
        # Get all entities
        entities = [(node, data) for node, data in self.graph.nodes(data=True) 
                    if data.get('type', '') in ('entity', 'concept')]
                    
        # Compare each pair
        for i, (entity_id1, data1) in enumerate(entities):
            label1 = data1.get('label', '').lower()
            
            if not label1:
                continue
                
            for entity_id2, data2 in entities[i+1:]:
                label2 = data2.get('label', '').lower()
                
                if not label2:
                    continue
                    
                # Check similarity
                similarity = self._calculate_similarity(label1, label2)
                
                # Add edge if similar enough
                if similarity > 0.7:
                    rel_type = 'similar_to'
                    self.graph.add_edge(
                        entity_id1,
                        entity_id2,
                        weight=similarity,
                        label=rel_type
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
        # If text is identical, return 1.0
        if text1 == text2:
            return 1.0
            
        # Try using spaCy vector similarity if available
        if self.nlp:
            try:
                doc1 = self.nlp(text1)
                doc2 = self.nlp(text2)
                
                if doc1.vector_norm and doc2.vector_norm:
                    return doc1.similarity(doc2)
            except Exception:
                # Fall back to Jaccard similarity
                pass
                
        # Jaccard similarity (fallback)
        set1 = set(text1.split())
        set2 = set(text2.split())
        
        if not set1 or not set2:
            return 0.0
            
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def visualize(self, graph: nx.DiGraph, output_path: str, title: str = "Knowledge Graph") -> str:
        """
        Create a visualization of the knowledge graph.
        
        Args:
            graph: NetworkX graph to visualize
            output_path: Path to save the visualization
            title: Title for the visualization
            
        Returns:
            Path to the visualization file
        """
        try:
            # Use pyvis for visualization
            from pyvis.network import Network
            
            # Create a pyvis network
            net = Network(height=self.height, width=self.width, notebook=False, directed=True)
            
            # Add nodes with styling
            for node, attrs in graph.nodes(data=True):
                label = attrs.get('label', str(node))
                node_type = attrs.get('type', 'entity')
                color = attrs.get('color', '#95a5a6')  # Default gray
                
                net.add_node(node, label=label, title=label, color=color, group=node_type)
            
            # Add edges
            for source, target, attrs in graph.edges(data=True):
                weight = attrs.get('weight', 1.0)
                label = attrs.get('label', '')
                
                net.add_edge(source, target, value=weight, title=label)
            
            # Set physics layout
            net.barnes_hut(gravity=-80000, central_gravity=0.3, spring_length=250)
            
            # Save visualization
            net.save_graph(output_path)
            logger.info(f"Visualization saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error visualizing graph: {e}")
            return ""
    
    def get_graph_summary(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Generate a summary of the knowledge graph.
        
        Args:
            graph: NetworkX graph to summarize
            
        Returns:
            Dictionary with graph summary
        """
        try:
            # Basic statistics
            node_count = graph.number_of_nodes()
            edge_count = graph.number_of_edges()
            
            # Count entity types
            entity_types = {}
            for _, attrs in graph.nodes(data=True):
                node_type = attrs.get('type', 'unknown')
                entity_types[node_type] = entity_types.get(node_type, 0) + 1
            
            return {
                "node_count": node_count,
                "edge_count": edge_count,
                "entity_types": entity_types,
                "top_keywords": self.top_keywords
            }
        except Exception as e:
            logger.error(f"Error generating graph summary: {e}")
            return {
                "node_count": 0,
                "edge_count": 0,
                "entity_types": {},
                "top_keywords": []
            }

    def _extract_top_keywords(self):
        """Extract top keywords from the graph based on centrality metrics"""
        if not self.graph or self.graph.number_of_nodes() == 0:
            self.top_keywords = []
            return
            
        try:
            # Calculate node centrality
            centrality = nx.degree_centrality(self.graph)
            
            # Filter for entity and concept nodes
            keywords = []
            for node, value in centrality.items():
                attrs = self.graph.nodes[node]
                node_type = attrs.get('type', '')
                
                if node_type in ['entity', 'concept', 'keyword']:
                    keywords.append((attrs.get('label', str(node)), value))
            
            # Sort by centrality and take top keywords
            keywords.sort(key=lambda x: x[1], reverse=True)
            self.top_keywords = [k[0] for k in keywords[:self.max_keywords]]
        except Exception as e:
            logger.error(f"Error extracting top keywords: {e}")
            self.top_keywords = [] 