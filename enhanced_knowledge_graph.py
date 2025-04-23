import logging
import json
import re
import os
from typing import List, Dict, Any, Optional, Tuple, Set
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pyvis.network import Network
from langchain.schema import Document
import spacy
from collections import Counter
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("knowledge_graph")

class EnhancedKnowledgeGraph:
    """Build and visualize knowledge graphs from retrieved documents with entity relationships."""
    
    def __init__(self, height: str = "600px", width: str = "100%"):
        """Initialize the knowledge graph builder.
        
        Args:
            height: Height of the visualization
            width: Width of the visualization
        """
        self.height = height
        self.width = width
        
        # Initialize spaCy NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model for knowledge graph")
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            try:
                logger.info("Attempting to download spaCy model...")
                spacy.cli.download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Downloaded and loaded spaCy model")
            except Exception as e:
                logger.error(f"Failed to download spaCy model: {e}")
                self.nlp = None
        
        # Define entity types of interest for the knowledge graph
        self.entity_types = {
            "PERSON": "People",
            "ORG": "Organizations",
            "GPE": "Locations",
            "LOC": "Locations",
            "PRODUCT": "Products",
            "EVENT": "Events",
            "WORK_OF_ART": "Creative Works",
            "FAC": "Facilities",
            "NORP": "Groups",
            "DATE": "Dates",
            "NOUN_CHUNK": "Keywords"
        }
        
        # Define source type colors
        self.source_colors = {
            "YouTube": "#3498db",  # Blue
            "News": "#e74c3c",     # Red
            "Blog": "#2ecc71",     # Green
            "Query": "#9b59b6"     # Purple
        }
        
        # Color scheme for entity types
        self.entity_colors = {
            "People": "#8e44ad",         # Purple
            "Organizations": "#9b59b6",  # Purple
            "Locations": "#3498db",      # Blue
            "Products": "#1abc9c",       # Teal
            "Events": "#f39c12",         # Orange
            "Creative Works": "#d35400", # Orange
            "Facilities": "#27ae60",     # Green
            "Groups": "#2980b9",         # Blue
            "Dates": "#7f8c8d",          # Gray
            "Keywords": "#8e44ad"        # Purple
        }
        
        # Color for relationship nodes
        self.relationship_color = "#e67e22"  # Orange
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text using spaCy.
        
        Args:
            text: Input text
            
        Returns:
            List of entities with type and text
        """
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text)
            entities = []
            
            # Extract named entities
            for ent in doc.ents:
                if ent.label_ in self.entity_types:
                    entities.append({
                        "text": ent.text,
                        "type": self.entity_types[ent.label_]
                    })
            
            # Extract noun chunks as keywords
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) >= 2:  # Only multi-word chunks
                    entities.append({
                        "text": chunk.text,
                        "type": self.entity_types["NOUN_CHUNK"]
                    })
            
            return entities
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    def _extract_relationships(self, doc: Document) -> List[Tuple[str, str, str]]:
        """Extract potential relationships between entities in a document.
        
        Args:
            doc: Document object
            
        Returns:
            List of tuples (entity1, relationship, entity2)
        """
        if not self.nlp:
            return []
        
        try:
            # Process with spaCy
            spacy_doc = self.nlp(doc.page_content)
            
            # Extract entities
            entity_mentions = {}
            for ent in spacy_doc.ents:
                if ent.label_ in self.entity_types:
                    entity_mentions[ent.text] = self.entity_types[ent.label_]
            
            # Extract noun chunks
            for chunk in spacy_doc.noun_chunks:
                if len(chunk.text.split()) >= 2:  # Only multi-word chunks
                    entity_mentions[chunk.text] = self.entity_types["NOUN_CHUNK"]
            
            # Find sentence-level co-occurrences
            relationships = []
            
            # Process each sentence
            for sent in spacy_doc.sents:
                sent_entities = []
                
                # Find entities in this sentence
                for ent_text, ent_type in entity_mentions.items():
                    if ent_text.lower() in sent.text.lower():
                        sent_entities.append((ent_text, ent_type))
                
                # Create relationships between entities in the same sentence
                for i in range(len(sent_entities)):
                    for j in range(i+1, len(sent_entities)):
                        entity1, type1 = sent_entities[i]
                        entity2, type2 = sent_entities[j]
                        
                        # Create a generic relationship based on document source
                        relationship = f"mentioned together in {doc.metadata.get('source', 'document')}"
                        
                        relationships.append((entity1, relationship, entity2))
            
            return relationships
        
        except Exception as e:
            logger.error(f"Error extracting relationships: {e}")
            return []
    
    def build_graph(self, documents: List[Document], query: str) -> nx.Graph:
        """Build a knowledge graph from documents and query.
        
        Args:
            documents: List of Document objects
            query: Original search query
            
        Returns:
            NetworkX graph object
        """
        # Create a graph
        G = nx.Graph()
        
        # Add query node
        G.add_node(query, 
                   size=30, 
                   type="Query",
                   color=self.source_colors["Query"],
                   title="Search Query")
        
        # Track entities to avoid duplicates
        all_entities = set()
        source_nodes = set()
        
        # Process each document
        for doc in documents:
            try:
                # Get document metadata
                source_type = doc.metadata.get("source", "Unknown")
                title = doc.metadata.get("title", "Untitled")
                url = doc.metadata.get("url", "")
                relevance_score = doc.metadata.get("relevance_score", 0.5)
                
                # Generate a unique node ID for this document
                node_id = f"{source_type}: {title}"
                
                # Truncate long titles
                if len(node_id) > 40:
                    node_id = node_id[:37] + "..."
                
                # Add document node if not already present
                if node_id not in source_nodes:
                    source_nodes.add(node_id)
                    
                    # Calculate node size based on relevance score
                    node_size = 15 + (relevance_score * 15)
                    
                    # Add source node
                    G.add_node(node_id, 
                              size=node_size, 
                              type=source_type,
                              color=self.source_colors.get(source_type, "#7f8c8d"),  # Default to gray
                              url=url,
                              relevance=relevance_score,
                              title=f"{source_type}: {title}\nRelevance: {relevance_score:.2f}")
                    
                    # Connect source to query
                    G.add_edge(query, node_id, 
                              width=1 + (relevance_score * 4),
                              title=f"Relevance: {relevance_score:.2f}")
                
                # Extract entities
                entities = self._extract_entities(doc.page_content)
                
                # Filter out common stop entities and keep track of unique entities
                for entity in entities:
                    entity_text = entity["text"]
                    entity_type = entity["type"]
                    
                    # Skip if too short
                    if len(entity_text) < 3:
                        continue
                    
                    # Create a unique identifier to prevent duplicates with different case
                    entity_id = f"{entity_text}|{entity_type}"
                    
                    # Add entity node if not already present
                    if entity_id not in all_entities:
                        all_entities.add(entity_id)
                        
                        # Add entity node
                        G.add_node(entity_text, 
                                  size=10, 
                                  type=entity_type,
                                  color=self.entity_colors.get(entity_type, "#95a5a6"),  # Default to light gray
                                  title=f"{entity_type}: {entity_text}")
                    
                    # Connect entity to source
                    if G.has_edge(node_id, entity_text):
                        # Increment weight if edge already exists
                        G[node_id][entity_text]["weight"] += 1
                        G[node_id][entity_text]["width"] = 1 + min(G[node_id][entity_text]["weight"], 5)
                    else:
                        # Create new edge
                        G.add_edge(node_id, entity_text, 
                                  weight=1, 
                                  width=1,
                                  title=f"Mentioned in: {title}")
                
                # Extract and add relationships between entities
                relationships = self._extract_relationships(doc)
                
                for entity1, relationship, entity2 in relationships:
                    # Skip if either entity is not in the graph
                    if entity1 not in G.nodes or entity2 not in G.nodes:
                        continue
                    
                    # Create a relationship node
                    rel_node_id = f"{relationship}"
                    
                    # Check if we already have this relationship type
                    if rel_node_id not in G.nodes:
                        G.add_node(rel_node_id, 
                                  size=5, 
                                  type="Relationship",
                                  color=self.relationship_color,
                                  title=relationship,
                                  shape="diamond")
                    
                    # Connect entities to relationship
                    if not G.has_edge(entity1, rel_node_id):
                        G.add_edge(entity1, rel_node_id, width=1)
                    
                    if not G.has_edge(entity2, rel_node_id):
                        G.add_edge(entity2, rel_node_id, width=1)
            
            except Exception as e:
                logger.error(f"Error processing document for graph: {e}")
        
        return G
    
    def visualize(self, graph: nx.Graph, notebook: bool = False) -> str:
        """Visualize the knowledge graph.
        
        Args:
            graph: NetworkX graph object
            notebook: Whether to return HTML for notebook display
            
        Returns:
            Path to the HTML file or HTML string
        """
        try:
            # Create PyVis network
            net = Network(height=self.height, width=self.width, notebook=notebook)
            
            # Set options for better visualization
            net.set_options("""
            {
              "physics": {
                "forceAtlas2Based": {
                  "springLength": 100,
                  "springConstant": 0.1
                },
                "minVelocity": 0.75,
                "solver": "forceAtlas2Based",
                "timestep": 0.5
              },
              "interaction": {
                "hover": true,
                "navigationButtons": true,
                "multiselect": true
              },
              "edges": {
                "smooth": {
                  "type": "continuous",
                  "forceDirection": "none"
                }
              }
            }
            """)
            
            # Add nodes and edges from NetworkX graph
            for node, attrs in graph.nodes(data=True):
                net.add_node(node, 
                            **{k: v for k, v in attrs.items() if k != 'id'})
            
            for source, target, attrs in graph.edges(data=True):
                net.add_edge(source, target, 
                            **{k: v for k, v in attrs.items() if k != 'id'})
            
            # Generate temporary HTML file
            temp_dir = tempfile.gettempdir()
            html_path = os.path.join(temp_dir, "knowledge_graph.html")
            
            # Save the graph to HTML
            net.save_graph(html_path)
            
            if notebook:
                # Return HTML string for embedding in notebook
                with open(html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                return html_content
            else:
                # Return the path to the HTML file
                return html_path
        
        except Exception as e:
            logger.error(f"Error visualizing graph: {e}")
            return ""
    
    def generate_knowledge_graph(self, documents: List[Document], query: str, notebook: bool = False) -> str:
        """Generate and visualize a knowledge graph from documents.
        
        Args:
            documents: List of Document objects
            query: Original search query
            notebook: Whether to return HTML for notebook display
            
        Returns:
            Path to the HTML file or HTML string
        """
        # Build the graph
        graph = self.build_graph(documents, query)
        
        # Log graph statistics
        logger.info(f"Knowledge graph created with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        
        # Visualize the graph
        return self.visualize(graph, notebook)
    
    def summarize_graph(self, graph: nx.Graph) -> Dict[str, Any]:
        """Summarize the knowledge graph contents.
        
        Args:
            graph: NetworkX graph object
            
        Returns:
            Dictionary with graph statistics and key entities
        """
        try:
            summary = {
                "node_count": len(graph.nodes),
                "edge_count": len(graph.edges),
                "sources": {},
                "entities": {},
                "key_entities": []
            }
            
            # Count nodes by type
            node_types = {}
            for node, data in graph.nodes(data=True):
                node_type = data.get("type", "Unknown")
                
                if node_type not in node_types:
                    node_types[node_type] = 0
                node_types[node_type] += 1
                
                # Count source types
                if node_type in self.source_colors.keys():
                    if node_type not in summary["sources"]:
                        summary["sources"][node_type] = 0
                    summary["sources"][node_type] += 1
                
                # Count entity types
                if node_type in self.entity_colors.keys():
                    if node_type not in summary["entities"]:
                        summary["entities"][node_type] = 0
                    summary["entities"][node_type] += 1
            
            # Find key entities (nodes with highest degree)
            entity_nodes = [(node, data, graph.degree(node)) 
                          for node, data in graph.nodes(data=True) 
                          if data.get("type", "") in self.entity_colors.keys()]
            
            # Sort by degree (connection count)
            entity_nodes.sort(key=lambda x: x[2], reverse=True)
            
            # Get top 10 key entities
            for node, data, degree in entity_nodes[:10]:
                summary["key_entities"].append({
                    "text": node,
                    "type": data.get("type", "Unknown"),
                    "connections": degree
                })
            
            return summary
        
        except Exception as e:
            logger.error(f"Error summarizing graph: {e}")
            return {"error": str(e)}
    
    def export_graph_data(self, graph: nx.Graph) -> Dict[str, Any]:
        """Export graph data as a dictionary.
        
        Args:
            graph: NetworkX graph object
            
        Returns:
            Dictionary with nodes and edges
        """
        try:
            # Convert graph to dictionary
            data = {
                "nodes": [],
                "edges": []
            }
            
            # Add nodes
            for node, attrs in graph.nodes(data=True):
                node_data = {
                    "id": node,
                    "label": node
                }
                node_data.update(attrs)
                data["nodes"].append(node_data)
            
            # Add edges
            for source, target, attrs in graph.edges(data=True):
                edge_data = {
                    "from": source,
                    "to": target
                }
                edge_data.update(attrs)
                data["edges"].append(edge_data)
            
            return data
        
        except Exception as e:
            logger.error(f"Error exporting graph data: {e}")
            return {"error": str(e)} 