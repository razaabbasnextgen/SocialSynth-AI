import os
import logging
import json
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Set, Tuple
import numpy as np
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessage, HumanMessagePromptTemplate, MessagesPlaceholder
import spacy
import pandas as pd
from pyvis.network import Network
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeGraphBuilder:
    """
    Builds a knowledge graph from retrieved documents.
    
    The knowledge graph connects source documents to entities and
    creates relationships between entities based on their co-occurrence
    and semantic similarity.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.1,
        nlp_model: str = "en_core_web_sm"
    ):
        """
        Initialize the knowledge graph builder.
        
        Args:
            model_name: Name of the LLM model to use
            temperature: Temperature parameter for the LLM
            nlp_model: SpaCy model to use for entity extraction
        """
        # Initialize LLM
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        
        # Initialize NLP pipeline
        try:
            self.nlp = spacy.load(nlp_model)
            logger.info(f"Loaded SpaCy model: {nlp_model}")
        except OSError:
            logger.warning(f"SpaCy model {nlp_model} not found. Downloading...")
            spacy.cli.download(nlp_model)
            self.nlp = spacy.load(nlp_model)
        
        # Initialize graph
        self.graph = nx.Graph()
        
        # Track entities and documents
        self.documents = []
        self.entities = set()
        self.relationships = []
        
        # Node type colors
        self.node_colors = {
            "document": "#5DADE2",  # Blue
            "entity": "#AF7AC5",    # Purple
            "relationship": "#F5B041" # Orange
        }
    
    def build_from_documents(self, documents: List[Document], query: Optional[str] = None) -> nx.Graph:
        """
        Build a knowledge graph from a list of documents.
        
        Args:
            documents: List of Document objects
            query: Optional query string to help with relationship extraction
            
        Returns:
            NetworkX graph object
        """
        if not documents:
            logger.warning("No documents provided for knowledge graph building")
            return self.graph
        
        # Store documents
        self.documents = documents
        
        # Add query node if provided
        if query:
            self.graph.add_node(
                f"query:{query}", 
                type="query",
                label=query,
                color="#E74C3C"  # Red
            )
        
        # Process each document
        for doc in documents:
            # Add document node
            doc_id = doc.metadata.get("source_id", f"doc_{len(self.graph.nodes)}")
            doc_title = doc.metadata.get("title", f"Document {doc_id}")
            doc_node_id = f"doc:{doc_id}"
            
            # Add document node with metadata
            self.graph.add_node(
                doc_node_id,
                type="document",
                label=doc_title,
                color=self.node_colors["document"],
                metadata=doc.metadata,
                url=doc.metadata.get("url", "")
            )
            
            # Connect query to document if query exists
            if query:
                relevance = doc.metadata.get("relevance_score", 0.5)
                self.graph.add_edge(
                    f"query:{query}", 
                    doc_node_id,
                    weight=relevance,
                    title=f"Relevance: {relevance:.2f}"
                )
            
            # Process entities if they exist in metadata
            if "entities" in doc.metadata and doc.metadata["entities"]:
                self._process_document_entities(doc, doc_node_id)
            else:
                # Extract entities using SpaCy
                self._extract_and_add_entities(doc.page_content, doc_node_id)
                
            # Extract relationships using LLM
            self._extract_relationships(doc, doc_node_id)
        
        # Create connections between entities based on co-occurrence
        self._create_entity_connections()
        
        logger.info(
            f"Built knowledge graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges"
        )
        
        return self.graph
    
    def _process_document_entities(self, doc: Document, doc_node_id: str) -> None:
        """Process pre-extracted entities from document metadata."""
        entities = doc.metadata.get("entities", {})
        
        for entity_type, entity_list in entities.items():
            for entity_text in entity_list:
                entity_node_id = f"entity:{entity_text.lower()}"
                
                # Add entity node if it doesn't exist
                if entity_node_id not in self.graph:
                    self.graph.add_node(
                        entity_node_id,
                        type="entity",
                        label=entity_text,
                        entity_type=entity_type,
                        color=self.node_colors["entity"]
                    )
                    self.entities.add(entity_text.lower())
                
                # Connect document to entity
                if not self.graph.has_edge(doc_node_id, entity_node_id):
                    # Use relevance score as edge weight if available
                    relevance = doc.metadata.get("relevance_score", 0.5)
                    self.graph.add_edge(
                        doc_node_id, 
                        entity_node_id,
                        weight=relevance,
                        title=f"Contains entity ({entity_type})"
                    )
    
    def _extract_and_add_entities(self, text: str, doc_node_id: str) -> None:
        """Extract entities from text using SpaCy and add to graph."""
        nlp_doc = self.nlp(text[:10000])  # Limit to first 10k chars
        
        for ent in nlp_doc.ents:
            entity_text = ent.text
            entity_type = ent.label_
            entity_node_id = f"entity:{entity_text.lower()}"
            
            # Add entity node if it doesn't exist
            if entity_node_id not in self.graph:
                self.graph.add_node(
                    entity_node_id,
                    type="entity",
                    label=entity_text,
                    entity_type=entity_type,
                    color=self.node_colors["entity"]
                )
                self.entities.add(entity_text.lower())
            
            # Connect document to entity
            if not self.graph.has_edge(doc_node_id, entity_node_id):
                self.graph.add_edge(
                    doc_node_id, 
                    entity_node_id,
                    weight=0.5,
                    title=f"Contains entity ({entity_type})"
                )
    
    def _extract_relationships(self, doc: Document, doc_node_id: str) -> None:
        """Extract relationships from document using LLM."""
        # Skip if content is too short
        if len(doc.page_content) < 100:
            return
        
        try:
            # Create prompt for relationship extraction
            system_prompt = """You are an expert at analyzing text and extracting key relationships between entities. 
            Given a document, identify up to 5 most important relationships between entities in the text.
            For each relationship:
            1. Identify two entities that are related
            2. Describe the relationship between them in 2-4 words
            
            Format your response as a JSON array with the following structure:
            [
                {
                    "entity1": "First entity name",
                    "entity2": "Second entity name",
                    "relationship": "Description of relationship"
                },
                ...
            ]
            
            Only include clear, specific relationships that are explicitly stated or strongly implied in the text.
            """
            
            human_prompt = "Extract relationships from the following document:\n\n{text}"
            
            # Create the prompt template
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=system_prompt),
                HumanMessagePromptTemplate.from_template(human_prompt)
            ])
            
            # Truncate content if needed
            text = doc.page_content[:4000]  # Limit to 4000 chars for LLM
            
            # Generate relationships
            formatted_prompt = prompt.format_messages(text=text)
            response = self.llm.invoke(formatted_prompt)
            
            # Parse response
            try:
                relationships = json.loads(response.content)
                
                # Add relationships to graph
                for rel in relationships:
                    entity1 = rel.get("entity1", "").lower()
                    entity2 = rel.get("entity2", "").lower()
                    relationship = rel.get("relationship", "related to")
                    
                    if not entity1 or not entity2:
                        continue
                    
                    # Create entity nodes if they don't exist
                    for entity, name in [(f"entity:{entity1}", entity1), (f"entity:{entity2}", entity2)]:
                        if entity not in self.graph:
                            self.graph.add_node(
                                entity,
                                type="entity",
                                label=name,
                                color=self.node_colors["entity"]
                            )
                            self.entities.add(name)
                    
                    # Create relationship node
                    rel_id = f"rel:{entity1}_{relationship}_{entity2}"
                    self.graph.add_node(
                        rel_id,
                        type="relationship",
                        label=relationship,
                        color=self.node_colors["relationship"]
                    )
                    
                    # Connect entities to relationship
                    self.graph.add_edge(f"entity:{entity1}", rel_id, weight=0.8)
                    self.graph.add_edge(f"entity:{entity2}", rel_id, weight=0.8)
                    
                    # Connect document to relationship
                    self.graph.add_edge(doc_node_id, rel_id, weight=0.5)
                    
                    # Store relationship
                    self.relationships.append({
                        "entity1": entity1,
                        "entity2": entity2,
                        "relationship": relationship,
                        "document": doc_node_id
                    })
                    
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse relationship extraction response: {response.content}")
        
        except Exception as e:
            logger.error(f"Error extracting relationships: {str(e)}")
    
    def _create_entity_connections(self) -> None:
        """Create connections between entities based on co-occurrence."""
        # Count co-occurrences of entities in the same document
        entity_cooccurrence = Counter()
        
        # Get all entity nodes
        entity_nodes = [n for n, attr in self.graph.nodes(data=True) if attr.get("type") == "entity"]
        
        # For each document, find connected entities and create co-occurrence pairs
        doc_nodes = [n for n, attr in self.graph.nodes(data=True) if attr.get("type") == "document"]
        
        for doc_node in doc_nodes:
            # Get entities connected to this document
            connected_entities = []
            for neighbor in self.graph.neighbors(doc_node):
                if neighbor.startswith("entity:"):
                    connected_entities.append(neighbor)
            
            # Create co-occurrence pairs
            for i, entity1 in enumerate(connected_entities):
                for entity2 in connected_entities[i+1:]:
                    entity_cooccurrence[(entity1, entity2)] += 1
        
        # Add edges between entities that co-occur
        for (entity1, entity2), count in entity_cooccurrence.items():
            # Only connect if co-occurrence is at least 2
            if count >= 1:
                # Use co-occurrence count as weight
                weight = min(0.9, 0.3 + (count * 0.1))  # Scale weight between 0.3 and 0.9
                self.graph.add_edge(
                    entity1, 
                    entity2, 
                    weight=weight,
                    title=f"Co-occurs in {count} documents",
                    type="co-occurrence"
                )
    
    def visualize(self, output_path: str = "knowledge_graph.html") -> None:
        """
        Visualize the knowledge graph using pyvis.
        
        Args:
            output_path: Path to save the interactive HTML visualization
        """
        # Create network
        net = Network(height="800px", width="100%", notebook=False, directed=False)
        
        # Set physics layout
        net.barnes_hut(gravity=-80000, central_gravity=0.3, spring_length=250)
        
        # Add nodes
        for node, attr in self.graph.nodes(data=True):
            node_type = attr.get("type", "unknown")
            label = attr.get("label", node)
            title = f"Type: {node_type}<br>Label: {label}"
            
            # Add extra metadata for tooltips
            if node_type == "document":
                source = attr.get("metadata", {}).get("source", "Unknown")
                url = attr.get("url", "")
                title += f"<br>Source: {source}"
                if url:
                    title += f"<br>URL: <a href='{url}' target='_blank'>{url}</a>"
            
            size = 25
            if node_type == "document":
                size = 30
            elif node_type == "relationship":
                size = 15
            elif node_type == "query":
                size = 40
            
            net.add_node(
                node, 
                label=label, 
                title=title, 
                color=attr.get("color", "#AAAAAA"),
                size=size
            )
        
        # Add edges
        for u, v, attr in self.graph.edges(data=True):
            width = attr.get("weight", 0.5) * 5  # Scale for visualization
            title = attr.get("title", "")
            net.add_edge(u, v, width=width, title=title)
        
        # Set options
        net.set_options("""
        const options = {
            "nodes": {
                "font": {
                    "size": 12
                },
                "borderWidth": 2,
                "borderWidthSelected": 4,
                "shadow": true
            },
            "edges": {
                "color": {
                    "inherit": true
                },
                "smooth": {
                    "enabled": true,
                    "type": "dynamic"
                },
                "shadow": true
            },
            "physics": {
                "barnesHut": {
                    "gravitationalConstant": -80000,
                    "centralGravity": 0.3,
                    "springLength": 250,
                    "springConstant": 0.01,
                    "damping": 0.09
                },
                "minVelocity": 0.75
            }
        }
        """)
        
        # Save visualization
        try:
            net.save_graph(output_path)
            logger.info(f"Saved interactive visualization to {output_path}")
        except Exception as e:
            logger.error(f"Error saving visualization: {str(e)}")
    
    def export_graph_data(self, output_dir: str = "./knowledge_graph_data") -> Dict[str, str]:
        """
        Export graph data to various formats.
        
        Args:
            output_dir: Directory to save the exported files
            
        Returns:
            Dictionary with paths to exported files
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        export_paths = {}
        
        try:
            # Export nodes as CSV
            nodes_data = []
            for node, attr in self.graph.nodes(data=True):
                node_data = {
                    "id": node,
                    "type": attr.get("type", "unknown"),
                    "label": attr.get("label", node)
                }
                
                # Add extra attributes
                for key, value in attr.items():
                    if key not in ["type", "label", "color"] and not isinstance(value, dict):
                        node_data[key] = value
                
                nodes_data.append(node_data)
            
            nodes_df = pd.DataFrame(nodes_data)
            nodes_path = os.path.join(output_dir, "nodes.csv")
            nodes_df.to_csv(nodes_path, index=False)
            export_paths["nodes_csv"] = nodes_path
            
            # Export edges as CSV
            edges_data = []
            for u, v, attr in self.graph.edges(data=True):
                edge_data = {
                    "source": u,
                    "target": v,
                    "weight": attr.get("weight", 1.0),
                    "title": attr.get("title", ""),
                    "type": attr.get("type", "connection")
                }
                edges_data.append(edge_data)
            
            edges_df = pd.DataFrame(edges_data)
            edges_path = os.path.join(output_dir, "edges.csv")
            edges_df.to_csv(edges_path, index=False)
            export_paths["edges_csv"] = edges_path
            
            # Export relationships
            if self.relationships:
                rel_df = pd.DataFrame(self.relationships)
                rel_path = os.path.join(output_dir, "relationships.csv")
                rel_df.to_csv(rel_path, index=False)
                export_paths["relationships_csv"] = rel_path
            
            # Export graph as GraphML
            graphml_path = os.path.join(output_dir, "knowledge_graph.graphml")
            nx.write_graphml(self.graph, graphml_path)
            export_paths["graphml"] = graphml_path
            
            # Export graph as JSON for visualization tools
            graph_data = {
                "nodes": [],
                "links": []
            }
            
            for node, attr in self.graph.nodes(data=True):
                node_data = {
                    "id": node,
                    "label": attr.get("label", node),
                    "type": attr.get("type", "unknown"),
                    "color": attr.get("color", "#AAAAAA")
                }
                graph_data["nodes"].append(node_data)
            
            for u, v, attr in self.graph.edges(data=True):
                edge_data = {
                    "source": u,
                    "target": v,
                    "weight": attr.get("weight", 1.0),
                    "title": attr.get("title", "")
                }
                graph_data["links"].append(edge_data)
            
            json_path = os.path.join(output_dir, "knowledge_graph.json")
            with open(json_path, "w") as f:
                json.dump(graph_data, f, indent=2)
            export_paths["json"] = json_path
            
            logger.info(f"Exported graph data to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error exporting graph data: {str(e)}")
        
        return export_paths
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph.
        
        Returns:
            Dictionary with graph statistics
        """
        if not self.graph:
            return {"error": "No graph available"}
        
        try:
            # Count node types
            node_types = {}
            for _, attr in self.graph.nodes(data=True):
                node_type = attr.get("type", "unknown")
                node_types[node_type] = node_types.get(node_type, 0) + 1
            
            # Count edge types
            edge_types = {}
            for _, _, attr in self.graph.edges(data=True):
                edge_type = attr.get("type", "unknown")
                edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
            
            # Calculate degree statistics
            degrees = [d for _, d in self.graph.degree()]
            
            # Get connected components
            connected_components = list(nx.connected_components(self.graph))
            
            # Find central nodes
            centrality = nx.degree_centrality(self.graph)
            central_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                "num_nodes": self.graph.number_of_nodes(),
                "num_edges": self.graph.number_of_edges(),
                "node_types": node_types,
                "edge_types": edge_types,
                "avg_degree": sum(degrees) / len(degrees) if degrees else 0,
                "max_degree": max(degrees) if degrees else 0,
                "num_connected_components": len(connected_components),
                "largest_component_size": len(max(connected_components, key=len)) if connected_components else 0,
                "density": nx.density(self.graph),
                "central_nodes": [
                    {"id": node, "centrality": round(score, 3)} 
                    for node, score in central_nodes
                ]
            }
            
        except Exception as e:
            logger.error(f"Error calculating graph statistics: {str(e)}")
            return {"error": str(e)} 