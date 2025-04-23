"""
Enhanced Knowledge Graph Adapter

This module adapts the EnhancedKnowledgeGraphBuilder to the SocialSynth-AI project.
"""

import os
import logging
import networkx as nx
from typing import List, Dict, Any, Optional
from collections import Counter
import json

# For visualization
from pyvis.network import Network
import matplotlib.colors as mcolors

# Import Document class with fallback
try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document

logger = logging.getLogger("socialsynth.knowledge_graph")

# Import EnhancedKnowledgeGraphBuilder with fallback
try:
    from .enhanced_builder import EnhancedKnowledgeGraphBuilder
except ImportError:
    try:
        from src.socialsynth.knowledge_graph.enhanced_builder import EnhancedKnowledgeGraphBuilder
    except ImportError:
        logger.warning("Could not import EnhancedKnowledgeGraphBuilder, using base implementation")
        EnhancedKnowledgeGraphBuilder = None

class KnowledgeGraphBuilderAdapter:
    """
    Adapter for the EnhancedKnowledgeGraphBuilder that provides compatibility
    with the SocialSynth-AI project structure.
    """
    
    def __init__(
        self,
        relevance_threshold: float = 0.6,
        max_entities_per_doc: int = 20,
        max_keywords: int = 30,
        height: str = "600px",
        width: str = "100%"
    ):
        """
        Initialize the adapter with configuration parameters.
        
        Args:
            relevance_threshold: Minimum relevance score for entities (0.0 to 1.0)
            max_entities_per_doc: Maximum entities to extract per document
            max_keywords: Maximum keywords to include in the graph
            height: Height of the visualization
            width: Width of the visualization
        """
        self.relevance_threshold = relevance_threshold
        self.max_entities_per_doc = max_entities_per_doc
        self.max_keywords = max_keywords
        self.height = height
        self.width = width
        
        # Color mapping for different node types
        self.node_colors = {
            "document": "#3498db",  # Blue
            "concept": "#2ecc71",   # Green
            "person": "#e74c3c",    # Red
            "organization": "#f39c12",  # Orange
            "location": "#9b59b6",  # Purple
            "event": "#1abc9c",     # Turquoise
            "date": "#34495e",      # Dark blue
            "action": "#e67e22",    # Dark orange
            "other": "#95a5a6"      # Gray
        }
        
        # Initialize the builder
        try:
            self.builder = EnhancedKnowledgeGraphBuilder(
                relevance_threshold=relevance_threshold,
                max_entities_per_doc=max_entities_per_doc,
                max_keywords=max_keywords
            )
            logger.info("KnowledgeGraphBuilderAdapter initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing knowledge graph builder: {e}")
            self.builder = None
    
    def build_graph(self, documents: List[Document], query: Optional[str] = None) -> nx.DiGraph:
        """
        Build a knowledge graph from documents.
        
        Args:
            documents: List of Document objects
            query: Optional query for highlighting relevant nodes
            
        Returns:
            NetworkX directed graph
        """
        if not self.builder:
            logger.error("Knowledge graph builder is not initialized")
            return nx.DiGraph()
            
        if not documents:
            logger.warning("No documents provided for building knowledge graph")
            return nx.DiGraph()
            
        try:
            # Convert from various document formats
            processed_docs = []
            for doc in documents:
                try:
                    if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                        # LangChain Document objects
                        processed_docs.append({
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "title": doc.metadata.get("title", doc.metadata.get("source", "Document"))
                        })
                    elif isinstance(doc, dict):
                        # Dictionary format with different possible key names
                        content = doc.get('content', '')
                        if not content and 'page_content' in doc:
                            content = doc['page_content']
                        if not content and 'text' in doc:
                            content = doc['text']
                            
                        metadata = doc.get('metadata', {})
                        title = metadata.get('title', metadata.get('source', 'Document'))
                        
                        # Ensure we have at least some content
                        if content:
                            processed_docs.append({
                                "content": content,
                                "metadata": metadata,
                                "title": title
                            })
                        else:
                            logger.warning(f"Skipping document with no content: {metadata}")
                    else:
                        # Unknown format - try converting to string
                        logger.warning(f"Unknown document type: {type(doc)}, attempting string conversion")
                        processed_docs.append({
                            "content": str(doc),
                            "metadata": {"source": "unknown"},
                            "title": "Unknown Document"
                        })
                except Exception as e:
                    logger.error(f"Error processing document: {e}")
            
            if not processed_docs:
                logger.warning("No valid documents after processing")
                return nx.DiGraph()
                
            # Log document summary
            logger.info(f"Processing {len(processed_docs)} documents for knowledge graph")
            for i, doc in enumerate(processed_docs[:5]):  # Log first 5 docs
                title = doc.get("title", f"Doc {i}")
                content_preview = doc.get("content", "")[:50] + "..." if doc.get("content") else "No content"
                logger.info(f"Document {i}: {title} - {content_preview}")
            
            if len(processed_docs) > 5:
                logger.info(f"... and {len(processed_docs) - 5} more documents")
                
            # Build the graph
            graph = self.builder.build_from_documents(processed_docs)
            logger.info(f"Knowledge graph built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
            return graph
            
        except Exception as e:
            logger.error(f"Error building knowledge graph: {e}")
            return nx.DiGraph()
            
    def visualize(self, graph: nx.DiGraph, output_path: str) -> str:
        """
        Generate an interactive visualization of the knowledge graph.
        
        Args:
            graph: NetworkX directed graph to visualize
            output_path: Path to save the HTML visualization
            
        Returns:
            Path to the generated HTML file
        """
        if not graph or graph.number_of_nodes() == 0:
            logger.warning("Empty graph provided for visualization")
            return ""
            
        try:
            # Create a pyvis network
            net = Network(height=self.height, width=self.width, directed=True, notebook=False)
            
            # Configure physics for better layout
            physics_options = {
                "solver": "forceAtlas2Based",
                "forceAtlas2Based": {
                    "gravitationalConstant": -100,
                    "centralGravity": 0.01,
                    "springLength": 150,
                    "springConstant": 0.08,
                    "damping": 0.4,
                    "avoidOverlap": 0.8
                },
                "stabilization": {
                    "enabled": True,
                    "iterations": 1000,
                    "updateInterval": 50
                }
            }
            
            # Configure other options
            options = {
                "physics": physics_options,
                "interaction": {
                    "navigationButtons": True,
                    "keyboard": {
                        "enabled": True,
                        "speed": {"x": 10, "y": 10, "zoom": 0.1}
                    },
                    "tooltipDelay": 300,
                    "hideEdgesOnDrag": True,
                    "multiselect": True
                },
                "edges": {
                    "smooth": {
                        "type": "continuous",
                        "forceDirection": "none"
                    },
                    "arrows": {
                        "to": {"enabled": True, "scaleFactor": 0.5}
                    },
                    "color": {"inherit": False, "color": "#666666", "opacity": 0.8},
                    "font": {"size": 10, "align": "middle"}
                },
                "nodes": {
                    "shape": "dot",
                    "font": {"size": 12, "face": "Arial"},
                    "borderWidth": 2,
                    "shadow": True
                },
                "groups": {
                    "document": {"color": self.node_colors["document"], "shape": "square", "size": 25},
                    "concept": {"color": self.node_colors["concept"], "shape": "dot", "size": 15},
                    "person": {"color": self.node_colors["person"], "shape": "dot", "size": 15},
                    "organization": {"color": self.node_colors["organization"], "shape": "dot", "size": 15},
                    "location": {"color": self.node_colors["location"], "shape": "dot", "size": 15},
                    "event": {"color": self.node_colors["event"], "shape": "dot", "size": 15},
                    "date": {"color": self.node_colors["date"], "shape": "dot", "size": 15},
                    "action": {"color": self.node_colors["action"], "shape": "dot", "size": 15},
                    "other": {"color": self.node_colors["other"], "shape": "dot", "size": 15}
                }
            }
            
            try:
                # Set options - This is where 'dict' object has no attribute 'set' might occur
                logger.info("Setting pyvis network options")
                net.options = json.dumps(options)
            except AttributeError as ae:
                logger.error(f"AttributeError setting options: {ae}")
                # Check if options is properly handled
                logger.info(f"Network object type: {type(net)}")
                logger.info(f"Network object dir: {dir(net)}")
                # Try alternative method
                try:
                    logger.info("Trying alternative method for setting options")
                    net.set_options(json.dumps(options))
                except Exception as alt_e:
                    logger.error(f"Alternative method also failed: {alt_e}")
                    # Fall back to simple visualization without custom options
                    logger.info("Falling back to simple visualization")
            except Exception as e:
                logger.error(f"Error setting options: {e}")
            
            # Add nodes to the visualization
            logger.info(f"Adding {graph.number_of_nodes()} nodes to visualization")
            for node_id, node_data in graph.nodes(data=True):
                try:
                    node_type = node_data.get("type", "other")
                    label = node_data.get("label", str(node_id))
                    title = node_data.get("title", label)
                    
                    # Prepare node attributes
                    node_attrs = {
                        "label": label,
                        "title": title,
                        "group": node_type,
                    }
                    
                    # Add node with attributes
                    net.add_node(node_id, **node_attrs)
                except Exception as e:
                    logger.error(f"Error adding node {node_id}: {e}")
            
            # Add edges to the visualization
            logger.info(f"Adding {graph.number_of_edges()} edges to visualization")
            for from_id, to_id, edge_data in graph.edges(data=True):
                try:
                    weight = edge_data.get("weight", 1.0)
                    relation = edge_data.get("relation", "")
                    
                    # Add edge with attributes
                    net.add_edge(
                        from_id,
                        to_id,
                        title=relation if relation else "Related to",
                        width=max(1, min(5, weight * 3)),
                        color="#666666"
                    )
                except Exception as e:
                    logger.error(f"Error adding edge from {from_id} to {to_id}: {e}")
            
            # Generate the visualization
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Save the visualization
                logger.info(f"Saving visualization to {output_path}")
                net.save_graph(output_path)
                logger.info(f"Visualization saved successfully to {output_path}")
                
                return output_path
            except Exception as e:
                logger.error(f"Error saving visualization to {output_path}: {e}")
                # Try an alternative location
                try:
                    alt_path = os.path.join("static", "fallback_visualization.html")
                    logger.info(f"Trying alternative path: {alt_path}")
                    os.makedirs(os.path.dirname(alt_path), exist_ok=True)
                    net.save_graph(alt_path)
                    logger.info(f"Visualization saved to alternative path: {alt_path}")
                    return alt_path
                except Exception as alt_e:
                    logger.error(f"Error saving to alternative path: {alt_e}")
                    return ""
                
        except Exception as e:
            logger.error(f"Error visualizing knowledge graph: {e}")
            # Include traceback for more detailed error information
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return ""
    
    def get_graph_summary(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Generate a summary of the knowledge graph.
        
        Args:
            graph: NetworkX directed graph
            
        Returns:
            Dictionary containing summary information
        """
        if not graph:
            return {"error": "No graph provided"}
            
        try:
            # Count node types
            node_types = {}
            for node, attrs in graph.nodes(data=True):
                node_type = attrs.get('type', 'unknown')
                node_types[node_type] = node_types.get(node_type, 0) + 1
            
            # Calculate key metrics
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            
            # Find most central nodes
            try:
                centrality = nx.degree_centrality(graph)
                top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
                top_central_nodes = [{"id": node, "label": graph.nodes[node].get('label', node)} 
                                    for node, _ in top_central]
            except:
                top_central_nodes = []
            
            # Get top keywords
            top_keywords = []
            for node, attrs in graph.nodes(data=True):
                if attrs.get('type') == 'concept':
                    relevance = attrs.get('relevance', 0)
                    label = attrs.get('label', node)
                    top_keywords.append((label, relevance))
            
            top_keywords = sorted(top_keywords, key=lambda x: x[1], reverse=True)[:10]
            
            # Find most connected nodes
            degrees = dict(graph.degree())
            top_connected = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
            top_connected_nodes = [{"id": node, "label": graph.nodes[node].get('label', node), "connections": count} 
                                for node, count in top_connected]
            
            return {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "node_types": node_types,
                "top_central_nodes": top_central_nodes,
                "top_keywords": top_keywords,
                "top_connected_nodes": top_connected_nodes,
                "density": nx.density(graph),
            }
            
        except Exception as e:
            logger.error(f"Error generating graph summary: {e}")
            return {"error": str(e)}

class EnhancedKnowledgeGraphAdapter(KnowledgeGraphBuilderAdapter):
    """
    Enhanced adapter for the EnhancedKnowledgeGraphBuilder that extends the base adapter
    with additional features and optimizations for the SocialSynth-AI project.
    """
    
    def __init__(
        self,
        relevance_threshold: float = 0.3,
        max_entities_per_doc: int = 30,
        max_keywords: int = 20,
        height: str = "700px",
        width: str = "100%",
        use_gpu: bool = False
    ):
        """
        Initialize the enhanced adapter with configuration parameters.
        
        Args:
            relevance_threshold: Minimum relevance score for entities (0.0 to 1.0)
            max_entities_per_doc: Maximum entities to extract per document
            max_keywords: Maximum keywords to include in the graph
            height: Height of the visualization
            width: Width of the visualization
            use_gpu: Whether to use GPU acceleration for NLP processing
        """
        # Initialize the base adapter
        super().__init__(
            relevance_threshold=relevance_threshold,
            max_entities_per_doc=max_entities_per_doc,
            max_keywords=max_keywords,
            height=height,
            width=width
        )
        
        # Initialize the enhanced builder with GPU support
        try:
            self.builder = EnhancedKnowledgeGraphBuilder(
                relevance_threshold=relevance_threshold,
                max_entities_per_doc=max_entities_per_doc,
                max_keywords=max_keywords,
                use_gpu=use_gpu
            )
            logger.info("EnhancedKnowledgeGraphAdapter initialized successfully with GPU support")
        except Exception as e:
            logger.error(f"Error initializing enhanced knowledge graph builder: {e}")
            self.builder = None
            
    def build_graph(self, documents: List[Document], query: Optional[str] = None) -> nx.DiGraph:
        """
        Build an enhanced knowledge graph from documents with additional processing steps.
        
        Args:
            documents: List of Document objects
            query: Optional query for highlighting relevant nodes
            
        Returns:
            NetworkX directed graph
        """
        # Log the start of graph building with enhanced features
        logger.info(f"Building enhanced knowledge graph from {len(documents)} documents")
        
        # Convert documents to the expected format
        processed_docs = []
        for doc in documents:
            try:
                if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                    # Handle LangChain Document objects
                    processed_docs.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "title": doc.metadata.get("title", doc.metadata.get("source", "Document"))
                    })
                elif isinstance(doc, dict):
                    # Handle dictionary format with different possible keys
                    content = doc.get('content', doc.get('page_content', ''))
                    metadata = doc.get('metadata', {})
                    title = metadata.get('title', metadata.get('source', 'Document'))
                    
                    if not content and 'text' in doc:
                        content = doc['text']
                    
                    processed_docs.append({
                        "content": content,
                        "metadata": metadata,
                        "title": title
                    })
                else:
                    # Last resort - try to convert to string
                    logger.warning(f"Unknown document format, attempting string conversion: {type(doc)}")
                    processed_docs.append({
                        "content": str(doc),
                        "metadata": {"source": "unknown"},
                        "title": "Unknown Document"
                    })
            except Exception as e:
                logger.error(f"Error processing document for graph building: {e}")
        
        if not processed_docs:
            logger.error("No documents could be processed for knowledge graph")
            return nx.DiGraph()
            
        logger.info(f"Successfully processed {len(processed_docs)} documents for knowledge graph")
        
        # Call parent method with properly processed documents
        try:
            # Build the base graph
            self.builder = EnhancedKnowledgeGraphBuilder(
                relevance_threshold=self.relevance_threshold,
                max_entities_per_doc=self.max_entities_per_doc,
                max_keywords=self.max_keywords,
                use_gpu=False  # Set to False for reliability
            )
            
            # Build the graph directly with our builder
            graph = self.builder.build_from_documents(processed_docs)
            
            # Perform additional processing on the graph if it's not empty
            if graph and graph.number_of_nodes() > 0:
                try:
                    # Find the central nodes (entities with high centrality)
                    centrality = nx.betweenness_centrality(graph)
                    
                    # Mark central nodes in the graph
                    for node, score in centrality.items():
                        if node in graph.nodes and score > 0.05:  # Threshold for central nodes
                            graph.nodes[node]['is_central'] = True
                            graph.nodes[node]['centrality'] = score
                    
                    # Log additional graph metrics
                    logger.info(f"Enhanced graph built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
                    logger.info(f"Found {sum(1 for _, data in graph.nodes(data=True) if data.get('is_central', False))} central nodes")
                    
                except Exception as e:
                    logger.error(f"Error in enhanced graph post-processing: {e}")
            else:
                logger.warning("Graph is empty after building")
                
            return graph
            
        except Exception as e:
            logger.error(f"Error building knowledge graph: {e}")
            return nx.DiGraph()
    
    def visualize(self, graph: nx.DiGraph, output_path: str) -> str:
        """
        Generate an enhanced interactive visualization of the knowledge graph.
        
        Args:
            graph: NetworkX directed graph to visualize
            output_path: Path to save the HTML visualization
            
        Returns:
            Path to the generated HTML file
        """
        if not graph or graph.number_of_nodes() == 0:
            logger.warning("Empty graph provided for enhanced visualization")
            return ""
            
        try:
            # Create a pyvis network
            net = Network(height=self.height, width=self.width, directed=True, notebook=False)
            
            # Configure enhanced physics for better layout
            physics_options = {
                "solver": "forceAtlas2Based",
                "forceAtlas2Based": {
                    "gravitationalConstant": -120,
                    "centralGravity": 0.01,
                    "springLength": 180,
                    "springConstant": 0.1,
                    "damping": 0.3,
                    "avoidOverlap": 1.0
                },
                "stabilization": {
                    "enabled": True,
                    "iterations": 1500,
                    "updateInterval": 25
                }
            }
            
            # Configure enhanced options
            options = {
                "physics": physics_options,
                "interaction": {
                    "navigationButtons": True,
                    "keyboard": {
                        "enabled": True,
                        "speed": {"x": 10, "y": 10, "zoom": 0.1}
                    },
                    "tooltipDelay": 200,
                    "hideEdgesOnDrag": True,
                    "multiselect": True,
                    "dragNodes": True,
                    "hover": True
                },
                "edges": {
                    "smooth": {
                        "type": "continuous",
                        "forceDirection": "none"
                    },
                    "arrows": {
                        "to": {"enabled": True, "scaleFactor": 0.5}
                    },
                    "color": {"inherit": False, "color": "#666666", "opacity": 0.8},
                    "font": {"size": 12, "align": "middle"}
                },
                "nodes": {
                    "shape": "dot",
                    "font": {"size": 14, "face": "Arial", "bold": False},
                    "borderWidth": 2,
                    "shadow": True
                },
                "groups": {
                    "document": {"color": self.node_colors["document"], "shape": "square", "size": 25},
                    "concept": {"color": self.node_colors["concept"], "size": 15},
                    "person": {"color": self.node_colors["person"], "size": 20},
                    "organization": {"color": self.node_colors["organization"], "size": 20},
                    "location": {"color": self.node_colors["location"], "size": 18},
                    "event": {"color": self.node_colors["event"], "size": 18},
                    "date": {"color": self.node_colors["date"], "size": 18},
                    "action": {"color": self.node_colors["action"], "size": 15},
                    "other": {"color": self.node_colors["other"], "size": 15}
                }
            }
            
            try:
                # Use set_options method instead of directly setting options property
                # This fixes the 'str' object has no attribute 'to_json' error
                net.set_options(json.dumps(options))
            except Exception as e:
                logger.error(f"EnhancedAdapter: Error setting options: {e}")
                # Try alternative method for setting options
                try:
                    logger.info("Trying alternative approach for network options")
                    # Apply individual physics settings
                    net.barnes_hut(
                        gravity=-120,
                        central_gravity=0.01,
                        spring_length=180,
                        spring_strength=0.1,
                        damping=0.3
                    )
                except Exception as alt_e:
                    logger.error(f"Alternative approach also failed: {alt_e}")
            
            # Add nodes to the visualization with enhanced styling
            logger.info(f"Adding {graph.number_of_nodes()} nodes to visualization (enhanced)")
            node_count = 0
            for node_id, node_data in graph.nodes(data=True):
                try:
                    node_type = node_data.get("type", "other")
                    label = node_data.get("label", node_id)
                    title = node_data.get("title", label)
                    is_central = node_data.get("is_central", False)
                    centrality = node_data.get("centrality", 0.0)
                    
                    # Customize node size based on centrality
                    size = None
                    if is_central:
                        # Increase size based on centrality
                        base_size = 15
                        if node_type == "document":
                            base_size = 25
                        elif node_type in ["person", "organization"]:
                            base_size = 20
                        size = base_size + (centrality * 20)
                        
                        # Add a title that includes centrality information
                        if title:
                            title = f"{title}\nCentrality: {centrality:.3f}"
                    
                    # Prepare node attributes
                    node_attrs = {
                        "label": label,
                        "title": title,
                        "group": node_type,
                    }
                    
                    # Add size if customized
                    if size:
                        node_attrs["size"] = size
                        
                    # Add bold font for central nodes
                    if is_central:
                        node_attrs["font"] = {"bold": True}
                    
                    # Add node with attributes
                    net.add_node(node_id, **node_attrs)
                    node_count += 1
                except Exception as e:
                    logger.error(f"EnhancedAdapter: Error adding node {node_id}: {e}")
            
            logger.info(f"Successfully added {node_count} nodes to visualization")
            
            # Add edges to the visualization with enhanced styling
            logger.info(f"Adding {graph.number_of_edges()} edges to visualization (enhanced)")
            edge_count = 0
            for from_id, to_id, edge_data in graph.edges(data=True):
                try:
                    weight = edge_data.get("weight", 1.0)
                    relation = edge_data.get("relation", "")
                    
                    # Check if edge connects two central nodes
                    from_central = graph.nodes[from_id].get("is_central", False) if from_id in graph.nodes else False
                    to_central = graph.nodes[to_id].get("is_central", False) if to_id in graph.nodes else False
                    
                    # Customize edge width and color based on nodes it connects
                    width = max(1, min(6, weight * 5))
                    color = "#666666"
                    
                    if from_central and to_central:
                        # Highlight edges between central nodes
                        width = max(width, 3)
                        color = "#ff6600"  # Bright orange for important connections
                    
                    # Add edge with appropriate styling
                    net.add_edge(
                        from_id,
                        to_id,
                        title=relation if relation else "Related to",
                        width=width,
                        color=color,
                        label=relation if relation else ""
                    )
                    edge_count += 1
                except Exception as e:
                    logger.error(f"EnhancedAdapter: Error adding edge from {from_id} to {to_id}: {e}")
            
            logger.info(f"Successfully added {edge_count} edges to visualization")
            
            # Generate the visualization
            try:
                # Create directory if it doesn't exist
                logger.info(f"Creating directory for output path: {os.path.dirname(output_path)}")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Add title and description to the visualization
                html_template = """
                <center>
                <h2>Enhanced Knowledge Graph Visualization</h2>
                <p>This interactive visualization shows entities, concepts, and their relationships extracted from the documents.</p>
                </center>
                """
                
                # Save the network to HTML file directly without additional options
                logger.info(f"Saving enhanced visualization to {output_path}")
                try:
                    # Make sure output directory exists
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    # Try using show method with notebook=False argument
                    net.show(output_path, notebook=False)
                    logger.info(f"Enhanced visualization saved successfully using show method")
                except Exception as show_err:
                    logger.error(f"Error using show method: {show_err}", exc_info=True)
                    try:
                        # Fall back to save_graph
                        net.save_graph(output_path)
                        logger.info(f"Enhanced visualization saved successfully using save_graph method")
                    except Exception as save_err:
                        logger.error(f"Error using save_graph method: {save_err}", exc_info=True)
                        # Try writing HTML directly
                        try:
                            html_data = net.generate_html()
                            with open(output_path, 'w', encoding='utf-8') as f:
                                f.write(html_data)
                            logger.info(f"Enhanced visualization saved by directly writing HTML")
                        except Exception as html_err:
                            logger.error(f"Error writing HTML directly: {html_err}", exc_info=True)
                            raise
                
                # Try to add HTML template if the file was created
                if os.path.exists(output_path):
                    try:
                        with open(output_path, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                        
                        # Insert our template after the <body> tag
                        if '<body>' in html_content:
                            modified_html = html_content.replace('<body>', f'<body>\n{html_template}')
                            
                            # Write the modified HTML back
                            with open(output_path, 'w', encoding='utf-8') as f:
                                f.write(modified_html)
                        
                        logger.info(f"Enhanced knowledge graph visualization saved to {output_path}")
                    except Exception as e:
                        logger.error(f"Error modifying HTML file: {e}")
                        # Still return the path as the visualization was saved
                
                return output_path
            except Exception as e:
                logger.error(f"Error saving enhanced visualization: {e}", exc_info=True)
                import traceback
                logger.error(f"Save visualization traceback: {traceback.format_exc()}")
                
                # Try creating a simpler visualization as fallback
                try:
                    # Try a different approach - create a simpler visualization
                    alt_path = os.path.join("static", "fallback_enhanced_viz.html")
                    logger.info(f"Trying alternative path with simpler visualization: {alt_path}")
                    os.makedirs(os.path.dirname(alt_path), exist_ok=True)
                    
                    # Create a basic HTML file with graph information
                    basic_html = f"""
                    <html>
                    <head>
                        <title>Knowledge Graph Visualization (Fallback)</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; margin: 20px; }}
                            .container {{ max-width: 800px; margin: 0 auto; }}
                            .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
                            .node-list {{ margin-top: 15px; column-count: 2; }}
                            .node-item {{ padding: 3px 0; }}
                        </style>
                    </head>
                    <body>
                        <div class="container">
                            <h1>Knowledge Graph Visualization</h1>
                            <p>The interactive visualization could not be generated, but here's a summary of the graph:</p>
                            
                            <div class="summary">
                                <h3>Graph Summary</h3>
                                <p><strong>Nodes:</strong> {graph.number_of_nodes()}</p>
                                <p><strong>Edges:</strong> {graph.number_of_edges()}</p>
                            </div>
                            
                            <h3>Top Nodes</h3>
                            <div class="node-list">
                    """
                    
                    # Add top 20 nodes by degree
                    node_degrees = sorted([(n, len(list(graph.neighbors(n)))) for n in graph.nodes()], 
                                         key=lambda x: x[1], reverse=True)[:20]
                    
                    for node, degree in node_degrees:
                        node_label = graph.nodes[node].get("label", str(node))
                        node_type = graph.nodes[node].get("type", "unknown")
                        basic_html += f'<div class="node-item"><strong>{node_label}</strong> ({node_type}) - {degree} connections</div>\n'
                    
                    basic_html += """
                            </div>
                        </div>
                    </body>
                    </html>
                    """
                    
                    with open(alt_path, 'w', encoding='utf-8') as f:
                        f.write(basic_html)
                        
                    logger.info(f"Created basic fallback visualization at {alt_path}")
                    return alt_path
                except Exception as alt_e:
                    logger.error(f"Error creating fallback visualization: {alt_e}")
                    return ""
                
        except Exception as e:
            logger.error(f"Error visualizing enhanced knowledge graph: {e}")
            import traceback
            logger.error(f"EnhancedAdapter visualization traceback: {traceback.format_exc()}")
            return ""
            
    def get_graph_summary(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Generate an enhanced summary of the knowledge graph.
        
        Args:
            graph: NetworkX directed graph
            
        Returns:
            Dictionary with detailed summary information
        """
        # Get basic summary from parent class
        base_summary = super().get_graph_summary(graph)
        
        # Return early if graph is empty
        if not graph or graph.number_of_nodes() == 0:
            return base_summary
            
        try:
            # Enhance the summary with additional metrics
            enhanced_summary = base_summary.copy()
            
            # Calculate centrality measures
            try:
                centrality = nx.betweenness_centrality(graph)
                # Get top central entities
                top_central = [{"id": k, "label": graph.nodes[k].get("label", k), "centrality": v}
                              for k, v in sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]]
                enhanced_summary["top_central_entities"] = top_central
            except Exception as e:
                logger.warning(f"Could not calculate centrality: {e}")
            
            # Calculate community detection using connected components
            try:
                communities = list(nx.connected_components(graph.to_undirected()))
                enhanced_summary["communities"] = len(communities)
                enhanced_summary["largest_community_size"] = len(max(communities, key=len)) if communities else 0
            except Exception as e:
                logger.warning(f"Could not identify communities: {e}")
            
            # Calculate network density
            try:
                enhanced_summary["network_density"] = nx.density(graph)
            except Exception as e:
                logger.warning(f"Could not calculate network density: {e}")
            
            # Add relationship type statistics
            try:
                relation_types = Counter()
                for _, _, data in graph.edges(data=True):
                    relation = data.get("relation", "unknown")
                    relation_types[relation] += 1
                
                # Get top relation types
                enhanced_summary["relation_types"] = dict(relation_types.most_common(5))
            except Exception as e:
                logger.warning(f"Could not analyze relation types: {e}")
            
            return enhanced_summary
            
        except Exception as e:
            logger.error(f"Error generating enhanced graph summary: {e}")
            return base_summary 