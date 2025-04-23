#!/usr/bin/env python3
"""
Enhanced RAG Retriever
This module provides an improved retrieval mechanism that combines vector similarity search
with knowledge graph navigation for more contextually relevant document retrieval.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set
import networkx as nx
from langchain_community.vectorstores import Chroma
from enhanced_knowledge_graph_v2 import EnhancedKnowledgeGraphBuilder

# Configure logging
logger = logging.getLogger(__name__)

class EnhancedRetriever:
    """
    An enhanced retriever that combines vector store similarity search with
    knowledge graph navigation to improve retrieval quality.
    """
    
    def __init__(
        self, 
        vector_store: Optional[Chroma] = None,
        knowledge_graph: Optional[nx.DiGraph] = None,
        top_k_vector: int = 5,
        top_k_graph: int = 3,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize the enhanced retriever.
        
        Args:
            vector_store: The vector store for similarity search
            knowledge_graph: The knowledge graph for navigation
            top_k_vector: Number of documents to retrieve from vector store
            top_k_graph: Number of additional documents to retrieve through graph
            similarity_threshold: Threshold for similarity search
        """
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.top_k_vector = top_k_vector
        self.top_k_graph = top_k_graph
        self.similarity_threshold = similarity_threshold
        logger.info("Enhanced Retriever initialized")
        
    def set_vector_store(self, vector_store: Chroma) -> None:
        """Set the vector store for similarity search."""
        self.vector_store = vector_store
        logger.info("Vector store set in EnhancedRetriever")
        
    def set_knowledge_graph(self, knowledge_graph: nx.DiGraph) -> None:
        """Set the knowledge graph for navigation."""
        self.knowledge_graph = knowledge_graph
        logger.info(f"Knowledge graph set in EnhancedRetriever: {len(knowledge_graph.nodes)} nodes")
        
    def retrieve(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve documents using both vector similarity and knowledge graph navigation.
        
        Args:
            query: The query string
            **kwargs: Additional parameters for retrieval
            
        Returns:
            A list of document dictionaries with content and metadata
        """
        if not self.vector_store:
            logger.warning("Vector store not set, cannot retrieve documents")
            return []
            
        # Phase 1: Vector similarity search
        logger.info(f"Performing vector similarity search for query: {query}")
        similarity_docs = self._retrieve_by_similarity(query, **kwargs)
        
        # Phase 2: Knowledge graph navigation (if available)
        graph_docs = []
        if self.knowledge_graph:
            logger.info("Performing knowledge graph navigation")
            # Extract document IDs from similarity search results
            seed_doc_ids = self._extract_doc_ids(similarity_docs)
            # Find related documents through graph navigation
            graph_docs = self._retrieve_by_graph(query, seed_doc_ids, **kwargs)
        
        # Combine and deduplicate results
        all_docs = self._merge_and_deduplicate(similarity_docs, graph_docs)
        
        logger.info(f"Enhanced retrieval complete, returned {len(all_docs)} documents")
        return all_docs
        
    def _retrieve_by_similarity(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve documents based on vector similarity.
        
        Args:
            query: The query string
            **kwargs: Additional parameters for similarity search
            
        Returns:
            A list of document dictionaries from vector similarity search
        """
        try:
            # Get the top_k_vector most similar documents
            top_k = kwargs.get('top_k_vector', self.top_k_vector)
            threshold = kwargs.get('similarity_threshold', self.similarity_threshold)
            
            # Perform vector store search
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query, k=top_k
            )
            
            # Convert to standard document format
            documents = []
            for doc, score in docs_with_scores:
                # Filter by similarity threshold
                if score >= threshold:
                    documents.append({
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'score': score,
                        'source': 'vector_similarity'
                    })
            
            logger.debug(f"Retrieved {len(documents)} documents from vector store")
            return documents
            
        except Exception as e:
            logger.error(f"Error in vector similarity search: {str(e)}")
            return []
            
    def _extract_doc_ids(self, documents: List[Dict[str, Any]]) -> Set[str]:
        """
        Extract document IDs from retrieved documents.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            Set of document identifiers
        """
        doc_ids = set()
        for doc in documents:
            # Try to extract ID from metadata
            metadata = doc.get('metadata', {})
            doc_id = metadata.get('source', None)
            if doc_id:
                doc_ids.add(doc_id)
        
        return doc_ids
        
    def _retrieve_by_graph(
        self, 
        query: str, 
        seed_doc_ids: Set[str], 
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Retrieve additional documents through knowledge graph navigation.
        
        Args:
            query: The query string
            seed_doc_ids: Set of seed document IDs from initial retrieval
            **kwargs: Additional parameters for graph navigation
            
        Returns:
            A list of additional document dictionaries from graph navigation
        """
        if not self.knowledge_graph or not seed_doc_ids:
            return []
            
        try:
            # Extract query entities - simplified approach
            query_terms = set(query.lower().split())
            
            # Find entity nodes that match query terms
            matching_entities = []
            for node, attrs in self.knowledge_graph.nodes(data=True):
                # Skip document nodes
                if attrs.get('type') == 'document':
                    continue
                    
                # Check if node label matches any query terms
                label = attrs.get('label', '').lower()
                if label and any(term in label for term in query_terms):
                    matching_entities.append(node)
            
            # Find documents connected to matching entities
            related_doc_ids = set()
            for entity in matching_entities:
                # Add documents directly connected to matching entities
                for neighbor in self.knowledge_graph.neighbors(entity):
                    if self.knowledge_graph.nodes[neighbor].get('type') == 'document':
                        related_doc_ids.add(neighbor)
                
                # Also find documents connected through other entities
                # This is a two-hop search in the graph
                for connected_entity in self.knowledge_graph.neighbors(entity):
                    if self.knowledge_graph.nodes[connected_entity].get('type') != 'document':
                        for doc in self.knowledge_graph.neighbors(connected_entity):
                            if self.knowledge_graph.nodes[doc].get('type') == 'document':
                                related_doc_ids.add(doc)
            
            # Filter out seed documents to avoid duplication
            new_doc_ids = related_doc_ids - seed_doc_ids
            
            # Limit to top_k_graph additional documents
            top_k = kwargs.get('top_k_graph', self.top_k_graph)
            selected_doc_ids = list(new_doc_ids)[:top_k]
            
            # Convert document IDs to document dictionaries
            graph_docs = []
            for doc_id in selected_doc_ids:
                doc_attrs = self.knowledge_graph.nodes[doc_id]
                # Find the corresponding document in the vector store or source
                # This is a simplified approach - in practice you would need to 
                # retrieve the actual document content from your document store
                if 'metadata' in doc_attrs:
                    graph_docs.append({
                        'content': f"Document content for {doc_id}",  # Placeholder
                        'metadata': doc_attrs['metadata'],
                        'score': 0.7,  # Default score for graph-retrieved docs
                        'source': 'knowledge_graph'
                    })
            
            logger.debug(f"Retrieved {len(graph_docs)} additional documents from knowledge graph")
            return graph_docs
            
        except Exception as e:
            logger.error(f"Error in knowledge graph navigation: {str(e)}")
            return []
            
    def _merge_and_deduplicate(
        self, 
        similarity_docs: List[Dict[str, Any]], 
        graph_docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge and deduplicate documents from different sources.
        
        Args:
            similarity_docs: Documents from vector similarity search
            graph_docs: Documents from knowledge graph navigation
            
        Returns:
            A merged and deduplicated list of documents
        """
        # Combine all documents
        all_docs = similarity_docs + graph_docs
        
        # Deduplicate by document ID/source
        deduplicated = {}
        for doc in all_docs:
            doc_id = doc.get('metadata', {}).get('source', None)
            if doc_id:
                # Keep the highest scoring document if duplicate
                if doc_id not in deduplicated or doc.get('score', 0) > deduplicated[doc_id].get('score', 0):
                    deduplicated[doc_id] = doc
        
        # Sort by score
        sorted_docs = sorted(
            deduplicated.values(), 
            key=lambda x: x.get('score', 0), 
            reverse=True
        )
        
        return sorted_docs
        
    def build_knowledge_graph(self, documents: List[Dict[str, Any]]) -> nx.DiGraph:
        """
        Build a knowledge graph from documents.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            A NetworkX directed graph representing the knowledge graph
        """
        try:
            # Create a knowledge graph builder
            kg_builder = EnhancedKnowledgeGraphBuilder()
            
            # Build the graph
            graph = kg_builder.build_from_documents(documents)
            
            # Set the graph in the retriever
            self.set_knowledge_graph(graph)
            
            return graph
            
        except Exception as e:
            logger.error(f"Error building knowledge graph: {str(e)}")
            return nx.DiGraph()  # Return empty graph on error 