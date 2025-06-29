"""
Vector store implementations with citations support.

This module provides vector store interfaces and implementations that support
both basic search and search with citations for source attribution.
"""

import numpy as np
from typing import List, Dict, Optional, Protocol, Tuple, Any, Union
from pathlib import Path
import json
import logging
from dataclasses import dataclass
from abc import abstractmethod
import time

from .citations import Citation, CitationType, SearchResultWithCitations

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Basic search result for backward compatibility."""
    id: str
    score: float
    document: str
    metadata: dict

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "score": self.score,
            "document": self.document,
            "metadata": self.metadata
        }


class VectorStore(Protocol):
    """
    Protocol defining vector store interface with citations support.
    
    Using Protocol instead of ABC for lighter weight and better type checking.
    """
    
    def add(self, ids: List[str], embeddings: np.ndarray, 
            documents: List[str], metadata: List[dict]) -> None:
        """Add documents with embeddings to the store."""
        ...
    
    def search(self, query_embedding: np.ndarray, 
               top_k: int = 10) -> List[SearchResult]:
        """Basic search without citations (backward compatibility)."""
        ...
    
    def search_with_citations(self, query_embedding: np.ndarray,
                            query_text: str,
                            top_k: int = 10) -> List[SearchResultWithCitations]:
        """Enhanced search that includes citations."""
        ...
    
    def delete_collection(self, name: str) -> None:
        """Delete a collection by name."""
        ...
    
    def get_collection_stats(self) -> dict:
        """Get statistics about the collection."""
        ...
    
    def clear(self) -> None:
        """Clear all data from the store."""
        ...


class ChromaVectorStore:
    """
    ChromaDB implementation with citations support.
    
    Provides persistent vector storage with metadata support for citations.
    """
    
    def __init__(self, 
                 persist_directory: Union[str, Path], 
                 collection_name: str = "default",
                 distance_metric: str = "cosine"):
        """
        Initialize ChromaDB vector store.
        
        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the collection to use
            distance_metric: Distance metric for similarity (cosine, l2, ip)
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.distance_metric = distance_metric
        self._client = None
        self._collection = None
        
        # Ensure persist directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Metrics
        self._add_count = 0
        self._search_count = 0
        self._last_operation_time = None
    
    @property
    def client(self):
        """Lazy load ChromaDB client."""
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings
                
                settings = Settings(
                    persist_directory=str(self.persist_directory),
                    anonymized_telemetry=False,
                    allow_reset=True
                )
                self._client = chromadb.Client(settings)
                logger.info(f"Initialized ChromaDB client at {self.persist_directory}")
                
            except ImportError:
                raise ImportError(
                    "ChromaDB not installed. "
                    "Install with: pip install chromadb"
                )
        return self._client
    
    @property
    def collection(self):
        """Get or create collection."""
        if self._collection is None:
            # Map distance metrics to ChromaDB's hnsw:space parameter
            metric_map = {
                "cosine": "cosine",
                "l2": "l2",
                "ip": "ip"  # inner product
            }
            
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": metric_map.get(self.distance_metric, "cosine")}
            )
            logger.info(f"Using collection: {self.collection_name}")
        return self._collection
    
    def add(self, 
            ids: List[str], 
            embeddings: np.ndarray, 
            documents: List[str], 
            metadata: List[dict]) -> None:
        """
        Add documents to the collection.
        
        Metadata should include fields needed for citations:
        - doc_id: Original document ID
        - doc_title: Human-readable document title
        - chunk_start: Character offset in original document
        - chunk_end: End character offset
        - chunk_index: Index of this chunk
        - author, date, url: Optional citation metadata
        """
        if len(ids) == 0:
            logger.warning("add() called with empty documents")
            return
        
        # Validate inputs
        if len(ids) != len(documents) or len(ids) != len(metadata):
            raise ValueError("ids, documents, and metadata must have same length")
        
        if isinstance(embeddings, np.ndarray):
            if embeddings.shape[0] != len(ids):
                raise ValueError(f"embeddings shape {embeddings.shape} doesn't match {len(ids)} documents")
            embeddings = embeddings.tolist()
        
        # Ensure metadata is properly formatted
        processed_metadata = []
        for i, meta in enumerate(metadata):
            # ChromaDB requires all metadata values to be strings, ints, or floats
            processed_meta = {}
            for key, value in meta.items():
                if value is None:
                    continue
                elif isinstance(value, (str, int, float)):
                    processed_meta[key] = value
                else:
                    # Convert other types to string
                    processed_meta[key] = str(value)
            processed_metadata.append(processed_meta)
        
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=processed_metadata
            )
            
            self._add_count += len(ids)
            self._last_operation_time = time.time()
            logger.info(f"Added {len(ids)} documents to collection")
            
        except Exception as e:
            logger.error(f"Failed to add to ChromaDB: {e}")
            raise
    
    def search(self, 
               query_embedding: np.ndarray, 
               top_k: int = 10) -> List[SearchResult]:
        """Basic search without citations (backward compatibility)."""
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            search_results = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    search_results.append(SearchResult(
                        id=results['ids'][0][i],
                        score=1 - results['distances'][0][i],  # Convert distance to similarity
                        document=results['documents'][0][i],
                        metadata=results['metadatas'][0][i] or {}
                    ))
            
            self._search_count += 1
            self._last_operation_time = time.time()
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def search_with_citations(self, 
                            query_embedding: np.ndarray,
                            query_text: str,
                            top_k: int = 10,
                            score_threshold: float = 0.0) -> List[SearchResultWithCitations]:
        """
        Enhanced search that includes citations.
        
        Args:
            query_embedding: Query vector
            query_text: Original query text (used for citation context)
            top_k: Number of results to return
            score_threshold: Minimum score threshold for results
            
        Returns:
            List of results with citations
        """
        # First, get basic search results
        basic_results = self.search(query_embedding, top_k * 2)  # Get extra for filtering
        
        # Filter by score threshold
        filtered_results = [r for r in basic_results if r.score >= score_threshold]
        
        # Convert to results with citations
        results_with_citations = []
        
        for result in filtered_results[:top_k]:
            # Create citations from metadata
            citations = self._create_citations_from_result(result, query_text)
            
            # Create result with citations
            result_with_citations = SearchResultWithCitations(
                id=result.id,
                score=result.score,
                document=result.document,
                metadata=result.metadata,
                citations=citations
            )
            results_with_citations.append(result_with_citations)
        
        logger.debug(f"Search with citations returned {len(results_with_citations)} results")
        return results_with_citations
    
    def _create_citations_from_result(self, 
                                    result: SearchResult, 
                                    query_text: str) -> List[Citation]:
        """Create citations from a search result."""
        citations = []
        metadata = result.metadata
        
        # Create semantic citation for the chunk
        citation = Citation(
            document_id=metadata.get("doc_id", result.id),
            document_title=metadata.get("doc_title", "Unknown Document"),
            chunk_id=result.id,
            text=result.document[:300] + "..." if len(result.document) > 300 else result.document,
            start_char=int(metadata.get("chunk_start", 0)),
            end_char=int(metadata.get("chunk_end", len(result.document))),
            confidence=result.score,
            match_type=CitationType.SEMANTIC,
            metadata={
                "author": metadata.get("author"),
                "date": metadata.get("date"),
                "url": metadata.get("url"),
                "chunk_index": metadata.get("chunk_index", 0),
                "query": query_text  # Include query for context
            }
        )
        citations.append(citation)
        
        # If we have keyword matches in metadata, add exact citations
        if "keyword_matches" in metadata:
            try:
                keyword_matches = json.loads(metadata["keyword_matches"])
                for match in keyword_matches:
                    exact_citation = Citation(
                        document_id=metadata.get("doc_id", result.id),
                        document_title=metadata.get("doc_title", "Unknown Document"),
                        chunk_id=result.id,
                        text=match.get("text", ""),
                        start_char=int(match.get("start", 0)),
                        end_char=int(match.get("end", 0)),
                        confidence=1.0,  # Exact match
                        match_type=CitationType.EXACT,
                        metadata=citation.metadata
                    )
                    citations.append(exact_citation)
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse keyword_matches: {e}")
        
        return citations
    
    def delete_collection(self, name: str) -> None:
        """Delete a collection."""
        try:
            self.client.delete_collection(name)
            if name == self.collection_name:
                self._collection = None
            logger.info(f"Deleted collection: {name}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
    
    def clear(self) -> None:
        """Clear all data from the current collection."""
        try:
            # Get all IDs and delete them
            all_data = self.collection.get()
            if all_data['ids']:
                self.collection.delete(ids=all_data['ids'])
                logger.info(f"Cleared {len(all_data['ids'])} documents from collection")
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
    
    def get_collection_stats(self) -> dict:
        """Get collection statistics."""
        try:
            count = self.collection.count()
            
            # Get sample of metadata to show available fields
            sample_data = self.collection.peek(limit=1)
            metadata_fields = set()
            if sample_data['metadatas']:
                for meta in sample_data['metadatas']:
                    metadata_fields.update(meta.keys())
            
            stats = {
                "name": self.collection_name,
                "count": count,
                "persist_directory": str(self.persist_directory),
                "distance_metric": self.distance_metric,
                "metadata_fields": list(metadata_fields),
                "add_count": self._add_count,
                "search_count": self._search_count,
                "last_operation": self._last_operation_time
            }
            
            # Try to get embedding dimension
            if sample_data['embeddings'] and sample_data['embeddings'][0]:
                stats["embedding_dimension"] = len(sample_data['embeddings'][0])
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {
                "name": self.collection_name, 
                "count": 0,
                "error": str(e)
            }


class InMemoryVectorStore:
    """
    Simple in-memory vector store for testing or when persistence isn't needed.
    
    Implements the same interface as ChromaVectorStore but keeps everything in memory.
    """
    
    def __init__(self, distance_metric: str = "cosine"):
        """
        Initialize in-memory vector store.
        
        Args:
            distance_metric: Distance metric for similarity (cosine, l2, ip)
        """
        self.distance_metric = distance_metric
        self.ids: List[str] = []
        self.embeddings: List[np.ndarray] = []
        self.documents: List[str] = []
        self.metadata: List[dict] = []
        
        # Metrics
        self._add_count = 0
        self._search_count = 0
    
    def add(self, 
            ids: List[str], 
            embeddings: np.ndarray, 
            documents: List[str], 
            metadata: List[dict]) -> None:
        """Add documents to memory."""
        if len(ids) == 0:
            return
        
        # Convert embeddings to numpy if needed
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        
        # Add documents, updating existing ones
        for i, id_val in enumerate(ids):
            if id_val in self.ids:
                # Update existing
                idx = self.ids.index(id_val)
                self.embeddings[idx] = embeddings[i]
                self.documents[idx] = documents[i]
                self.metadata[idx] = metadata[i]
            else:
                # Add new
                self.ids.append(id_val)
                self.embeddings.append(embeddings[i])
                self.documents.append(documents[i])
                self.metadata.append(metadata[i])
        
        self._add_count += len(ids)
        logger.debug(f"Added {len(ids)} documents to in-memory store")
    
    def _compute_similarity(self, 
                          query_embedding: np.ndarray, 
                          doc_embedding: np.ndarray) -> float:
        """Compute similarity based on distance metric."""
        if self.distance_metric == "cosine":
            # Cosine similarity
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
            doc_norm = doc_embedding / (np.linalg.norm(doc_embedding) + 1e-9)
            return float(np.dot(query_norm, doc_norm))
        elif self.distance_metric == "l2":
            # Negative L2 distance (so higher is more similar)
            return -float(np.linalg.norm(query_embedding - doc_embedding))
        elif self.distance_metric == "ip":
            # Inner product
            return float(np.dot(query_embedding, doc_embedding))
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def search(self, 
               query_embedding: np.ndarray, 
               top_k: int = 10) -> List[SearchResult]:
        """Search using the specified distance metric."""
        if not self.embeddings:
            return []
        
        # Ensure query_embedding is numpy array
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)
        
        # Compute similarities
        similarities = []
        for i, emb in enumerate(self.embeddings):
            similarity = self._compute_similarity(query_embedding, emb)
            similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:top_k]
        
        # Convert to SearchResult
        results = []
        for idx, score in top_results:
            # Normalize score to [0, 1] range for consistency
            if self.distance_metric == "cosine":
                normalized_score = (score + 1) / 2  # From [-1, 1] to [0, 1]
            elif self.distance_metric == "l2":
                # Convert negative distance to similarity
                normalized_score = 1 / (1 + abs(score))
            else:  # ip
                normalized_score = max(0, min(1, score))  # Clamp to [0, 1]
            
            results.append(SearchResult(
                id=self.ids[idx],
                score=normalized_score,
                document=self.documents[idx],
                metadata=self.metadata[idx]
            ))
        
        self._search_count += 1
        return results
    
    def search_with_citations(self, 
                            query_embedding: np.ndarray,
                            query_text: str,
                            top_k: int = 10) -> List[SearchResultWithCitations]:
        """Enhanced search with citations."""
        # Get basic results
        basic_results = self.search(query_embedding, top_k)
        
        # Convert to results with citations
        results_with_citations = []
        for result in basic_results:
            # Create citation
            citation = Citation(
                document_id=result.metadata.get("doc_id", result.id),
                document_title=result.metadata.get("doc_title", "Unknown Document"),
                chunk_id=result.id,
                text=result.document[:300] + "..." if len(result.document) > 300 else result.document,
                start_char=int(result.metadata.get("chunk_start", 0)),
                end_char=int(result.metadata.get("chunk_end", len(result.document))),
                confidence=result.score,
                match_type=CitationType.SEMANTIC,
                metadata={
                    "author": result.metadata.get("author"),
                    "date": result.metadata.get("date"),
                    "url": result.metadata.get("url"),
                    "chunk_index": result.metadata.get("chunk_index", 0)
                }
            )
            
            result_with_citations = SearchResultWithCitations(
                id=result.id,
                score=result.score,
                document=result.document,
                metadata=result.metadata,
                citations=[citation]
            )
            results_with_citations.append(result_with_citations)
        
        return results_with_citations
    
    def delete_collection(self, name: str) -> None:
        """Clear all data (collection name ignored for in-memory)."""
        self.clear()
    
    def clear(self) -> None:
        """Clear all data."""
        self.ids.clear()
        self.embeddings.clear()
        self.documents.clear()
        self.metadata.clear()
        logger.info("Cleared in-memory vector store")
    
    def get_collection_stats(self) -> dict:
        """Get stats."""
        stats = {
            "type": "in_memory",
            "count": len(self.ids),
            "distance_metric": self.distance_metric,
            "add_count": self._add_count,
            "search_count": self._search_count
        }
        
        # Add embedding dimension if we have data
        if self.embeddings:
            stats["embedding_dimension"] = self.embeddings[0].shape[0]
        
        return stats


# Factory function for creating vector stores

def create_vector_store(store_type: str, 
                       persist_directory: Optional[Union[str, Path]] = None,
                       collection_name: str = "default",
                       distance_metric: str = "cosine",
                       **kwargs) -> VectorStore:
    """
    Factory function to create vector stores.
    
    Args:
        store_type: Type of vector store - "chroma", "memory"
        persist_directory: Directory for persistent stores (required for chroma)
        collection_name: Name of the collection
        distance_metric: Distance metric to use
        **kwargs: Additional arguments passed to the store constructor
        
    Returns:
        Vector store instance
        
    Raises:
        ValueError: If required parameters are missing or store type is unknown
    """
    store_type = store_type.lower()
    
    if store_type == "chroma":
        if persist_directory is None:
            raise ValueError("persist_directory required for ChromaDB")
        return ChromaVectorStore(
            persist_directory=persist_directory,
            collection_name=collection_name,
            distance_metric=distance_metric,
            **kwargs
        )
    
    elif store_type == "memory" or store_type == "inmemory":
        return InMemoryVectorStore(
            distance_metric=distance_metric,
            **kwargs
        )
    
    else:
        # Default to in-memory if unknown type
        logger.warning(f"Unknown vector store type: {store_type}. Using in-memory store.")
        return InMemoryVectorStore(distance_metric=distance_metric)