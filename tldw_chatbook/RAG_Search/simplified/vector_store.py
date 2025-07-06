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
import psutil

from .citations import Citation, CitationType, SearchResultWithCitations
from tldw_chatbook.Metrics.metrics_logger import log_counter, log_histogram, log_gauge, timeit

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
        
        # Log initialization
        logger.info(f"Initializing ChromaVectorStore: collection={collection_name}, "
                   f"metric={distance_metric}, persist_dir={persist_directory}")
        log_counter("vector_store_initialized", labels={
            "type": "chroma",
            "collection": collection_name,
            "metric": distance_metric
        })
        
        # Metrics
        self._add_count = 0
        self._search_count = 0
        self._last_operation_time = None
        self._embedding_dim = None
    
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
    
    @timeit("vector_store_add_documents")
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
        
        start_time = time.time()
        
        # Log operation metrics
        log_counter("vector_store_add_attempt", labels={"collection": self.collection_name})
        log_histogram("vector_store_batch_size", len(ids))
        
        # Validate inputs
        if len(ids) != len(documents) or len(ids) != len(metadata):
            raise ValueError("ids, documents, and metadata must have same length")
        
        if isinstance(embeddings, np.ndarray):
            if embeddings.shape[0] != len(ids):
                raise ValueError(f"embeddings shape {embeddings.shape} doesn't match {len(ids)} documents")
            
            # Store embedding dimension if not set
            if self._embedding_dim is None and embeddings.shape[1] > 0:
                self._embedding_dim = embeddings.shape[1]
                log_gauge("vector_store_embedding_dim", self._embedding_dim)
                logger.info(f"Detected embedding dimension: {self._embedding_dim}")
            
            embeddings = embeddings.tolist()
        
        # Log document statistics
        doc_lengths = [len(doc) for doc in documents]
        log_histogram("vector_store_document_length", sum(doc_lengths) / len(doc_lengths))
        
        # Ensure metadata is properly formatted
        processed_metadata = []
        metadata_sizes = []
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
            metadata_sizes.append(len(json.dumps(processed_meta)))
        
        # Log metadata statistics
        if metadata_sizes:
            log_histogram("vector_store_metadata_size", sum(metadata_sizes) / len(metadata_sizes))
        
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=processed_metadata
            )
            
            self._add_count += len(ids)
            self._last_operation_time = time.time()
            
            # Log success metrics
            elapsed = time.time() - start_time
            log_counter("vector_store_documents_added", value=len(ids))
            log_histogram("vector_store_add_time", elapsed)
            log_gauge("vector_store_total_documents", self._add_count)
            
            logger.info(f"Added {len(ids)} documents to collection in {elapsed:.3f}s")
            
        except Exception as e:
            log_counter("vector_store_add_error", labels={"error": type(e).__name__})
            logger.error(f"Failed to add to ChromaDB: {e}", exc_info=True)
            raise
    
    @timeit("vector_store_search")
    def search(self, 
               query_embedding: np.ndarray, 
               top_k: int = 10) -> List[SearchResult]:
        """Basic search without citations (backward compatibility)."""
        start_time = time.time()
        
        # Log search attempt
        log_counter("vector_store_search_attempt", labels={"collection": self.collection_name})
        log_histogram("vector_store_search_top_k", top_k)
        
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            search_results = []
            scores = []
            
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    score = 1 - results['distances'][0][i]  # Convert distance to similarity
                    scores.append(score)
                    search_results.append(SearchResult(
                        id=results['ids'][0][i],
                        score=score,
                        document=results['documents'][0][i],
                        metadata=results['metadatas'][0][i] or {}
                    ))
            
            # Log search metrics
            elapsed = time.time() - start_time
            log_histogram("vector_store_search_time", elapsed)
            log_histogram("vector_store_search_results_count", len(search_results))
            
            if scores:
                log_histogram("vector_store_search_avg_score", sum(scores) / len(scores))
                log_histogram("vector_store_search_max_score", max(scores))
                log_histogram("vector_store_search_min_score", min(scores))
            
            self._search_count += 1
            self._last_operation_time = time.time()
            log_gauge("vector_store_total_searches", self._search_count)
            
            logger.info(f"Search completed in {elapsed:.3f}s, found {len(search_results)} results")
            return search_results
            
        except Exception as e:
            log_counter("vector_store_search_error", labels={"error": type(e).__name__})
            logger.error(f"Search failed: {e}", exc_info=True)
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
    
    @timeit("vector_store_get_stats")
    def get_collection_stats(self) -> dict:
        """Get collection statistics."""
        try:
            count = self.collection.count()
            
            # Log collection statistics
            log_gauge("vector_store_collection_size", count, labels={"collection": self.collection_name})
            
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
                self._embedding_dim = stats["embedding_dimension"]
            
            # Log comprehensive stats
            logger.info(f"Collection stats: {count} documents, {len(metadata_fields)} metadata fields")
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {
                "name": self.collection_name, 
                "count": 0,
                "error": str(e)
            }
    
    def add_documents(self, 
                     collection_name: str,
                     documents: List[str],
                     embeddings: Union[List[List[float]], np.ndarray],
                     metadatas: List[dict],
                     ids: List[str]) -> bool:
        """
        Add documents to a collection (compatibility method).
        
        Args:
            collection_name: Name of the collection
            documents: List of document texts
            embeddings: Document embeddings
            metadatas: List of metadata dicts
            ids: Document IDs
            
        Returns:
            True if successful
        """
        # Switch to the specified collection if different
        if collection_name != self.collection_name:
            self.collection_name = collection_name
            self._collection = None  # Force reload
        
        # Convert embeddings to numpy array if needed
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        
        try:
            self.add(ids, embeddings, documents, metadatas)
            return True
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False
    
    def list_collections(self) -> List[str]:
        """List all collection names."""
        try:
            # Get all collections from ChromaDB
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []


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
        
        # Collection support (for compatibility)
        self._collections: Dict[str, Dict[str, Any]] = {}
        self._current_collection = "default"
        
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
                            top_k: int = 10,
                            score_threshold: float = 0.0) -> List[SearchResultWithCitations]:
        """Enhanced search with citations."""
        # Get basic results (get extra to account for filtering)
        basic_results = self.search(query_embedding, top_k * 2)
        
        # Filter by score threshold
        filtered_results = [r for r in basic_results if r.score >= score_threshold]
        
        # Convert to results with citations
        results_with_citations = []
        for result in filtered_results[:top_k]:
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
    
    def add_documents(self, 
                     collection_name: str,
                     documents: List[str],
                     embeddings: Union[List[List[float]], np.ndarray],
                     metadatas: List[dict],
                     ids: List[str]) -> bool:
        """
        Add documents to a collection (compatibility method).
        
        Args:
            collection_name: Name of the collection
            documents: List of document texts
            embeddings: Document embeddings
            metadatas: List of metadata dicts
            ids: Document IDs
            
        Returns:
            True if successful
        """
        # Store current collection
        self._current_collection = collection_name
        
        # Convert embeddings to numpy array if needed
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        
        # Initialize collection if it doesn't exist
        if collection_name not in self._collections:
            self._collections[collection_name] = {
                "ids": [],
                "embeddings": [],
                "documents": [],
                "metadata": []
            }
        
        # Add to the main store (for backward compatibility)
        self.add(ids, embeddings, documents, metadatas)
        
        # Also track in collections
        collection = self._collections[collection_name]
        collection["ids"].extend(ids)
        collection["embeddings"].extend(embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings)
        collection["documents"].extend(documents)
        collection["metadata"].extend(metadatas)
        
        return True
    
    def list_collections(self) -> List[str]:
        """List all collection names."""
        # Always include default collection if we have data
        collections = set(self._collections.keys())
        if self.ids:
            collections.add("default")
        return list(collections)


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