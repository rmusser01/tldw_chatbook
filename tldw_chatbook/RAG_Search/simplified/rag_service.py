"""
Main RAG service coordinator with citations support.

This is the main entry point for the simplified RAG implementation, coordinating
embeddings, vector stores, chunking, and search operations.
"""

import asyncio
from typing import List, Optional, Dict, Any, Literal, Union, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass
import time
import numpy as np

from .embeddings_wrapper import EmbeddingsServiceWrapper
from .vector_store import (
    create_vector_store, VectorStore, SearchResult, 
    SearchResultWithCitations
)
from .citations import Citation, CitationType, merge_citations
from .config import RAGConfig
from ..chunking_service import ChunkingService
from .simple_cache import SimpleRAGCache, get_rag_cache

logger = logging.getLogger(__name__)


@dataclass
class IndexingResult:
    """Result of indexing operation."""
    doc_id: str
    chunks_created: int
    time_taken: float
    success: bool
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "doc_id": self.doc_id,
            "chunks_created": self.chunks_created,
            "time_taken": self.time_taken,
            "success": self.success,
            "error": self.error
        }


class RAGService:
    """
    Main RAG service with citations support.
    
    This service coordinates:
    - Document chunking and indexing
    - Embedding creation using existing Embeddings_Lib
    - Vector storage with ChromaDB or in-memory
    - Search with semantic, keyword, and hybrid modes
    - Citation generation for source attribution
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Initialize RAG service with configuration.
        
        Args:
            config: RAG configuration (uses defaults if None)
        """
        self.config = config or RAGConfig()
        
        # Initialize embeddings using wrapper around existing library
        logger.info(f"Initializing embeddings service with model: {self.config.embedding_model}")
        self.embeddings = EmbeddingsServiceWrapper(
            model_name=self.config.embedding_model,
            cache_size=self.config.embedding_cache_size,
            device=self.config.device
        )
        
        # Initialize vector store
        logger.info(f"Initializing {self.config.vector_store_type} vector store")
        self.vector_store = create_vector_store(
            store_type=self.config.vector_store_type,
            persist_directory=self.config.persist_directory,
            collection_name=self.config.collection_name,
            distance_metric=self.config.distance_metric
        )
        
        # Initialize chunking service
        self.chunking = ChunkingService()
        
        # Initialize cache
        cache_config = config.search.__dict__ if hasattr(config.search, '__dict__') else {}
        self.cache = get_rag_cache(
            max_size=cache_config.get('cache_size', 100),
            ttl_seconds=cache_config.get('cache_ttl', 3600),
            enabled=cache_config.get('enable_cache', True)
        )
        
        # Metrics
        self._docs_indexed = 0
        self._searches_performed = 0
        self._last_index_time = None
        self._total_chunks_created = 0
    
    # === Indexing Methods ===
    
    async def index_document(self, 
                           doc_id: str, 
                           content: str,
                           title: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None,
                           chunk_size: Optional[int] = None,
                           chunk_overlap: Optional[int] = None,
                           chunking_method: Optional[str] = None) -> IndexingResult:
        """
        Index a document with metadata for citations.
        
        Args:
            doc_id: Unique document identifier
            content: Document content to index
            title: Human-readable document title
            metadata: Optional metadata (author, date, url, etc.)
            chunk_size: Override default chunk size
            chunk_overlap: Override default chunk overlap
            chunking_method: Override default chunking method
            
        Returns:
            IndexingResult with status and statistics
        """
        start_time = time.time()
        metadata = metadata or {}
        title = title or doc_id
        
        try:
            # Chunk the document
            chunks = await self._chunk_document(
                content, 
                chunk_size or self.config.chunk_size,
                chunk_overlap or self.config.chunk_overlap,
                chunking_method or self.config.chunking_method
            )
            
            if not chunks:
                logger.warning(f"No chunks created for document {doc_id}")
                return IndexingResult(
                    doc_id=doc_id,
                    chunks_created=0,
                    time_taken=time.time() - start_time,
                    success=True
                )
            
            # Extract chunk texts
            chunk_texts = [chunk["text"] for chunk in chunks]
            
            # Create embeddings
            logger.debug(f"Creating embeddings for {len(chunk_texts)} chunks")
            embeddings = await self.embeddings.create_embeddings_async(chunk_texts)
            
            # Prepare for storage with citation metadata
            chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
            chunk_metadata = []
            
            for i, chunk in enumerate(chunks):
                # Combine chunk metadata with document metadata
                meta = {
                    **metadata,
                    "doc_id": doc_id,
                    "doc_title": title,
                    "chunk_index": i,
                    "chunk_start": chunk.get("start_char", 0),
                    "chunk_end": chunk.get("end_char", len(chunk["text"])),
                    "chunk_size": len(chunk["text"]),
                    "word_count": chunk.get("word_count", 0),
                    # Store part of text for keyword matching
                    "text_preview": chunk["text"][:200]
                }
                chunk_metadata.append(meta)
            
            # Store in vector database
            await self._store_chunks(chunk_ids, embeddings, chunk_texts, chunk_metadata)
            
            # Update metrics
            self._docs_indexed += 1
            self._total_chunks_created += len(chunks)
            self._last_index_time = time.time()
            
            elapsed = time.time() - start_time
            logger.info(f"Indexed document {doc_id} with {len(chunks)} chunks in {elapsed:.2f}s")
            
            return IndexingResult(
                doc_id=doc_id,
                chunks_created=len(chunks),
                time_taken=elapsed,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Failed to index document {doc_id}: {e}", exc_info=True)
            return IndexingResult(
                doc_id=doc_id,
                chunks_created=0,
                time_taken=time.time() - start_time,
                success=False,
                error=str(e)
            )
    
    def index_document_sync(self, doc_id: str, content: str, **kwargs) -> IndexingResult:
        """Synchronous version of index_document."""
        return asyncio.run(self.index_document(doc_id, content, **kwargs))
    
    async def index_batch(self, 
                         documents: List[Dict[str, Any]],
                         show_progress: bool = True,
                         continue_on_error: bool = True) -> List[IndexingResult]:
        """
        Index multiple documents in batch.
        
        Args:
            documents: List of dicts with 'id', 'content', and optional 'title', 'metadata'
            show_progress: Whether to log progress
            continue_on_error: Whether to continue if a document fails
            
        Returns:
            List of IndexingResult for each document
        """
        results = []
        total = len(documents)
        
        for i, doc in enumerate(documents):
            if show_progress and i % 10 == 0 and i > 0:
                logger.info(f"Indexing progress: {i}/{total} documents")
            
            try:
                result = await self.index_document(
                    doc_id=doc['id'],
                    content=doc['content'],
                    title=doc.get('title'),
                    metadata=doc.get('metadata')
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to index document {doc.get('id', 'unknown')}: {e}")
                if not continue_on_error:
                    raise
                results.append(IndexingResult(
                    doc_id=doc.get('id', 'unknown'),
                    chunks_created=0,
                    time_taken=0,
                    success=False,
                    error=str(e)
                ))
        
        if show_progress:
            successful = sum(1 for r in results if r.success)
            logger.info(f"Indexed {successful}/{total} documents successfully")
        
        return results
    
    # === Search Methods ===
    
    async def search(self,
                    query: str,
                    top_k: Optional[int] = None,
                    search_type: Literal["semantic", "hybrid", "keyword"] = "semantic",
                    filter_metadata: Optional[Dict[str, Any]] = None,
                    include_citations: Optional[bool] = None,
                    score_threshold: Optional[float] = None) -> Union[List[SearchResult], List[SearchResultWithCitations]]:
        """
        Search with optional citations.
        
        Args:
            query: Search query text
            top_k: Number of results to return (default from config)
            search_type: Type of search to perform
            filter_metadata: Metadata filters to apply
            include_citations: Whether to include citations (default from config)
            score_threshold: Minimum score threshold (default from config)
            
        Returns:
            List of search results (with or without citations)
        """
        # Use defaults from config if not specified
        top_k = top_k or self.config.default_top_k
        include_citations = include_citations if include_citations is not None else self.config.include_citations
        score_threshold = score_threshold if score_threshold is not None else self.config.score_threshold
        
        # Check cache first
        cached_result = self.cache.get(query, search_type, top_k, filter_metadata)
        if cached_result is not None:
            results, context = cached_result
            logger.debug(f"Returning cached results for query: '{query}'")
            return results
        
        self._searches_performed += 1
        start_time = time.time()
        
        try:
            if search_type == "semantic":
                results = await self._semantic_search(
                    query, top_k, filter_metadata, include_citations, score_threshold
                )
            elif search_type == "hybrid":
                results = await self._hybrid_search(
                    query, top_k, filter_metadata, include_citations, score_threshold
                )
            elif search_type == "keyword":
                results = await self._keyword_search(
                    query, top_k, filter_metadata, include_citations
                )
            else:
                raise ValueError(f"Unknown search type: {search_type}")
            
            elapsed = time.time() - start_time
            logger.info(f"Search completed in {elapsed:.2f}s, found {len(results)} results")
            
            # Cache the results
            # For caching, we need to extract a simple context string
            context = self._extract_context_from_results(results)
            self.cache.put(query, search_type, top_k, results, context, filter_metadata)
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            raise
    
    def search_sync(self, query: str, **kwargs) -> Union[List[SearchResult], List[SearchResultWithCitations]]:
        """Synchronous version of search."""
        return asyncio.run(self.search(query, **kwargs))
    
    async def _semantic_search(self, 
                              query: str, 
                              top_k: int,
                              filter_metadata: Optional[Dict[str, Any]] = None,
                              include_citations: bool = True,
                              score_threshold: float = 0.0) -> Union[List[SearchResult], List[SearchResultWithCitations]]:
        """Perform semantic similarity search."""
        # Create query embedding
        logger.debug("Creating query embedding")
        query_embedding = await self.embeddings.create_embeddings_async([query])
        query_embedding = query_embedding[0]
        
        # Search vector store
        if include_citations:
            results = self.vector_store.search_with_citations(
                query_embedding, query, top_k * 2, score_threshold
            )
        else:
            results = self.vector_store.search(query_embedding, top_k * 2)
            # Apply score threshold for basic results
            results = [r for r in results if r.score >= score_threshold]
        
        # Apply metadata filters if provided
        if filter_metadata:
            results = [
                r for r in results
                if all(r.metadata.get(k) == v for k, v in filter_metadata.items())
            ]
        
        return results[:top_k]
    
    async def _keyword_search(self,
                             query: str,
                             top_k: int,
                             filter_metadata: Optional[Dict[str, Any]] = None,
                             include_citations: bool = True) -> Union[List[SearchResult], List[SearchResultWithCitations]]:
        """
        Perform keyword search using FTS5.
        
        TODO: Implement actual FTS5 integration.
        For now, returns empty results.
        """
        logger.warning("Keyword search not yet implemented in simplified version")
        return []
    
    async def _hybrid_search(self,
                            query: str,
                            top_k: int,
                            filter_metadata: Optional[Dict[str, Any]] = None,
                            include_citations: bool = True,
                            score_threshold: float = 0.0) -> Union[List[SearchResult], List[SearchResultWithCitations]]:
        """
        Perform hybrid search combining semantic and keyword.
        
        Merges results from both search types, removing duplicates and
        combining citations when the same chunk appears in both.
        """
        # Get results from both search types
        semantic_task = self._semantic_search(
            query, top_k * 2, filter_metadata, include_citations, score_threshold
        )
        keyword_task = self._keyword_search(
            query, top_k * 2, filter_metadata, include_citations
        )
        
        # Run both searches in parallel
        semantic_results, keyword_results = await asyncio.gather(
            semantic_task, keyword_task
        )
        
        if include_citations:
            # Merge results with citations
            return self._merge_results_with_citations(
                semantic_results, keyword_results, top_k
            )
        else:
            # Simple merging for basic results
            return self._merge_basic_results(
                semantic_results, keyword_results, top_k
            )
    
    def _merge_results_with_citations(self,
                                    semantic_results: List[SearchResultWithCitations],
                                    keyword_results: List[SearchResultWithCitations],
                                    top_k: int) -> List[SearchResultWithCitations]:
        """Merge results while combining citations from both sources."""
        merged = {}
        
        # Process semantic results
        for result in semantic_results:
            merged[result.id] = result
        
        # Merge keyword results
        for result in keyword_results:
            if result.id in merged:
                # Combine citations from both
                existing = merged[result.id]
                # Merge citations, keeping highest confidence for duplicates
                all_citations = existing.citations + result.citations
                existing.citations = merge_citations([existing.citations, result.citations])
                # Update score (weighted average based on number of citations)
                total_citations = len(existing.citations) + len(result.citations)
                if total_citations > 0:
                    existing.score = (
                        existing.score * len(existing.citations) + 
                        result.score * len(result.citations)
                    ) / total_citations
            else:
                merged[result.id] = result
        
        # Sort by score and return top-k
        sorted_results = sorted(merged.values(), key=lambda x: x.score, reverse=True)
        return sorted_results[:top_k]
    
    def _merge_basic_results(self,
                           semantic_results: List[SearchResult],
                           keyword_results: List[SearchResult],
                           top_k: int) -> List[SearchResult]:
        """Simple merging for basic results."""
        # Use dict to track seen IDs and keep highest score
        merged = {}
        
        for result in semantic_results:
            merged[result.id] = result
        
        for result in keyword_results:
            if result.id in merged:
                # Keep the one with higher score
                if result.score > merged[result.id].score:
                    merged[result.id] = result
            else:
                merged[result.id] = result
        
        # Sort by score and return top-k
        sorted_results = sorted(merged.values(), key=lambda x: x.score, reverse=True)
        return sorted_results[:top_k]
    
    # === Helper Methods ===
    
    async def _chunk_document(self, 
                            content: str,
                            chunk_size: int,
                            chunk_overlap: int,
                            method: str) -> List[Dict[str, Any]]:
        """Chunk document asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.chunking.chunk_text,
            content,
            chunk_size,
            chunk_overlap,
            method
        )
    
    async def _store_chunks(self,
                          ids: List[str],
                          embeddings: np.ndarray,
                          documents: List[str],
                          metadata: List[dict]) -> None:
        """Store chunks in vector database asynchronously."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self.vector_store.add,
            ids, embeddings, documents, metadata
        )
    
    # === Management Methods ===
    
    def _extract_context_from_results(self, results: List[Union[SearchResult, SearchResultWithCitations]], 
                                     max_length: int = 10000) -> str:
        """Extract a context string from search results for caching."""
        context_parts = []
        total_chars = 0
        
        for result in results:
            # Format result
            title = result.metadata.get('title', 'Untitled')
            source = result.metadata.get('source', 'unknown')
            
            result_text = f"[{source.upper()} - {title}]\n"
            remaining_chars = max_length - total_chars - len(result_text)
            
            if remaining_chars <= 0:
                break
            
            content_preview = result.document[:remaining_chars]
            result_text += content_preview
            
            if len(result.document) > remaining_chars:
                result_text += "...\n"
            else:
                result_text += "\n"
            
            context_parts.append(result_text)
            total_chars += len(result_text)
            
            if total_chars >= max_length:
                break
        
        return "\n---\n".join(context_parts)
    
    def clear_cache(self):
        """Clear all caches."""
        self.embeddings.clear_cache()
        self.cache.clear()
        logger.info("Cleared embeddings and search result caches")
    
    def clear_index(self):
        """Clear the vector store index."""
        self.vector_store.clear()
        self._docs_indexed = 0
        self._total_chunks_created = 0
        logger.info("Cleared vector store index")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive service metrics."""
        metrics = {
            "embeddings_metrics": self.embeddings.get_metrics(),
            "vector_store_stats": self.vector_store.get_collection_stats(),
            "cache_metrics": self.cache.get_metrics(),
            "service_metrics": {
                "documents_indexed": self._docs_indexed,
                "total_chunks_created": self._total_chunks_created,
                "searches_performed": self._searches_performed,
                "last_index_time": self._last_index_time
            },
            "config": {
                "embedding_model": self.config.embedding_model,
                "vector_store_type": self.config.vector_store_type,
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "default_top_k": self.config.default_top_k
            }
        }
        return metrics
    
    def get_document_count(self) -> int:
        """Get the number of indexed documents."""
        return self._docs_indexed
    
    def get_chunk_count(self) -> int:
        """Get the total number of chunks in the index."""
        stats = self.vector_store.get_collection_stats()
        return stats.get("count", 0)
    
    def close(self):
        """Clean up resources."""
        try:
            self.embeddings.close()
            logger.info("RAG service closed successfully")
        except Exception as e:
            logger.error(f"Error closing RAG service: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Convenience functions

async def create_and_index(
    documents: List[Dict[str, Any]],
    config: Optional[RAGConfig] = None,
    show_progress: bool = True
) -> Tuple[RAGService, List[IndexingResult]]:
    """
    Create a RAG service and index documents in one go.
    
    Args:
        documents: List of documents to index
        config: Optional RAG configuration
        show_progress: Whether to show progress
        
    Returns:
        Tuple of (RAGService instance, List of indexing results)
    """
    service = RAGService(config)
    results = await service.index_batch(documents, show_progress)
    return service, results


def create_rag_service(
    embedding_model: Optional[str] = None,
    vector_store: str = "chroma",
    persist_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> RAGService:
    """
    Create a RAG service with common configurations.
    
    Args:
        embedding_model: Embedding model to use
        vector_store: Vector store type ("chroma" or "memory")
        persist_dir: Directory for persistence (if using chroma)
        **kwargs: Additional config parameters
        
    Returns:
        Configured RAGService instance
    """
    config = RAGConfig(
        embedding_model=embedding_model or "sentence-transformers/all-MiniLM-L6-v2",
        vector_store_type=vector_store,
        persist_directory=persist_dir,
        **kwargs
    )
    return RAGService(config)