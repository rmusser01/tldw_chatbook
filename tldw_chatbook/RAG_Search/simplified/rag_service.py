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
import uuid
import psutil

from .embeddings_wrapper import EmbeddingsServiceWrapper
from .vector_store import (
    create_vector_store, VectorStore, SearchResult, 
    SearchResultWithCitations
)
from .citations import Citation, CitationType, merge_citations
from .config import RAGConfig
from ..chunking_service import ChunkingService
from .simple_cache import SimpleRAGCache, get_rag_cache
from .db_connection_pool import get_connection_pool
from .indexing_helpers import chunk_documents_batch, generate_embeddings_batch, store_documents_batch
from .health_check import init_health_checker, get_health_status
from .data_models import IndexingResult
from tldw_chatbook.Metrics.metrics_logger import log_counter, log_histogram, log_gauge, timeit
from tldw_chatbook.Utils.path_validation import validate_path

logger = logging.getLogger(__name__)

# Constants
DEFAULT_EMBEDDING_DIM = 768  # Default dimension if detection fails
KEYWORD_SEARCH_SCORE = 0.8  # Fixed score for keyword search results
MAX_CITATION_MATCHES = 3  # Maximum keyword matches to show as citations
CITATION_CONTEXT_CHARS = 50  # Characters of context around keyword matches
KEYWORD_BATCH_SIZE = 10  # Batch size for processing keyword results
FTS5_CONNECTION_POOL_SIZE = 3  # Connection pool size for FTS5 searches
CACHE_TIMEOUT_SECONDS = 3600.0  # Cache timeout: 1 hour for better performance


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
        
        # Log comprehensive configuration
        logger.info("RAG Service Configuration", extra={
            "embedding_model": self.config.embedding_model,
            "vector_store_type": self.config.vector_store_type,
            "collection_name": self.config.collection_name,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "device": self.config.device
        })
        
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
        logger.info("Initialized chunking service")
        
        # Initialize cache
        cache_config = config.search.__dict__ if hasattr(config.search, '__dict__') else {}
        cache_size = cache_config.get('cache_size', 100)
        cache_ttl = cache_config.get('cache_ttl', 3600)
        cache_enabled = cache_config.get('enable_cache', True)
        
        self.cache = get_rag_cache(
            max_size=cache_size,
            ttl_seconds=cache_ttl,
            enabled=cache_enabled
        )
        logger.info(f"Initialized cache: size={cache_size}, ttl={cache_ttl}s, enabled={cache_enabled}")
        
        # Log initialization metrics
        log_counter("rag_service_initialized", labels={
            "model": self.config.embedding_model,
            "vector_store": self.config.vector_store_type,
            "device": self.config.device
        })
        
        # Metrics
        self._docs_indexed = 0
        self._searches_performed = 0
        self._last_index_time = None
        self._total_chunks_created = 0
        self._search_type_counts = {"semantic": 0, "keyword": 0, "hybrid": 0}
        
        # Get and store embedding dimension
        self._embedding_dim = self._get_embedding_dimension()
        logger.info(f"Detected embedding dimension: {self._embedding_dim}")
        
        # Initialize health checker
        init_health_checker(self)
    
    # === Indexing Methods ===
    
    @timeit("rag_indexing_document")
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
        
        # Create correlation ID for tracking
        correlation_id = str(uuid.uuid4())
        
        # Input validation
        if not doc_id or not isinstance(doc_id, str):
            raise ValueError("doc_id must be a non-empty string")
        
        if not content or not isinstance(content, str):
            raise ValueError("content must be a non-empty string")
        
        # Document size limit: 10MB (configurable)
        max_doc_size = getattr(self.config, 'max_document_size', 10 * 1024 * 1024)  # 10MB default
        if len(content) > max_doc_size:
            raise ValueError(f"Document too large: {len(content)} bytes exceeds limit of {max_doc_size} bytes")
        
        # Validate chunk parameters if provided
        if chunk_size is not None and (not isinstance(chunk_size, int) or chunk_size < 1):
            raise ValueError("chunk_size must be a positive integer")
        
        if chunk_overlap is not None and (not isinstance(chunk_overlap, int) or chunk_overlap < 0):
            raise ValueError("chunk_overlap must be a non-negative integer")
        
        if chunk_overlap is not None and chunk_size is not None and chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        # Log document metrics
        log_counter("rag_document_index_attempt")
        log_histogram("rag_document_size_chars", len(content))
        
        try:
            # Chunk the document with timing
            chunk_start = time.time()
            chunks = await self._chunk_document(
                content, 
                chunk_size or self.config.chunk_size,
                chunk_overlap or self.config.chunk_overlap,
                chunking_method or self.config.chunking_method
            )
            chunk_time = time.time() - chunk_start
            log_histogram("rag_chunking_time", chunk_time)
            
            if not chunks:
                logger.warning(f"No chunks created for document {doc_id}")
                log_counter("rag_document_empty_chunks")
                return IndexingResult(
                    doc_id=doc_id,
                    chunks_created=0,
                    time_taken=time.time() - start_time,
                    success=True
                )
            
            # Log chunk statistics
            chunk_sizes = [len(chunk["text"]) for chunk in chunks]
            log_histogram("rag_chunks_per_document", len(chunks))
            log_histogram("rag_chunk_size_chars", sum(chunk_sizes) / len(chunk_sizes))
            log_counter("rag_chunks_created", value=len(chunks))
            
            # Extract chunk texts
            chunk_texts = [chunk["text"] for chunk in chunks]
            
            # Create embeddings with timing
            embed_start = time.time()
            logger.info(f"Creating embeddings for {len(chunk_texts)} chunks")
            embeddings = await self.embeddings.create_embeddings_async(chunk_texts)
            embed_time = time.time() - embed_start
            log_histogram("rag_embedding_time", embed_time)
            log_histogram("rag_embeddings_per_document", len(embeddings))
            
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
            
            # Store in vector database with timing
            store_start = time.time()
            await self._store_chunks(chunk_ids, embeddings, chunk_texts, chunk_metadata)
            store_time = time.time() - store_start
            log_histogram("rag_vector_store_time", store_time)
            
            # Update metrics
            self._docs_indexed += 1
            self._total_chunks_created += len(chunks)
            self._last_index_time = time.time()
            
            elapsed = time.time() - start_time
            
            # Log success metrics
            log_counter("rag_document_index_success")
            log_histogram("rag_document_index_total_time", elapsed)
            log_gauge("rag_total_documents_indexed", self._docs_indexed)
            log_gauge("rag_total_chunks_in_index", self._total_chunks_created)
            
            logger.info(f"Indexed document {doc_id} with {len(chunks)} chunks in {elapsed:.2f}s " +
                       f"(chunk: {chunk_time:.2f}s, embed: {embed_time:.2f}s, store: {store_time:.2f}s)")
            
            return IndexingResult(
                doc_id=doc_id,
                chunks_created=len(chunks),
                time_taken=elapsed,
                success=True
            )
            
        except Exception as e:
            log_counter("rag_document_index_error", labels={"error": type(e).__name__})
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
    
    async def index_batch_optimized(self, 
                                   documents: List[Dict[str, Any]],
                                   show_progress: bool = True,
                                   batch_size: int = 32) -> List[IndexingResult]:
        """
        Optimized batch indexing with batched embeddings for better performance.
        
        This method processes multiple documents more efficiently by:
        1. Chunking all documents first
        2. Creating embeddings in batches
        3. Storing all results together
        
        Args:
            documents: List of documents to index
            show_progress: Whether to show progress
            batch_size: Batch size for embedding generation
            
        Returns:
            List of IndexingResult for each document
        """
        total = len(documents)
        if not documents:
            return []
        
        logger.info(f"Starting optimized batch indexing for {total} documents")
        batch_start_time = time.time()
        
        # Phase 1: Chunk all documents
        chunk_start = time.time()
        all_chunks, doc_chunk_info, failed_results = await chunk_documents_batch(
            self, documents, show_progress
        )
        chunk_time = time.time() - chunk_start
        logger.info(f"Chunking completed in {chunk_time:.2f}s, total chunks: {len(all_chunks)}")
        
        if not all_chunks:
            logger.warning("No chunks created from any documents")
            return failed_results
        
        # Phase 2: Generate embeddings in batches
        embed_start = time.time()
        chunk_texts = [chunk['text'] for chunk in all_chunks]
        all_embeddings = await generate_embeddings_batch(
            self, chunk_texts, batch_size, show_progress
        )
        embed_time = time.time() - embed_start
        logger.info(f"Embedding generation completed in {embed_time:.2f}s")
        
        # Phase 3: Store documents with their embeddings
        store_start = time.time()
        storage_results = await store_documents_batch(
            self, documents, doc_chunk_info, all_embeddings, batch_start_time
        )
        store_time = time.time() - store_start
        
        # Combine results
        results = failed_results + storage_results
        total_time = time.time() - batch_start_time
        
        # Summary
        successful = sum(1 for r in results if r and r.success)
        logger.info(
            f"Batch indexing completed: {successful}/{total} documents, "
            f"total time: {total_time:.2f}s "
            f"(chunk: {chunk_time:.2f}s, embed: {embed_time:.2f}s, store: {store_time:.2f}s)"
        )
        
        # Update metrics
        log_counter("rag_batch_index_completed", value=successful)
        log_histogram("rag_batch_index_total_time", total_time)
        log_histogram("rag_batch_chunk_time", chunk_time)
        log_histogram("rag_batch_embed_time", embed_time)
        log_histogram("rag_batch_store_time", store_time)
        
        return results
    
    # === Search Methods ===
    
    @timeit("rag_search_operation")
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
        
        # Create correlation ID for tracking
        correlation_id = str(uuid.uuid4())
        
        # Log search metrics
        log_counter("rag_search_attempt", labels={"type": search_type})
        log_histogram("rag_search_query_length", len(query))
        self._search_type_counts[search_type] += 1
        
        # Check cache first
        cached_result = await self.cache.get_async(query, search_type, top_k, filter_metadata)
        if cached_result is not None:
            results, context = cached_result
            log_counter("rag_search_cache_hit", labels={"type": search_type})
            logger.info(f"[{correlation_id}] Cache hit for query: '{query[:50]}...'")
            return results
        
        log_counter("rag_search_cache_miss", labels={"type": search_type})
        
        self._searches_performed += 1
        start_time = time.time()
        results_before_filter = 0
        
        try:
            logger.info(f"[{correlation_id}] Performing {search_type} search with top_k={top_k}, threshold={score_threshold}")
            
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
            
            # Log result statistics
            if results:
                scores = [r.score for r in results]
                log_histogram("rag_search_result_score", sum(scores) / len(scores))
                log_histogram("rag_search_min_score", min(scores))
                log_histogram("rag_search_max_score", max(scores))
                
                # Log score distribution
                for i, score in enumerate(scores[:5]):  # Top 5 results
                    log_histogram("rag_search_score_distribution", score, 
                                labels={"rank": str(i+1), "type": search_type})
            
            elapsed = time.time() - start_time
            
            # Log search success metrics
            log_counter("rag_search_success", labels={"type": search_type})
            log_histogram("rag_search_time", elapsed, labels={"type": search_type})
            log_histogram("rag_search_results_count", len(results))
            log_gauge("rag_total_searches_performed", self._searches_performed)
            
            # Log search type distribution
            total_searches = sum(self._search_type_counts.values())
            for stype, count in self._search_type_counts.items():
                log_gauge(f"rag_search_type_{stype}_ratio", 
                         count / total_searches if total_searches > 0 else 0)
            
            logger.info(f"[{correlation_id}] Search completed in {elapsed:.2f}s, found {len(results)} results")
            
            # Cache the results
            # For caching, we need to extract a simple context string
            context = self._extract_context_from_results(results)
            await self.cache.put_async(query, search_type, top_k, results, context, filter_metadata)
            
            return results
            
        except Exception as e:
            log_counter("rag_search_error", labels={"type": search_type, "error": type(e).__name__})
            logger.error(f"[{correlation_id}] Search failed: {e}", exc_info=True)
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
        Perform keyword search using FTS5 from the media database.
        
        This implementation leverages the existing FTS5 index in the MediaDatabase
        for efficient keyword-based search with proper connection pooling.
        """
        try:
            # Get database path from vector store persist directory
            db_path = None
            base_dir = Path.home() / ".local" / "share" / "tldw_cli"
            
            if self.config.persist_directory:
                # Try common locations for the media database
                possible_paths = [
                    self.config.persist_directory.parent / "media_db.db",
                    self.config.persist_directory.parent / "chacha_notes.db",
                    base_dir / "chacha_notes.db"
                ]
                
                for path in possible_paths:
                    # Validate path to prevent traversal attacks
                    try:
                        validated_path = validate_path(str(path), str(base_dir))
                        validated_path_obj = Path(validated_path)
                        
                        # Check if path exists and is not a symlink (security check)
                        if validated_path_obj.exists() and not validated_path_obj.is_symlink():
                            # Additional check: ensure it's a regular file
                            if validated_path_obj.is_file():
                                db_path = validated_path
                                logger.debug(f"Found media database at: {db_path}")
                                break
                            else:
                                logger.warning(f"Path {validated_path} is not a regular file")
                        elif validated_path_obj.is_symlink():
                            logger.warning(f"Skipping symlink at {validated_path} for security reasons")
                    except ValueError as e:
                        logger.warning(f"Invalid path {path}: {e}")
                        continue
            
            if not db_path:
                logger.warning("Could not find media database for keyword search")
                return []
            
            # Get connection pool for this database
            pool_size = getattr(self.config.search, 'fts5_connection_pool_size', FTS5_CONNECTION_POOL_SIZE)
            pool = get_connection_pool(db_path, pool_size=pool_size)
            
            # Perform FTS5 search directly using connection pool
            loop = asyncio.get_event_loop()
            search_results = await loop.run_in_executor(
                None,
                self._perform_fts5_search,
                pool,
                query,
                top_k * 2  # Get extra for filtering
            )
            
            # Process results in batches for better performance
            if include_citations:
                results = await self._process_keyword_results_with_citations(
                    search_results, query, filter_metadata, top_k
                )
            else:
                results = self._process_keyword_results_basic(
                    search_results, filter_metadata, top_k
                )
            
            logger.info(f"Keyword search found {len(results)} results for query: '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}", exc_info=True)
            # Return empty list on error to maintain compatibility
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
    
    def _get_embedding_dimension(self) -> int:
        """Get the embedding dimension from the model."""
        try:
            dim = self.embeddings.get_embedding_dimension()
            if dim is None:
                # Default if we can't determine
                logger.warning(f"Could not determine embedding dimension, defaulting to {DEFAULT_EMBEDDING_DIM}")
                return DEFAULT_EMBEDDING_DIM
            return dim
        except Exception as e:
            logger.warning(f"Error getting embedding dimension: {e}, defaulting to {DEFAULT_EMBEDDING_DIM}")
            return DEFAULT_EMBEDDING_DIM
    
    @timeit("rag_chunking_operation")
    async def _chunk_document(self, 
                            content: str,
                            chunk_size: int,
                            chunk_overlap: int,
                            method: str) -> List[Dict[str, Any]]:
        """Chunk document asynchronously."""
        logger.info(f"Chunking document with method={method}, size={chunk_size}, overlap={chunk_overlap}")
        log_histogram("rag_chunk_size_config", chunk_size)
        log_histogram("rag_chunk_overlap_config", chunk_overlap)
        log_counter("rag_chunking_method", labels={"method": method})
        
        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(
            None,
            self.chunking.chunk_text,
            content,
            chunk_size,
            chunk_overlap,
            method
        )
        
        # Log chunk statistics
        if chunks:
            chunk_lengths = [len(chunk.get("text", "")) for chunk in chunks]
            avg_chunk_length = sum(chunk_lengths) / len(chunk_lengths)
            log_histogram("rag_avg_chunk_length", avg_chunk_length)
            log_histogram("rag_min_chunk_length", min(chunk_lengths))
            log_histogram("rag_max_chunk_length", max(chunk_lengths))
            logger.debug(f"Created {len(chunks)} chunks, avg length: {avg_chunk_length:.0f} chars")
        
        return chunks
    
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
    
    def _process_keyword_results_basic(self, search_results: List[Dict], 
                                      filter_metadata: Optional[Dict[str, Any]], 
                                      top_k: int) -> List[SearchResult]:
        """Process keyword search results without citations."""
        results = []
        
        for item in search_results:
            # Apply metadata filters if provided
            if filter_metadata:
                item_meta = {
                    'media_type': item.get('type'),
                    'source': 'media',
                    'author': item.get('author')
                }
                if not all(item_meta.get(k) == v for k, v in filter_metadata.items() if k in item_meta):
                    continue
            
            # Create base SearchResult
            content = item.get('content', '')[:1000]  # Limit content size
            
            base_result = SearchResult(
                id=f"media_{item['id']}",
                score=KEYWORD_SEARCH_SCORE,  # FTS5 doesn't provide normalized scores
                document=content,
                metadata={
                    'doc_id': str(item['id']),
                    'doc_title': item.get('title', 'Untitled'),
                    'media_type': item.get('type'),
                    'url': item.get('url'),
                    'author': item.get('author'),
                    'ingestion_date': item.get('ingestion_date'),
                    'text_preview': content[:200]
                }
            )
            results.append(base_result)
            
            if len(results) >= top_k:
                break
                
        return results
    
    async def _process_keyword_results_with_citations(self, search_results: List[Dict],
                                                     query: str,
                                                     filter_metadata: Optional[Dict[str, Any]],
                                                     top_k: int) -> List[SearchResultWithCitations]:
        """Process keyword search results with citations - batch processing for efficiency."""
        import re
        import asyncio
        
        results = []
        
        # Process in batches for efficiency
        batch_size = KEYWORD_BATCH_SIZE
        
        for i in range(0, len(search_results), batch_size):
            batch = search_results[i:i + batch_size]
            
            # Process batch concurrently
            batch_results = await asyncio.gather(*[
                self._create_keyword_result_with_citations(item, query, filter_metadata)
                for item in batch
            ])
            
            # Filter out None results and add to results
            for result in batch_results:
                if result is not None:
                    results.append(result)
                    if len(results) >= top_k:
                        return results
        
        return results
    
    async def _create_keyword_result_with_citations(self, item: Dict, query: str,
                                                   filter_metadata: Optional[Dict[str, Any]]) -> Optional[SearchResultWithCitations]:
        """Create a single keyword result with citations."""
        import re
        
        # Apply metadata filters
        if filter_metadata:
            item_meta = {
                'media_type': item.get('type'),
                'source': 'media',
                'author': item.get('author')
            }
            if not all(item_meta.get(k) == v for k, v in filter_metadata.items() if k in item_meta):
                return None
        
        # Create base result
        content = item.get('content', '')[:1000]
        base_metadata = {
            'doc_id': str(item['id']),
            'doc_title': item.get('title', 'Untitled'),
            'media_type': item.get('type'),
            'url': item.get('url'),
            'author': item.get('author'),
            'ingestion_date': item.get('ingestion_date'),
            'text_preview': content[:200]
        }
        
        # Find citations
        escaped_query = re.escape(query)
        pattern = re.compile(escaped_query, re.IGNORECASE)
        
        full_content = item.get('content', '')
        matches = list(pattern.finditer(full_content))
        
        citations = []
        
        # Create citations for limited number of matches
        for match in matches[:MAX_CITATION_MATCHES]:
            start_context = max(0, match.start() - CITATION_CONTEXT_CHARS)
            end_context = min(len(full_content), match.end() + CITATION_CONTEXT_CHARS)
            
            citation = Citation(
                document_id=str(item['id']),
                document_title=item.get('title', 'Untitled'),
                chunk_id=f"media_{item['id']}_kw_{match.start()}",
                text=full_content[start_context:end_context],
                start_char=match.start(),
                end_char=match.end(),
                confidence=1.0,
                match_type=CitationType.EXACT,
                metadata={
                    'query': query,
                    'match_text': match.group(),
                    'media_type': item.get('type')
                }
            )
            citations.append(citation)
        
        # If no exact matches, create general citation
        if not citations and query.lower() in full_content.lower():
            citation = Citation(
                document_id=str(item['id']),
                document_title=item.get('title', 'Untitled'),
                chunk_id=f"media_{item['id']}_general",
                text=content,
                start_char=0,
                end_char=len(content),
                confidence=0.7,
                match_type=CitationType.KEYWORD,
                metadata={
                    'query': query,
                    'media_type': item.get('type')
                }
            )
            citations.append(citation)
        
        return SearchResultWithCitations(
            id=f"media_{item['id']}",
            score=KEYWORD_SEARCH_SCORE,
            document=content,
            metadata=base_metadata,
            citations=citations
        )
    
    def _escape_fts5_query(self, query: str) -> str:
        """
        Properly escape FTS5 query to prevent SQL injection.
        
        FTS5 special characters that need escaping:
        - Double quotes (") for phrase queries
        - Parentheses for grouping
        - Operators: OR, AND, NOT, NEAR
        - Wildcards: *
        - Column filters: :
        
        For safety, we'll use FTS5 phrase query syntax which treats
        the entire query as a literal phrase.
        
        Args:
            query: Raw search query
            
        Returns:
            Safely escaped query for FTS5
        """
        # Escape any double quotes within the query by doubling them
        # This is the proper way to escape quotes in FTS5 phrase queries
        escaped_query = query.replace('"', '""')
        
        # Validate query length to prevent DoS
        MAX_QUERY_LENGTH = 1000
        if len(escaped_query) > MAX_QUERY_LENGTH:
            logger.warning(f"Query truncated from {len(escaped_query)} to {MAX_QUERY_LENGTH} characters")
            escaped_query = escaped_query[:MAX_QUERY_LENGTH]
        
        # For safety, treat entire query as a phrase by wrapping in quotes
        # This prevents any FTS5 operators or special syntax from being interpreted
        # The phrase query syntax is the safest approach for user input
        return f'"{escaped_query}"'
    
    def _perform_fts5_search(self, pool, query: str, limit: int) -> List[Dict[str, Any]]:
        """
        Perform FTS5 search using connection pool with proper SQL injection prevention.
        
        Args:
            pool: Connection pool instance
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of search results
        """
        # Properly escape the query for FTS5
        escaped_query = self._escape_fts5_query(query)
        
        # Validate limit parameter
        if not isinstance(limit, int) or limit < 1:
            limit = 100  # Safe default
        limit = min(limit, 1000)  # Cap maximum results
        
        sql = """
        SELECT 
            m.id,
            m.title,
            m.content,
            m.url,
            m.type,
            m.author,
            m.ingestion_date,
            m.tags,
            rank
        FROM Media m
        JOIN MediaSearchIndex msi ON m.id = msi.media_id
        WHERE MediaSearchIndex MATCH ?
        AND m.is_trash = 0
        ORDER BY rank
        LIMIT ?
        """
        
        results = []
        try:
            with pool.get_connection() as conn:
                cursor = conn.cursor()
                # Use parameterized query - the escaped_query is already safe
                cursor.execute(sql, (escaped_query, limit))
                
                for row in cursor:
                    results.append({
                        'id': row['id'],
                        'title': row['title'],
                        'content': row['content'],
                        'url': row['url'],
                        'type': row['type'],
                        'author': row['author'],
                        'ingestion_date': row['ingestion_date'],
                        'tags': row['tags']
                    })
        except Exception as e:
            logger.error(f"FTS5 search failed for query '{query}': {e}")
            # Re-raise with more context
            raise RuntimeError(f"Database search failed: {str(e)}") from e
        
        return results
    
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
    
    async def clear_cache_async(self):
        """Clear all caches asynchronously."""
        self.embeddings.clear_cache()
        await self.cache.clear_async()
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
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of the RAG service.
        
        Returns:
            Dictionary with health status information
        """
        return get_health_status()
    
    def get_document_count(self) -> int:
        """Get the number of indexed documents."""
        return self._docs_indexed
    
    def get_chunk_count(self) -> int:
        """Get the total number of chunks in the index."""
        stats = self.vector_store.get_collection_stats()
        return stats.get("count", 0)
    
    def close(self):
        """Clean up all resources including connection pools."""
        try:
            # Close embeddings service
            self.embeddings.close()
            
            # Close all database connection pools
            from .db_connection_pool import close_all_pools
            close_all_pools()
            
            # Clear cache
            if hasattr(self, 'cache'):
                self.cache.clear()
            
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
    embedding_model: Optional[Union[str, RAGConfig]] = None,
    vector_store: str = "chroma",
    persist_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> RAGService:
    """
    Create a RAG service with common configurations.
    
    Args:
        embedding_model: Embedding model to use, or a RAGConfig object
        vector_store: Vector store type ("chroma" or "memory")
        persist_dir: Directory for persistence (if using chroma)
        **kwargs: Additional config parameters
        
    Returns:
        Configured RAGService instance
    """
    # If the first argument is already a RAGConfig, use it directly
    if isinstance(embedding_model, RAGConfig):
        return RAGService(embedding_model)
    
    # Create a default config
    config = RAGConfig()
    
    # Update embedding model if provided
    if embedding_model:
        config.embedding.model = embedding_model
    
    # Update vector store settings
    config.vector_store.type = vector_store
    if persist_dir:
        config.vector_store.persist_directory = Path(persist_dir)
    
    # Update any additional kwargs that match config structure
    for key, value in kwargs.items():
        if hasattr(config.embedding, key):
            setattr(config.embedding, key, value)
        elif hasattr(config.vector_store, key):
            setattr(config.vector_store, key, value)
        elif hasattr(config.chunking, key):
            setattr(config.chunking, key, value)
        elif hasattr(config.search, key):
            setattr(config.search, key, value)
    
    return RAGService(config)