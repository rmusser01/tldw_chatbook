"""
Main RAG service coordinator with citations support.

This is the main entry point for the simplified RAG implementation, coordinating
embeddings, vector stores, chunking, and search operations.
"""

import asyncio
import re
import sqlite3
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from typing import (
    Any,
    Collection,
    Dict,
    FrozenSet,
    Hashable,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    Union,
)

from loguru import logger

# Optional numpy import
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None
import psutil

from tldw_chatbook.config import load_settings
from tldw_chatbook.Metrics.metrics_logger import (
    log_counter,
    log_histogram,
    log_gauge,
    timeit,
)
from .embeddings_wrapper import EmbeddingsServiceWrapper
from .vector_store import create_vector_store, SearchResult, SearchResultWithCitations
from .citations import Citation, CitationType, merge_citations
from .config import RAGConfig, DEFAULT_HYBRID_POOL_MULTIPLIER
from .collection_fingerprint import fingerprinted_collection_name, collection_provenance
from ..fusion import (
    reciprocal_rank_fusion,
    resolve_hybrid_alpha,
    resolve_rrf_k,
    interleave_rankings,
    DEFAULT_RRF_K,
)
from ..chunking_service import ChunkingService
from .simple_cache import SimpleRAGCache
from .db_connection_pool import get_connection_pool
from .indexing_helpers import (
    chunk_documents_batch,
    generate_embeddings_batch,
    store_documents_batch,
)
from .health_check import init_health_checker, get_health_status
from .data_models import IndexingResult


# Load constants from config with fallbacks
_config = load_settings()
_rag_service_config = _config.get("rag", {}).get("service", {})

# Constants from config
DEFAULT_EMBEDDING_DIM = _rag_service_config.get("default_embedding_dim", 768)
KEYWORD_SEARCH_SCORE = _rag_service_config.get("keyword_search_score", 0.8)
MAX_CITATION_MATCHES = _rag_service_config.get("max_citation_matches", 3)
CITATION_CONTEXT_CHARS = _rag_service_config.get("citation_context_chars", 50)
# Two keyword spans this close together, with nothing alphanumeric between
# them, are one piece of evidence (e.g. the two tokens of a contiguous
# "spindle runout" match) rather than two separate citations.
CITATION_SPAN_MERGE_GAP_CHARS = _rag_service_config.get(
    "citation_span_merge_gap_chars", 3
)
# Confidence for a span that evidences only SOME of the query's tokens --
# a real match, but weaker evidence than a span covering the whole query.
PARTIAL_CITATION_CONFIDENCE = _rag_service_config.get(
    "partial_citation_confidence", 0.8
)
KEYWORD_BATCH_SIZE = _rag_service_config.get("keyword_batch_size", 10)
FTS5_CONNECTION_POOL_SIZE = _rag_service_config.get("fts5_connection_pool_size", 3)
CACHE_TIMEOUT_SECONDS = _rag_service_config.get("cache_timeout_seconds", 3600.0)
MAX_QUERY_LENGTH = _rag_service_config.get("max_query_length", 1000)
DEFAULT_FTS5_LIMIT = _rag_service_config.get("default_fts5_limit", 100)
MAX_FTS5_LIMIT = _rag_service_config.get("max_fts5_limit", 1000)
SEARCH_RESULT_MULTIPLIER = _rag_service_config.get("search_result_multiplier", 2)
MIN_SYSTEM_MEMORY_MB = _rag_service_config.get("min_system_memory_mb", 500)
MEMORY_PRESSURE_REDUCTION = _rag_service_config.get("memory_pressure_reduction", 0.2)
MAX_DOCUMENT_SIZE_MB = _rag_service_config.get("max_document_size_mb", 10)
EMBEDDING_IDLE_TIMEOUT = _rag_service_config.get("embedding_idle_timeout", 900)
CHUNK_PROGRESS_INTERVAL = _rag_service_config.get("chunk_progress_interval", 10)
EMBEDDING_PROGRESS_INTERVAL = _rag_service_config.get("embedding_progress_interval", 5)
DEFAULT_BATCH_SIZE = _rag_service_config.get(
    "batch_size", 32
)  # This one matches the rag.embedding.batch_size


# The keyword leg's sub-leg vocabulary. These MUST stay byte-identical to
# `ingestion_indexing.ITEM_TYPE_*` (the singular `media`/`note`/
# `conversation` the vector leg stamps): `_fusion_doc_key` compares the raw
# strings, so a plural or variant spelling would leave keyword rows present
# but never merging with their vector twins -- a silent, test-green
# reversion of TASK-3996's purpose. They are copies rather than imports so
# the search path does not depend on the indexing module (which reaches back
# into this package for the service factory); drift is caught instead of
# prevented, by `test_keyword_leg_chacha.test_cross_leg_merge_per_source_
# type`, which builds its vector rows from the REAL `ingestion_indexing`
# documents and fails the moment the two definitions disagree.
SOURCE_TYPE_MEDIA = "media"
SOURCE_TYPE_NOTE = "note"
SOURCE_TYPE_CONVERSATION = "conversation"

# Every source type the keyword (FTS5) leg has a sub-leg for, and the two of
# those that live in the ChaChaNotes database. A caller's
# ``keyword_source_types`` selection is expressed in THIS vocabulary (the
# engine's singular spelling), never in a UI's plural scope identifiers.
KEYWORD_LEG_SOURCE_TYPES = frozenset(
    {SOURCE_TYPE_MEDIA, SOURCE_TYPE_NOTE, SOURCE_TYPE_CONVERSATION}
)
CHACHA_KEYWORD_SOURCE_TYPES = frozenset({SOURCE_TYPE_NOTE, SOURCE_TYPE_CONVERSATION})


def _resolve_keyword_source_types(
    selection: Optional[Collection[str]],
) -> FrozenSet[str]:
    """Resolve a caller's keyword-leg source-type selection.

    TASK-14751. TASK-3996 made the keyword leg three sub-legs sharing one
    fixed ``top_k`` budget, round-robined rank-fairly. Callers that only
    want some of those types (the Library post-filters the fused rows by the
    user's selected scope) were spending up to two thirds of the budget on
    rows they then discarded; a media-only hybrid search over an empty
    vector index showed roughly a third of the media rows the pre-TASK-3996
    leg returned. Naming the types up front lets the leg give its whole
    budget to the sub-legs whose rows will actually survive.

    Args:
        selection: Source types in the engine's vocabulary, or ``None`` for
            "every sub-leg" -- the behavior of every caller that predates
            this parameter.

    Returns:
        The selected subset of ``KEYWORD_LEG_SOURCE_TYPES``. Unrecognized
        values are dropped with one debug log rather than raising: a
        selection is a retrieval hint, and failing open to fewer sub-legs
        can never be worse than failing the search. An empty result means
        "run no sub-legs" -- which is a real request, not a mistake, and is
        why ``None`` and ``set()`` are deliberately different.
    """
    if selection is None:
        return KEYWORD_LEG_SOURCE_TYPES
    requested = frozenset(str(source_type) for source_type in selection)
    unknown = requested - KEYWORD_LEG_SOURCE_TYPES
    if unknown:
        logger.debug(
            "Ignoring unknown keyword-leg source type(s) {}; the leg serves {}",
            sorted(unknown),
            sorted(KEYWORD_LEG_SOURCE_TYPES),
        )
    return requested & KEYWORD_LEG_SOURCE_TYPES


async def _no_keyword_rows() -> List[Any]:
    """A skipped sub-leg's contribution: nothing, and no query to get it.

    Used so ``_keyword_search`` can keep gathering a fixed pair of awaitables
    while an unselected sub-leg never touches a database.
    """
    return []


# Sanity ceiling for _resolve_hybrid_pool_multiplier (TASK-4110 review,
# minor a). Fusion still narrows back to top_k regardless of how wide the
# legs over-fetch, so a multiplier this large protects nothing further and
# only multiplies retrieval cost -- an absurd config value (typo, a stray
# extra zero) is capped rather than honored outright.
MAX_HYBRID_POOL_MULTIPLIER = 100


def _resolve_hybrid_pool_multiplier(value: Any) -> int:
    """Resolve ``config.search.hybrid_pool_multiplier`` for ``_hybrid_search``.

    Use-time validation, matching ``resolve_hybrid_alpha``/``resolve_rrf_k``'s
    pattern: an invalid (non-numeric) config value falls back to
    ``DEFAULT_HYBRID_POOL_MULTIPLIER`` -- this field's OWN default (2), not
    the separate module-level ``SEARCH_RESULT_MULTIPLIER`` constant. The two
    happened to both be 2 but are different knobs (``SEARCH_RESULT_
    MULTIPLIER`` governs ``_semantic_search``'s own internal over-fetch on
    every search path, untouched by this field); falling back to the wrong
    one would silently hand a user's tuned ``search_result_multiplier`` back
    out of an invalid ``hybrid_pool_multiplier`` (TASK-4110 review minor b --
    see ``DEFAULT_HYBRID_POOL_MULTIPLIER``'s docstring in config.py for the
    release-note disclosure). Any value below 1 is floored to 1 -- each
    hybrid leg must fetch at least ``top_k`` candidates for fusion to have
    anything to work with -- and any value above
    ``MAX_HYBRID_POOL_MULTIPLIER`` is capped there. Never raises: a
    misconfigured pipeline must not abort search at merge time.

    Args:
        value: Caller/config-supplied multiplier, if any.

    Returns:
        An int in ``[1, MAX_HYBRID_POOL_MULTIPLIER]``.
    """
    try:
        multiplier = int(value)
    except (TypeError, ValueError, OverflowError):
        # OverflowError: `int()` on an infinite float raises OverflowError
        # rather than ValueError -- TOML accepts a literal `inf`/`-inf`, so
        # `hybrid_pool_multiplier = inf` reaches here straight from a
        # hand-edited config (Qodo PR-1487, same defect as
        # ``fusion.resolve_rrf_k``). Must fall back like every other
        # invalid value, not raise before either hybrid leg launches.
        logger.warning(
            f"Invalid hybrid_pool_multiplier {value!r}; falling back to "
            f"{DEFAULT_HYBRID_POOL_MULTIPLIER}"
        )
        return DEFAULT_HYBRID_POOL_MULTIPLIER
    if multiplier < 1:
        logger.warning(f"hybrid_pool_multiplier {multiplier} < 1; flooring to 1")
        return 1
    if multiplier > MAX_HYBRID_POOL_MULTIPLIER:
        logger.warning(
            f"hybrid_pool_multiplier {multiplier} > {MAX_HYBRID_POOL_MULTIPLIER}; "
            f"capping to {MAX_HYBRID_POOL_MULTIPLIER}"
        )
        return MAX_HYBRID_POOL_MULTIPLIER
    return multiplier


def _fusion_doc_key(result: Any) -> Hashable:
    """Document-identity fusion key: (source_type, source_id-or-doc_id).

    TASK-3994: the two hybrid legs speak different id spaces -- the FTS leg
    emits document rows (``media_15``) and the vector leg emits chunk rows
    (``media_15_chunk_0``) -- so matching on ``SearchResult.id`` could never
    fuse the same document across legs. Both legs *do* agree on ingestion
    metadata, which is what this key reads:

    * vector rows carry ``source_id`` (the bare row id, spread from
      ``ingestion_indexing.media_document`` into every chunk) plus a
      ``doc_id`` that is the PREFIXED document id (``media_15``);
    * keyword rows carry ``doc_id`` AND ``source_id``, both the bare row id
      (``15``) -- built from scratch in ``_keyword_row_metadata``. When this
      key was written the keyword leg stamped only ``doc_id``; TASK-3996
      added ``source_id`` so its note/conversation rows speak the vector
      leg's id space (media rows got it too, for one id space rather than
      two).

    The precedence -- ``source_id`` first, ``doc_id`` as a fallback --
    therefore still matters for any producer that stamps only one of them:
    comparing ``doc_id`` to ``doc_id`` across the legs would match ``15``
    against ``media_15`` and never fuse anything.

    ``source_type`` is compared as the raw string the indexers write (the
    singular ``ITEM_TYPE_*`` vocabulary: ``media`` / ``note`` /
    ``conversation``); it keeps note 15 and media 15 apart.

    Args:
        result: A leg result (``SearchResult`` / ``SearchResultWithCitations``).

    Returns:
        ``(source_type, source_id)`` when both components are present,
        otherwise the row id -- preserving the pre-fix no-merge behavior for
        rows without ingestion metadata (e.g. hand-built rows in tests, or a
        future producer that stamps neither key).
    """
    md = getattr(result, "metadata", None) or {}
    source_type = md.get("source_type")
    source_id = md.get("source_id") or md.get("doc_id")
    if source_type and source_id:
        return (str(source_type), str(source_id))
    return result.id


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
        logger.info(
            "RAG Service Configuration",
            extra={
                "embedding_model": self.config.embedding_model,
                "vector_store_type": self.config.vector_store_type,
                "collection_name": self.config.collection_name,
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "device": self.config.device,
            },
        )

        # Initialize embeddings using wrapper around existing library
        logger.info(
            f"Initializing embeddings service with model: {self.config.embedding_model}"
        )

        # Get model cache directory from config
        if str(self.config.embedding_model).lower() in {
            "mock",
            "mock-embedding-model",
            "mock_embedding_model",
        }:
            cache_dir = None
        else:
            from tldw_chatbook.config import get_model_cache_dir

            cache_dir = str(get_model_cache_dir())

        self.embeddings = EmbeddingsServiceWrapper(
            model_name=self.config.embedding_model,
            cache_size=self.config.embedding_cache_size,
            device=self.config.device,
            cache_dir=cache_dir,
        )

        # Initialize vector store
        logger.info(f"Initializing {self.config.vector_store_type} vector store")
        self.vector_store = create_vector_store(
            store_type=self.config.vector_store_type,
            persist_directory=self.config.persist_directory,
            collection_name=fingerprinted_collection_name(self.config),
            distance_metric=self.config.distance_metric,
            collection_metadata=collection_provenance(self.config),
        )

        # Initialize chunking service
        self.chunking = ChunkingService()
        logger.info("Initialized chunking service")

        # Initialize cache
        cache_config = (
            config.search.__dict__ if hasattr(config.search, "__dict__") else {}
        )
        cache_size = cache_config.get("cache_size", 100)
        cache_ttl = cache_config.get("cache_ttl", 3600)
        cache_enabled = cache_config.get("enable_cache", True)

        # Search-type specific TTLs (optional)
        ttl_by_search_type = {}
        if hasattr(config.search, "semantic_cache_ttl"):
            ttl_by_search_type["semantic"] = config.search.semantic_cache_ttl
        if hasattr(config.search, "keyword_cache_ttl"):
            ttl_by_search_type["keyword"] = config.search.keyword_cache_ttl
        if hasattr(config.search, "hybrid_cache_ttl"):
            ttl_by_search_type["hybrid"] = config.search.hybrid_cache_ttl

        # Search results belong to this service's vector-store/profile state.
        # A process-global cache can return documents from another collection
        # and cannot be invalidated reliably when one service mutates.
        self.cache = SimpleRAGCache(
            max_size=cache_size,
            ttl_seconds=cache_ttl,
            enabled=cache_enabled,
            ttl_by_search_type=ttl_by_search_type if ttl_by_search_type else None,
        )
        logger.info(
            f"Initialized cache: size={cache_size}, ttl={cache_ttl}s, enabled={cache_enabled}, "
            f"search_type_ttls={ttl_by_search_type}"
        )

        # Log initialization metrics
        log_counter(
            "rag_service_initialized",
            labels={
                "model": self.config.embedding_model,
                "vector_store": self.config.vector_store_type,
                "device": self.config.device,
            },
        )

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

        # Create dedicated thread pool for CPU-intensive work
        # Size based on CPU count but capped to avoid excessive threads
        cpu_count = psutil.cpu_count(logical=False) or 2
        self._executor = ThreadPoolExecutor(
            max_workers=min(cpu_count * 2, 8), thread_name_prefix="rag_worker"
        )
        logger.info(
            f"Initialized thread pool with {self._executor._max_workers} workers"
        )

    # === Indexing Methods ===

    @timeit("rag_indexing_document")
    async def index_document(
        self,
        doc_id: str,
        content: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        chunking_method: Optional[str] = None,
    ) -> IndexingResult:
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
        str(uuid.uuid4())

        # Input validation
        if not doc_id or not isinstance(doc_id, str):
            raise ValueError("doc_id must be a non-empty string")

        if not isinstance(content, str):
            raise TypeError("content must be a string")
        if not content:
            raise ValueError("content must be a non-empty string")

        # Document size limit (configurable)
        max_doc_size = getattr(
            self.config, "max_document_size", MAX_DOCUMENT_SIZE_MB * 1024 * 1024
        )
        if len(content) > max_doc_size:
            raise ValueError(
                f"Document too large: {len(content)} bytes exceeds limit of {max_doc_size} bytes"
            )

        # Validate chunk parameters if provided
        if chunk_size is not None and (
            not isinstance(chunk_size, int) or chunk_size < 1
        ):
            raise ValueError("chunk_size must be a positive integer")

        if chunk_overlap is not None and (
            not isinstance(chunk_overlap, int) or chunk_overlap < 0
        ):
            raise ValueError("chunk_overlap must be a non-negative integer")

        if (
            chunk_overlap is not None
            and chunk_size is not None
            and chunk_overlap >= chunk_size
        ):
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
                chunking_method or self.config.chunking_method,
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
                    success=True,
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
                    "chunk_id": chunk_ids[i],
                    "chunk_index": i,
                    "chunk_start": chunk.get("start_char", 0),
                    "chunk_end": chunk.get("end_char", len(chunk["text"])),
                    "chunk_size": len(chunk["text"]),
                    "word_count": chunk.get("word_count", 0),
                    # Store part of text for keyword matching
                    "text_preview": chunk["text"][:200],
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

            logger.info(
                f"Indexed document {doc_id} with {len(chunks)} chunks in {elapsed:.2f}s "
                + f"(chunk: {chunk_time:.2f}s, embed: {embed_time:.2f}s, store: {store_time:.2f}s)"
            )

            return IndexingResult(
                doc_id=doc_id,
                chunks_created=len(chunks),
                time_taken=elapsed,
                success=True,
            )

        except Exception as e:
            log_counter("rag_document_index_error", labels={"error": type(e).__name__})
            logger.opt(exception=True).error(f"Failed to index document {doc_id}: {e}")
            return IndexingResult(
                doc_id=doc_id,
                chunks_created=0,
                time_taken=time.time() - start_time,
                success=False,
                error=str(e),
            )

    def index_document_sync(
        self, doc_id: str, content: str, **kwargs
    ) -> IndexingResult:
        """Synchronous version of index_document."""
        return asyncio.run(self.index_document(doc_id, content, **kwargs))

    async def index_batch(
        self,
        documents: List[Dict[str, Any]],
        show_progress: bool = True,
        continue_on_error: bool = True,
    ) -> List[IndexingResult]:
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
            if show_progress and i % CHUNK_PROGRESS_INTERVAL == 0 and i > 0:
                logger.info(f"Indexing progress: {i}/{total} documents")

            try:
                result = await self.index_document(
                    doc_id=doc["id"],
                    content=doc["content"],
                    title=doc.get("title"),
                    metadata=doc.get("metadata"),
                )
                results.append(result)

            except Exception as e:
                logger.error(
                    f"Failed to index document {doc.get('id', 'unknown')}: {e}"
                )
                if not continue_on_error:
                    raise
                results.append(
                    IndexingResult(
                        doc_id=doc.get("id", "unknown"),
                        chunks_created=0,
                        time_taken=0,
                        success=False,
                        error=str(e),
                    )
                )

        if show_progress:
            successful = sum(1 for r in results if r.success)
            logger.info(f"Indexed {successful}/{total} documents successfully")

        return results

    async def index_batch_optimized(
        self,
        documents: List[Dict[str, Any]],
        show_progress: bool = True,
        batch_size: int = 32,
    ) -> List[IndexingResult]:
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
        logger.info(
            f"Chunking completed in {chunk_time:.2f}s, total chunks: {len(all_chunks)}"
        )

        if not all_chunks:
            logger.warning("No chunks created from any documents")
            return failed_results

        # Phase 2: Generate embeddings in batches
        embed_start = time.time()
        chunk_texts = [chunk["text"] for chunk in all_chunks]
        all_embeddings, failed_embedding_indices = await generate_embeddings_batch(
            self, chunk_texts, batch_size, show_progress, retry_failed=True
        )
        embed_time = time.time() - embed_start
        logger.info(f"Embedding generation completed in {embed_time:.2f}s")

        # Phase 3: Store documents with their embeddings
        store_start = time.time()
        storage_results = await store_documents_batch(
            self,
            documents,
            doc_chunk_info,
            all_embeddings,
            batch_start_time,
            failed_embedding_indices,
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
    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        search_type: Literal["semantic", "hybrid", "keyword"] = "semantic",
        filter_metadata: Optional[Dict[str, Any]] = None,
        include_citations: Optional[bool] = None,
        score_threshold: Optional[float] = None,
        *,
        metadata_allowlist: Optional[Mapping[str, Collection[str]]] = None,
        keyword_source_types: Optional[Collection[str]] = None,
    ) -> Union[List[SearchResult], List[SearchResultWithCitations]]:
        """
        Search with optional citations.

        Args:
            query: Search query text
            top_k: Number of results to return (default from config)
            search_type: Type of search to perform
            filter_metadata: Metadata filters to apply (Python-side equality
                post-filter, applied after the store call; unchanged from
                before, kept for backward compatibility)
            include_citations: Whether to include citations (default from config)
            score_threshold: Minimum score threshold (default from config)
            metadata_allowlist: Metadata key -> allowed values, pushed down
                into the vector store's own candidate selection instead of
                filtered afterward. Only supported for
                ``search_type="semantic"``; see ``_semantic_search``. Passing
                a non-empty allowlist with ``search_type="hybrid"`` or
                ``search_type="keyword"`` raises ``ValueError`` rather than
                silently ignoring the scoping request.
            keyword_source_types: Source types the keyword (FTS5) leg should
                budget for, in the engine's vocabulary
                (``media``/``note``/``conversation``). ``None`` -- the
                default, and every pre-TASK-14751 caller -- serves all three.
                The mirror image of ``metadata_allowlist``: it scopes the
                KEYWORD leg only, so passing it with
                ``search_type="semantic"`` raises ``ValueError`` rather than
                silently ignoring the scoping request. It is part of the
                cache key, so two selections of the same query never share
                an entry.

        Returns:
            List of search results (with or without citations)

        Raises:
            ValueError: If ``metadata_allowlist`` is provided with a
                ``search_type`` other than ``"semantic"``, or if
                ``keyword_source_types`` is provided with
                ``search_type="semantic"``.
        """
        if metadata_allowlist and search_type != "semantic":
            raise ValueError(
                "metadata_allowlist is only supported for search_type='semantic'"
            )
        if keyword_source_types is not None and search_type == "semantic":
            raise ValueError(
                "keyword_source_types is not supported for search_type='semantic' "
                "(the semantic leg has no keyword sub-legs to scope)"
            )

        # Use defaults from config if not specified
        top_k = top_k or self.config.default_top_k
        include_citations = (
            include_citations
            if include_citations is not None
            else self.config.include_citations
        )
        score_threshold = (
            score_threshold
            if score_threshold is not None
            else self.config.score_threshold
        )

        # Create correlation ID for tracking
        correlation_id = str(uuid.uuid4())

        # Log search metrics
        log_counter("rag_search_attempt", labels={"type": search_type})
        log_histogram("rag_search_query_length", len(query))
        self._search_type_counts[search_type] += 1

        # The RESOLVED fusion parameters a HYBRID search will actually use,
        # so the cache key below reflects them (TASK-4110 review, important
        # 1). Without this, two hybrid searches identical except for
        # `rrf_k` (or `hybrid_alpha`, or `hybrid_pool_multiplier`) shared
        # one cache entry -- the SECOND request was silently served the
        # FIRST's stale results forever, which would have made Task 4's
        # per-k strategy sweep report every value as "no effect" on a
        # single cached service. Resolved (not raw config) values, so two
        # configs that happen to resolve to the same effective number
        # correctly SHARE an entry rather than needlessly splitting one.
        # `None` for semantic/keyword: those legs never depend on these
        # three, so their cache key stays exactly as it was.
        hybrid_fusion_key: Optional[Tuple[float, int, int]] = None
        if search_type == "hybrid":
            hybrid_fusion_key = (
                resolve_hybrid_alpha(self.config.search.hybrid_alpha),
                resolve_rrf_k(self.config.search.rrf_k),
                _resolve_hybrid_pool_multiplier(
                    self.config.search.hybrid_pool_multiplier
                ),
            )

        # Check cache first
        cached_result = await self.cache.get_async(
            query,
            search_type,
            top_k,
            filter_metadata,
            metadata_allowlist,
            keyword_source_types=keyword_source_types,
            hybrid_fusion=hybrid_fusion_key,
        )
        if cached_result is not None:
            results, context = cached_result
            log_counter("rag_search_cache_hit", labels={"type": search_type})
            logger.info(f"[{correlation_id}] Cache hit for query: '{query[:50]}...'")
            return results

        log_counter("rag_search_cache_miss", labels={"type": search_type})

        self._searches_performed += 1
        start_time = time.time()

        try:
            logger.info(
                f"[{correlation_id}] Performing {search_type} search with top_k={top_k}, threshold={score_threshold}"
            )

            if search_type == "semantic":
                results = await self._semantic_search(
                    query,
                    top_k,
                    filter_metadata,
                    include_citations,
                    score_threshold,
                    metadata_allowlist=metadata_allowlist,
                )
            elif search_type == "hybrid":
                results = await self._hybrid_search(
                    query,
                    top_k,
                    filter_metadata,
                    include_citations,
                    score_threshold,
                    keyword_source_types=keyword_source_types,
                )
            elif search_type == "keyword":
                results = await self._keyword_search(
                    query,
                    top_k,
                    filter_metadata,
                    include_citations,
                    keyword_source_types=keyword_source_types,
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
                    log_histogram(
                        "rag_search_score_distribution",
                        score,
                        labels={"rank": str(i + 1), "type": search_type},
                    )

            elapsed = time.time() - start_time

            # Log search success metrics
            log_counter("rag_search_success", labels={"type": search_type})
            log_histogram("rag_search_time", elapsed, labels={"type": search_type})
            log_histogram("rag_search_results_count", len(results))
            log_gauge("rag_total_searches_performed", self._searches_performed)

            # Log search type distribution
            total_searches = sum(self._search_type_counts.values())
            for stype, count in self._search_type_counts.items():
                log_gauge(
                    f"rag_search_type_{stype}_ratio",
                    count / total_searches if total_searches > 0 else 0,
                )

            logger.info(
                f"[{correlation_id}] Search completed in {elapsed:.2f}s, found {len(results)} results"
            )

            # Cache the results
            # For caching, we need to extract a simple context string
            context = self._extract_context_from_results(results)
            await self.cache.put_async(
                query,
                search_type,
                top_k,
                results,
                context,
                filter_metadata,
                metadata_allowlist,
                keyword_source_types=keyword_source_types,
                hybrid_fusion=hybrid_fusion_key,
            )

            return results

        except Exception as e:
            log_counter(
                "rag_search_error",
                labels={"type": search_type, "error": type(e).__name__},
            )
            logger.opt(exception=True).error(f"[{correlation_id}] Search failed: {e}")
            raise

    def search_sync(
        self, query: str, **kwargs
    ) -> Union[List[SearchResult], List[SearchResultWithCitations]]:
        """Synchronous version of search."""
        return asyncio.run(self.search(query, **kwargs))

    async def _semantic_search(
        self,
        query: str,
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]] = None,
        include_citations: bool = True,
        score_threshold: float = 0.0,
        *,
        metadata_allowlist: Optional[Mapping[str, Collection[str]]] = None,
    ) -> Union[List[SearchResult], List[SearchResultWithCitations]]:
        """Perform semantic similarity search.

        Args:
            query: Search query text.
            top_k: Number of results to return.
            filter_metadata: Metadata equality filters applied *after* the
                store call (Python-side post-filter, unchanged from before
                this parameter existed). Kept for backward compatibility;
                prefer ``metadata_allowlist`` for scoping, since a narrow
                post-filter can starve top-k on large corpora.
            include_citations: Whether to fetch citations from the store.
            score_threshold: Minimum similarity score to keep.
            metadata_allowlist: Metadata key -> allowed values, threaded
                through to the vector store's ``search``/
                ``search_with_citations`` so out-of-scope candidates are
                excluded before the store ranks and truncates to
                ``top_k * SEARCH_RESULT_MULTIPLIER`` results.

        Returns:
            Up to ``top_k`` search results, most similar first.
        """
        # Create query embedding
        logger.debug("Creating query embedding")
        query_embedding = await self.embeddings.create_embeddings_async([query])
        query_embedding = query_embedding[0]

        # Search vector store
        if include_citations:
            results = self.vector_store.search_with_citations(
                query_embedding,
                query,
                top_k * SEARCH_RESULT_MULTIPLIER,
                score_threshold,
                metadata_allowlist=metadata_allowlist,
            )
        else:
            results = self.vector_store.search(
                query_embedding,
                top_k * SEARCH_RESULT_MULTIPLIER,
                metadata_allowlist=metadata_allowlist,
            )
            # Apply score threshold for basic results
            results = [r for r in results if r.score >= score_threshold]

        # Apply metadata filters if provided
        if filter_metadata:
            results = [
                r
                for r in results
                if all(r.metadata.get(k) == v for k, v in filter_metadata.items())
            ]

        return results[:top_k]

    async def _keyword_search(
        self,
        query: str,
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]] = None,
        include_citations: bool = True,
        *,
        keyword_source_types: Optional[Collection[str]] = None,
    ) -> Union[List[SearchResult], List[SearchResultWithCitations]]:
        """
        Perform keyword (FTS5) search across media, notes and conversations.

        TASK-3996: this leg used to join ``media_fts`` and nothing else, so
        the keyword half of hybrid search could only ever return media rows
        -- notes and conversations were structurally unreachable through it
        no matter what the query said (29 of the P1 fixture corpus's 49
        documents). It is now three sub-legs over two databases:

        * media -- ``media_fts`` in the media DB, via the connection pool;
        * notes -- ``notes_fts`` in the ChaChaNotes DB;
        * conversations -- ``messages_fts`` in the ChaChaNotes DB, one row
          per matching conversation.

        The two ChaChaNotes sub-legs run over a READ-ONLY raw connection
        (never ``CharactersRAGDB``, whose constructor does schema work), and
        each sub-leg degrades independently: a missing chacha DB costs the
        notes/conversation rows and leaves media untouched, and vice versa.
        The leg is empty only when every sub-leg is empty or unavailable.

        The sub-legs are merged rank-fairly (``interleave_rankings``, round
        robin by rank position) rather than concatenated: FTS5 scores from
        different tables are not comparable, and concatenation would let one
        well-stocked source consume every ``top_k`` slot.

        TASK-14751 narrows *which* sub-legs run without touching that
        merge. ``keyword_source_types`` names the types the caller will
        actually keep (``None`` = all three, i.e. unchanged for every caller
        that predates it); the unnamed sub-legs are never queried, so the
        whole ``top_k`` budget goes to the ones that were asked for. A
        single-type selection therefore gets that sub-leg's full natural
        best-first order -- the pre-TASK-3996 behavior for media -- and a
        multi-type selection keeps the round robin among exactly the types
        selected.

        Args:
            query: Raw user query (escaped for FTS5 downstream).
            top_k: Maximum rows the merged leg returns.
            filter_metadata: Optional metadata equality filters.
            include_citations: Whether to build citation-carrying rows.
            keyword_source_types: Source types to serve, in the engine's
                vocabulary (``media``/``note``/``conversation``). ``None``
                serves all three; an empty collection serves none and
                returns ``[]`` without a database lookup; unrecognized
                values are dropped (see ``_resolve_keyword_source_types``).

        Returns:
            The merged leg, best first, capped at ``top_k``.
        """
        selected = _resolve_keyword_source_types(keyword_source_types)
        if not selected:
            # An explicitly empty selection is "no keyword leg" -- an
            # answer, not a failure. Hybrid degrades to its semantic leg
            # through the same disclosed path an empty FTS result already
            # takes.
            logger.debug(
                "Keyword search asked for no source types; returning no "
                "results without a database lookup."
            )
            return []

        # TASK-3995: a query with no FTS5-searchable tokens (empty,
        # whitespace-only, or all punctuation) escapes to "" and can only
        # ever match nothing. Short-circuit before resolving any DB
        # path or acquiring a connection -- no FTS5 call, no DB touch.
        if not self._escape_fts5_query(query):
            logger.debug(
                "Keyword search query has no FTS5-searchable tokens after "
                "escaping; returning no results without a database lookup."
            )
            return []

        chacha_types = selected & CHACHA_KEYWORD_SOURCE_TYPES
        media_ranking, chacha_rankings = await asyncio.gather(
            self._media_keyword_subleg(
                query, top_k, filter_metadata, include_citations
            )
            if SOURCE_TYPE_MEDIA in selected
            else _no_keyword_rows(),
            self._chacha_keyword_sublegs(
                query,
                top_k,
                filter_metadata,
                include_citations,
                source_types=chacha_types,
            )
            if chacha_types
            else _no_keyword_rows(),
        )

        rankings = [
            ranking for ranking in (media_ranking, *chacha_rankings) if ranking
        ]
        if not rankings:
            return []

        # Deduplicate on the same document identity fusion uses, so a
        # document that somehow appears in two sub-legs occupies one slot.
        results = interleave_rankings(rankings, key=_fusion_doc_key)[:top_k]
        logger.info(
            "Keyword search found {} results across {} sub-leg(s)",
            len(results),
            len(rankings),
        )
        return results

    async def _media_keyword_subleg(
        self,
        query: str,
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]] = None,
        include_citations: bool = True,
    ) -> Union[List[SearchResult], List[SearchResultWithCitations]]:
        """The media sub-leg of the keyword search: FTS5 over the media DB.

        Args:
            query: Raw user query (escaped for FTS5 downstream).
            top_k: Maximum rows this sub-leg contributes.
            filter_metadata: Optional metadata equality filters.
            include_citations: Whether to build citation-carrying rows.

        Returns:
            Media rows, best first; ``[]`` on any failure (this sub-leg
            never breaks the other two).
        """
        try:
            # Resolve the media DB path -- explicit override wins, otherwise
            # defer to the single authoritative resolver. No guessing across
            # a list of candidate filenames, and no create-on-miss: a search
            # must never have the side effect of creating a database.
            from tldw_chatbook.config import get_media_db_path
            from tldw_chatbook.Utils.path_validation import validate_path_simple
            from tldw_chatbook.Utils.private_paths import lexical_path

            db_path_raw = self.config.search.media_db_path or get_media_db_path()

            # Qodo PR #1428 finding 1: `config.search.media_db_path` is a
            # config-sourced override reaching a filesystem check + DB open
            # without running through path_validation.py. Mirror config.py's
            # `_get_custom_database_path` treatment for custom DB paths --
            # lexical normalization plus `validate_path_simple`'s
            # traversal/injection screen -- rather than a base-dir jail that
            # would reject a legitimate custom media DB living outside the
            # default data dir. `probe_existing=False` matches that same
            # helper: filesystem/symlink authority is deferred to the
            # private SQLite owner (MediaDatabase -> connect_private_sqlite)
            # that actually opens the file below.
            try:
                db_path = lexical_path(
                    validate_path_simple(
                        Path(str(db_path_raw)).expanduser(),
                        require_exists=False,
                        probe_existing=False,
                    )
                )
            except ValueError as e:
                logger.warning(
                    f"Rejected media_db_path from config ({db_path_raw!r}): "
                    f"{e}; keyword search returning no results (a search "
                    "never creates a database)."
                )
                return []

            if not db_path.exists() or not db_path.is_file():
                logger.warning(
                    f"Media database not found at {db_path}; keyword search "
                    "returning no results (a search never creates a database)."
                )
                return []

            # Get connection pool for this database
            pool_size = getattr(
                self.config.search,
                "fts5_connection_pool_size",
                FTS5_CONNECTION_POOL_SIZE,
            )
            pool = get_connection_pool(str(db_path), pool_size=pool_size)

            # Perform FTS5 search directly using connection pool with retry
            loop = asyncio.get_event_loop()
            retry_count = 0
            max_retries = 2

            while retry_count <= max_retries:
                try:
                    search_results = await loop.run_in_executor(
                        None,
                        self._perform_fts5_search,
                        pool,
                        query,
                        top_k * SEARCH_RESULT_MULTIPLIER,  # Get extra for filtering
                    )
                    break  # Success, exit retry loop
                except Exception as e:
                    retry_count += 1
                    if retry_count > max_retries:
                        logger.error(
                            f"FTS5 search failed after {max_retries} retries: {e}"
                        )
                        raise
                    else:
                        logger.warning(
                            f"FTS5 search attempt {retry_count} failed, retrying: {e}"
                        )
                        await asyncio.sleep(0.1 * retry_count)  # Exponential backoff

            # Process results in batches for better performance
            if include_citations:
                results = await self._process_keyword_results_with_citations(
                    search_results, query, filter_metadata, top_k
                )
            else:
                results = self._process_keyword_results_basic(
                    search_results, filter_metadata, top_k
                )

            logger.debug("Media keyword sub-leg found {} results", len(results))
            return results

        except Exception as e:
            logger.error(
                "Media keyword sub-leg failed (error_type={})",
                type(e).__name__,
            )
            # Log additional context for debugging
            logger.debug(
                f"Search parameters: top_k={top_k}, include_citations={include_citations}"
            )
            # Return empty list on error to maintain compatibility
            return []

    async def _chacha_keyword_sublegs(
        self,
        query: str,
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]] = None,
        include_citations: bool = True,
        *,
        source_types: Optional[Collection[str]] = None,
    ) -> List[Union[List[SearchResult], List[SearchResultWithCitations]]]:
        """The notes and conversation sub-legs, over a read-only chacha DB.

        Args:
            query: Raw user query (escaped for FTS5 downstream).
            top_k: Maximum rows each sub-leg contributes.
            filter_metadata: Optional metadata equality filters.
            include_citations: Whether to build citation-carrying rows.
            source_types: Which of ``note``/``conversation`` to run
                (TASK-14751). ``None`` runs both. An unselected sub-query is
                never issued, and when neither is selected the database is
                not even opened.

        Returns:
            One ranking per non-empty sub-leg (notes, then conversations),
            each best first -- ready for ``interleave_rankings``. An
            unavailable database or a failed sub-query yields fewer (or no)
            rankings and one logged warning; it never raises and never
            affects the media sub-leg.
        """
        selected = (
            _resolve_keyword_source_types(source_types)
            & CHACHA_KEYWORD_SOURCE_TYPES
        )
        if not selected:
            return []

        # Nothing below may raise: `_hybrid_search` gathers this leg with the
        # semantic one without `return_exceptions`, so an escaping exception
        # would fail the whole search rather than degrade one sub-leg.
        try:
            db_path = self._resolve_chachanotes_db_path()
            if db_path is None:
                return []

            loop = asyncio.get_event_loop()
            raw_rows = await loop.run_in_executor(
                None,
                self._chacha_fts_rows,
                db_path,
                query,
                top_k * SEARCH_RESULT_MULTIPLIER,  # Get extra for filtering
                selected,
            )

            rankings: List[Any] = []
            for source_type in (SOURCE_TYPE_NOTE, SOURCE_TYPE_CONVERSATION):
                items = raw_rows.get(source_type) or []
                if not items:
                    continue
                if include_citations:
                    rows = await self._process_keyword_results_with_citations(
                        items, query, filter_metadata, top_k, source_type=source_type
                    )
                else:
                    rows = self._process_keyword_results_basic(
                        items, filter_metadata, top_k, source_type=source_type
                    )
                if rows:
                    rankings.append(rows)
            return rankings
        except Exception as e:
            logger.warning(
                "ChaChaNotes keyword sub-legs failed; the media sub-leg is "
                "unaffected (error_type={})",
                type(e).__name__,
            )
            return []

    def _resolve_chachanotes_db_path(self) -> Optional[Path]:
        """Resolve (and validate) the ChaChaNotes DB path for the FTS leg.

        Mirrors the media sub-leg's treatment exactly: an explicit config
        override wins, otherwise the single authoritative resolver
        (``get_chachanotes_db_path``) decides -- no guessing across
        candidate filenames, and never a create-on-miss (a search must not
        have the side effect of creating a database). The config-sourced
        override is run through ``path_validation``'s traversal/injection
        screen plus lexical normalization before it reaches a filesystem
        check, the same as ``media_db_path``.

        Returns:
            The validated, existing path, or ``None`` (with one logged
            warning naming the reason and the path) when the notes and
            conversation sub-legs must be skipped.
        """
        from tldw_chatbook.Utils.path_validation import validate_path_simple
        from tldw_chatbook.Utils.private_paths import lexical_path

        try:
            from tldw_chatbook.config import get_chachanotes_db_path

            db_path_raw = (
                self.config.search.chachanotes_db_path or get_chachanotes_db_path()
            )
        except Exception as e:
            logger.warning(
                "Could not resolve the ChaChaNotes database path; the notes "
                "and conversation keyword sub-legs return no results "
                "(error_type={})",
                type(e).__name__,
            )
            return None

        try:
            db_path = lexical_path(
                validate_path_simple(
                    Path(str(db_path_raw)).expanduser(),
                    require_exists=False,
                    probe_existing=False,
                )
            )
        except ValueError as e:
            logger.warning(
                "Rejected chachanotes_db_path from config; the notes and "
                "conversation keyword sub-legs return no results "
                "(error_type={})",
                type(e).__name__,
            )
            return None

        # Existence only. Every other filesystem question -- symlinked
        # components, untrusted parent directories, a no-follow open of the
        # file itself -- belongs to the private SQLite seam this leg opens
        # through (see `_connect_chacha_readonly`), exactly as the media
        # sub-leg defers those to `MediaDatabase -> connect_private_sqlite`.
        # This check exists only so the common "no chacha DB yet" case is
        # reported as such instead of as an open failure.
        if not db_path.exists() or not db_path.is_file():
            logger.warning(
                "ChaChaNotes database not found; the notes and conversation "
                "keyword sub-legs return no results (a search never creates "
                "a database)."
            )
            return None

        return db_path

    def _connect_chacha_readonly(self, db_path: Union[str, Path]) -> sqlite3.Connection:
        """Open the ChaChaNotes database read-only, without the ORM.

        Three properties, all deliberate (TASK-3996):

        * **Read-only by construction.** The seam builds a ``mode=ro`` URI,
          so any write raises ``sqlite3.OperationalError`` rather than this
          code being trusted never to issue one.
        * **Not the ORM.** ``CharactersRAGDB``'s constructor runs schema
          creation/migration checks and client registration on open; the
          engine's search path must never do that to the user's main
          database.
        * **The same path guarantees the media sub-leg gets.** This goes
          through ``connect_private_sqlite`` (owner
          ``rag.chachanotes_keyword_leg``, read-only-URI target kind), whose
          ``verify_trusted_directory`` walks EVERY path component with
          ``O_NOFOLLOW`` and opens the file itself no-follow. A hand-rolled
          ``Path.is_symlink()`` check tested only the FINAL component and
          was strictly weaker: review reproduced a symlinked PARENT
          directory being followed here while the media sub-leg refused it.
          The owner preserves the source file's mode, so a read never
          reasserts permissions on a file ``db.chachanotes.primary`` owns.

        Args:
            db_path: An absolute database path (existence already checked).

        Returns:
            A read-only connection with ``sqlite3.Row`` rows. The caller
            owns closing it.

        Raises:
            PrivatePathError / sqlite3.Error / OSError / ValueError: when
            the path or the file fails the seam's checks. Callers degrade
            (warn + no rows); they never let it reach the search.
        """
        from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

        conn = connect_private_sqlite(
            "rag.chachanotes_keyword_leg",
            Path(db_path),
            read_only=True,
        )
        conn.row_factory = sqlite3.Row
        return conn

    def _chacha_fts_rows(
        self,
        db_path: Path,
        query: str,
        limit: int,
        source_types: Optional[Collection[str]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Run the selected ChaChaNotes FTS sub-queries on one connection.

        Args:
            db_path: Validated path to the ChaChaNotes database.
            query: Raw user query.
            limit: Maximum rows per sub-query.
            source_types: Which of ``note``/``conversation`` to query
                (TASK-14751); ``None`` queries both. An unselected sub-query
                is never issued and its key stays empty.

        Returns:
            ``{"note": [...], "conversation": [...]}`` -- an unopenable
            database or a failing sub-query yields empty lists plus one
            logged warning, never an exception.
        """
        rows: Dict[str, List[Dict[str, Any]]] = {
            SOURCE_TYPE_NOTE: [],
            SOURCE_TYPE_CONVERSATION: [],
        }

        selected = (
            _resolve_keyword_source_types(source_types)
            & CHACHA_KEYWORD_SOURCE_TYPES
        )
        if not selected:
            return rows

        escaped_query = self._escape_fts5_query(query)
        if not escaped_query:
            return rows

        if not isinstance(limit, int) or limit < 1:
            limit = DEFAULT_FTS5_LIMIT
        limit = min(limit, MAX_FTS5_LIMIT)

        try:
            conn = self._connect_chacha_readonly(db_path)
        except (sqlite3.Error, ValueError, OSError) as e:
            # `PrivatePathError` is an `OSError`, so a path the seam refuses
            # (symlinked component, untrusted parent) lands here alongside a
            # genuinely unopenable file -- both degrade this sub-leg only.
            logger.warning(
                "Could not open the ChaChaNotes database read-only; the notes "
                "and conversation keyword sub-legs return no results "
                "(error_type={})",
                type(e).__name__,
            )
            return rows

        with closing(conn):
            if SOURCE_TYPE_NOTE in selected:
                rows[SOURCE_TYPE_NOTE] = self._chacha_notes_fts(
                    conn, escaped_query, limit
                )
            if SOURCE_TYPE_CONVERSATION in selected:
                rows[SOURCE_TYPE_CONVERSATION] = self._chacha_conversations_fts(
                    conn, escaped_query, limit
                )
        return rows

    @staticmethod
    def _chacha_notes_fts(
        conn: sqlite3.Connection, escaped_query: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Notes sub-query: mirrors ``CharactersRAGDB.search_notes``.

        Same FTS table and join (``notes_fts`` is an external-content table
        over ``notes``, joined on ``rowid``), same ``rank`` ordering, and the
        same ``notes.deleted = 0`` filter. That filter reads as redundant --
        the soft-delete trigger evicts the row from the index -- and is not:
        an external-content ``'rebuild'`` re-indexes the content table,
        deleted rows included, and this predicate is then the only thing
        keeping a deleted note out of search results (pinned by
        ``test_deleted_notes_and_conversations_are_excluded``, which rebuilds
        both indexes; without the rebuild, dropping this line changed
        nothing).

        Args:
            conn: Read-only ChaChaNotes connection.
            escaped_query: A per-token-quoted FTS5 MATCH expression.
            limit: Maximum rows.

        Returns:
            Row dicts (``id``/``title``/``content``), best match first.
        """
        sql = """
        SELECT
            main.id AS id,
            main.title AS title,
            main.content AS content
        FROM notes_fts fts
        JOIN notes main ON fts.rowid = main.rowid
        WHERE fts.notes_fts MATCH ?
          AND main.deleted = 0
        ORDER BY rank
        LIMIT ?
        """
        try:
            with closing(conn.execute(sql, (escaped_query, limit))) as cursor:
                return [
                    {
                        "id": row["id"],
                        "title": row["title"] or f"Note {row['id']}",
                        "content": row["content"] or "",
                    }
                    for row in cursor
                ]
        except sqlite3.Error as e:
            logger.warning(
                "Notes keyword sub-leg failed; returning no note rows (error_type={})",
                type(e).__name__,
            )
            return []

    @staticmethod
    def _chacha_conversations_fts(
        conn: sqlite3.Connection, escaped_query: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Conversations sub-query: mirrors
        ``CharactersRAGDB.search_conversations_by_content``.

        Message content is what is indexed (``messages_fts``), but the unit
        of retrieval is the CONVERSATION -- one row per conversation at its
        best matching message's rank (``GROUP BY c.id``, ``MIN(rank)``,
        ``ORDER BY best_rank``), matching both the ORM's convention and the
        ``conversation`` document the vector leg indexes (whose ``source_id``
        is the conversation id -- the two legs must agree for fusion to
        merge them).

        Both soft-delete filters the ORM applies are replicated:
        ``messages.deleted = 0`` AND ``conversations.deleted = 0``. The
        conversations one is load-bearing at all times -- deleting a
        conversation does not soft-delete its messages, so without it a
        deleted conversation keeps matching through its surviving messages.
        The messages one matters after an index ``'rebuild'`` (see
        ``_chacha_notes_fts``), which re-admits soft-deleted messages.

        The document text is the matched messages rendered as
        ``sender: content`` lines, the same shape
        ``ingestion_indexing.conversation_document`` indexes (restricted to
        the matching messages, which is what the user searched for) -- and
        in the same order. That ordering is why this is two statements
        rather than one ``group_concat``: SQLite defines NO order for the
        rows an aggregate consumes, so the concatenation order was whatever
        the query plan produced (in practice storage order, i.e. insertion
        order), and every snippet or span built on that text inherited the
        plan's whim. Selecting the matching lines in the ORM's own order
        (``get_messages_for_conversation``: ``timestamp ASC``, with the
        rowid as a deterministic tie-break for equal timestamps) and
        joining them in Python makes the order a property of the query,
        not of the planner.

        Args:
            conn: Read-only ChaChaNotes connection.
            escaped_query: A per-token-quoted FTS5 MATCH expression.
            limit: Maximum conversations.

        Returns:
            Row dicts (``id``/``title``/``content``), best match first.
        """
        conversations_sql = """
        SELECT
            c.id AS id,
            c.title AS title,
            MIN(rank) AS best_rank
        FROM messages_fts fts
        JOIN messages m ON fts.rowid = m.rowid
        JOIN conversations c ON m.conversation_id = c.id
        WHERE fts.messages_fts MATCH ?
          AND m.deleted = 0
          AND c.deleted = 0
        GROUP BY c.id
        ORDER BY best_rank
        LIMIT ?
        """
        try:
            with closing(
                conn.execute(conversations_sql, (escaped_query, limit))
            ) as cursor:
                conversations = [
                    {
                        "id": row["id"],
                        "title": row["title"] or f"Conversation {row['id']}",
                    }
                    for row in cursor
                ]
            if not conversations:
                return []

            # Only "?" characters are interpolated; every value is bound.
            placeholders = ",".join("?" * len(conversations))
            messages_sql = f"""
            SELECT
                m.conversation_id AS conversation_id,
                COALESCE(m.sender, 'unknown') || ': '
                    || COALESCE(m.content, '') AS line
            FROM messages_fts fts
            JOIN messages m ON fts.rowid = m.rowid
            WHERE fts.messages_fts MATCH ?
              AND m.deleted = 0
              AND m.conversation_id IN ({placeholders})
            ORDER BY m.timestamp ASC, m.rowid ASC
            """
            lines: Dict[Any, List[str]] = {}
            params = [escaped_query, *(row["id"] for row in conversations)]
            with closing(conn.execute(messages_sql, params)) as cursor:
                for row in cursor:
                    lines.setdefault(row["conversation_id"], []).append(row["line"])

            return [
                {
                    "id": row["id"],
                    "title": row["title"],
                    "content": "\n".join(lines.get(row["id"], ())),
                }
                for row in conversations
            ]
        except sqlite3.Error as e:
            logger.warning(
                "Conversations keyword sub-leg failed; returning no conversation "
                "rows (error_type={})",
                type(e).__name__,
            )
            return []

    async def _hybrid_search(
        self,
        query: str,
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]] = None,
        include_citations: bool = True,
        score_threshold: float = 0.0,
        *,
        keyword_source_types: Optional[Collection[str]] = None,
    ) -> Union[List[SearchResult], List[SearchResultWithCitations]]:
        """
        Perform hybrid search combining semantic (vector) and keyword (FTS5) legs.

        The legs run in parallel and are fused with Reciprocal Rank Fusion
        (default k=5, measured for this window -- see
        ``config.DEFAULT_HYBRID_RRF_K``; the server's 60 starves FTS-only
        rows here) plus an alpha-weighted blend of the per-leg RRF
        scores, matching the tldw_server reference design. Alpha comes from
        ``config.search.hybrid_alpha`` (0 = FTS only, 1 = vector only) and k
        from ``config.search.rrf_k``. Each leg over-fetches
        ``top_k * config.search.hybrid_pool_multiplier`` candidates before
        fusion narrows back down to ``top_k``. Citations are combined when
        the same chunk appears in both legs.

        ``keyword_source_types`` (TASK-14751) narrows the FTS leg to the
        types the caller will keep, so its budget is not spent on rows a
        downstream source-type filter would discard; ``None`` leaves the leg
        exactly as it was. It does not scope the semantic leg -- that is
        ``metadata_allowlist``'s job, and it is semantic-only.
        """
        # Get results from both search types. The pool multiplier widens
        # ONLY these two leg fetches -- `_semantic_search`'s own internal
        # over-fetch (its raw vector-store call) still uses the module
        # SEARCH_RESULT_MULTIPLIER, on this path and on the direct
        # semantic-search path alike.
        pool_multiplier = _resolve_hybrid_pool_multiplier(
            self.config.search.hybrid_pool_multiplier
        )
        semantic_task = self._semantic_search(
            query,
            top_k * pool_multiplier,
            filter_metadata,
            include_citations,
            score_threshold,
        )
        keyword_task = self._keyword_search(
            query,
            top_k * pool_multiplier,
            filter_metadata,
            include_citations,
            keyword_source_types=keyword_source_types,
        )

        # Run both searches in parallel
        semantic_results, keyword_results = await asyncio.gather(
            semantic_task, keyword_task
        )

        return self._fuse_hybrid_results(
            keyword_results=keyword_results,
            semantic_results=semantic_results,
            top_k=top_k,
            alpha=self.config.search.hybrid_alpha,
            rrf_k=self.config.search.rrf_k,
            include_citations=include_citations,
        )

    @staticmethod
    def _fuse_hybrid_results(
        keyword_results: Union[List[SearchResult], List[SearchResultWithCitations]],
        semantic_results: Union[List[SearchResult], List[SearchResultWithCitations]],
        top_k: int,
        alpha: float,
        rrf_k: int = DEFAULT_RRF_K,
        include_citations: bool = True,
    ) -> Union[List[SearchResult], List[SearchResultWithCitations]]:
        """Fuse the FTS (keyword) and vector (semantic) legs via RRF + alpha.

        Rank-based fusion (server parity): each leg's returned ordering is
        the ranking; the fused score replaces the leg scores. Leg provenance
        (per-leg rank and RRF contribution) is stored in
        ``metadata['hybrid_fusion']``.

        Legs are matched on DOCUMENT identity (``_fusion_doc_key``), not on
        row id: the FTS leg ranks documents and the vector leg ranks chunks,
        so an id match was impossible (TASK-3994). Consequences worth
        knowing: several chunks of one document collapse into a single fused
        row at that document's best vector rank (freeing top-k slots the
        keyword leg can now reach), a merged row displays the matched CHUNK,
        and a merged row's citations are the union of both legs'.

        Args:
            keyword_results: FTS/keyword leg, best first.
            semantic_results: Vector/semantic leg, best first.
            top_k: Maximum number of fused results to return.
            alpha: Vector-leg weight (0 = FTS only, 1 = vector only).
                Validated via resolve_hybrid_alpha: out-of-range/invalid
                config values fall back to the 0.7 default, matching the
                pipeline path.
            rrf_k: RRF constant. The live caller (`_hybrid_search`) passes
                `config.search.rrf_k`, whose shipped default is 5
                (`config.DEFAULT_HYBRID_RRF_K`, TASK-4110). Validated via
                fusion.resolve_rrf_k, the same use-time pattern as `alpha`:
                out-of-range/invalid config values fall back to that
                shipped default (5) with a warning rather than distorting
                or crashing the fusion math -- every fallback in the
                app-config resolver is the shipped value. The SIGNATURE
                default here stays DEFAULT_RRF_K (60, server parity) for
                every caller that predates this parameter: a pure no-config
                fallback, not the shipped default, and never reached by the
                live caller.
            include_citations: Whether results carry citations to merge.

        Returns:
            Fused results sorted by fused score descending. The recorded
            `metadata['hybrid_fusion']['rrf_k']` is always this resolved
            value (never a literal) -- `local_citation_capture._reliable_rrf`
            re-derives the fused score from it to certify the row as RRF, so
            a metadata block that ever drifted from the value actually used
            in the math would silently degrade every hybrid row to LEGACY.
        """
        # config.search.hybrid_alpha is not range-checked at load time
        # (RAGConfig.validate() has no callers); resolve here so this path
        # gets the same fallback semantics as the pipeline merge. Same for
        # rrf_k via resolve_rrf_k.
        alpha = resolve_hybrid_alpha(alpha)
        rrf_k = resolve_rrf_k(rrf_k)
        fused = reciprocal_rank_fusion(
            keyword_results,
            semantic_results,
            key=_fusion_doc_key,
            alpha=alpha,
            rrf_k=rrf_k,
            max_results=top_k,
        )

        results = []
        for entry in fused:
            # Display preference (TASK-3994): the VECTOR leg's item, i.e. the
            # matched chunk, not the whole-document FTS row. Deliberately not
            # `entry.item`, which prefers the FTS leg for server parity and
            # has its own consumer -- the choice is made here, at the call
            # site. A merged row now shows the passage that actually matched,
            # keeps the vector leg's real similarity for score banding, and
            # carries the chunk metadata (`source_id`, `chunk_id`) that the
            # downstream row mappers read.
            result = entry.vector_item if entry.vector_item is not None else entry.fts_item
            # `result` aliases one of the two leg items, so both legs'
            # original scores must be read *before* result.score is
            # overwritten below, or the in-place mutation clobbers the very
            # value we're trying to preserve (it is now the vector leg's
            # score that would be lost, previously the FTS leg's).
            fts_score = entry.fts_item.score if entry.fts_item is not None else None
            vector_score = entry.vector_item.score if entry.vector_item is not None else None
            # Combine citations when the same document surfaced in both legs.
            # Read defensively: only the displayed item is guaranteed to be a
            # citation-carrying shape, and the two legs can disagree (a
            # citation-less leg must not raise AttributeError here).
            if (
                include_citations
                and entry.fts_item is not None
                and entry.vector_item is not None
                and hasattr(result, "citations")
            ):
                result.citations = merge_citations(
                    [
                        getattr(entry.fts_item, "citations", None) or [],
                        getattr(entry.vector_item, "citations", None) or [],
                    ]
                )
            result.score = entry.score
            result.metadata = {
                **(result.metadata or {}),
                "hybrid_fusion": {
                    **entry.provenance(),
                    "fts_score": fts_score,
                    "vector_score": vector_score,
                    "alpha": alpha,
                    "rrf_k": rrf_k,
                },
            }
            results.append(result)
        return results

    # === Helper Methods ===

    def _get_embedding_dimension(self) -> int:
        """Get the embedding dimension from the model."""
        try:
            dim = self.embeddings.get_embedding_dimension()
            if dim is None:
                # Default if we can't determine
                logger.warning(
                    f"Could not determine embedding dimension, defaulting to {DEFAULT_EMBEDDING_DIM}"
                )
                return DEFAULT_EMBEDDING_DIM
            return dim
        except Exception as e:
            logger.warning(
                f"Error getting embedding dimension: {e}, defaulting to {DEFAULT_EMBEDDING_DIM}"
            )
            return DEFAULT_EMBEDDING_DIM

    @timeit("rag_chunking_operation")
    async def _chunk_document(
        self, content: str, chunk_size: int, chunk_overlap: int, method: str
    ) -> List[Dict[str, Any]]:
        """Chunk document asynchronously."""
        logger.info(
            f"Chunking document with method={method}, size={chunk_size}, overlap={chunk_overlap}"
        )
        log_histogram("rag_chunk_size_config", chunk_size)
        log_histogram("rag_chunk_overlap_config", chunk_overlap)
        log_counter("rag_chunking_method", labels={"method": method})

        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(
            self._executor,  # Use dedicated thread pool
            self.chunking.chunk_text,
            content,
            chunk_size,
            chunk_overlap,
            method,
        )

        # Log chunk statistics
        if chunks:
            chunk_lengths = [len(chunk.get("text", "")) for chunk in chunks]
            avg_chunk_length = sum(chunk_lengths) / len(chunk_lengths)
            log_histogram("rag_avg_chunk_length", avg_chunk_length)
            log_histogram("rag_min_chunk_length", min(chunk_lengths))
            log_histogram("rag_max_chunk_length", max(chunk_lengths))
            logger.debug(
                f"Created {len(chunks)} chunks, avg length: {avg_chunk_length:.0f} chars"
            )

        return chunks

    async def _store_chunks(
        self,
        ids: List[str],
        embeddings: Union["np.ndarray", List[List[float]]],
        documents: List[str],
        metadata: List[dict],
    ) -> None:
        """Store chunks in vector database asynchronously."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self.vector_store.add, ids, embeddings, documents, metadata
        )

    # === Management Methods ===

    @staticmethod
    def _keyword_row_metadata(
        item: Dict, content: str, source_type: str
    ) -> Dict[str, Any]:
        """Build one keyword-leg row's metadata, for any sub-leg.

        Args:
            item: The raw sub-leg row (media / note / conversation).
            content: The already size-limited document text.
            source_type: The sub-leg's ``SOURCE_TYPE_*`` value. Stamped
                verbatim -- ``_fusion_doc_key`` compares these raw strings
                against what ``ingestion_indexing`` stamps on the vector
                side, so the singular vocabulary is load-bearing.

        Returns:
            The row metadata. ``source_id`` is the BARE row id, which is
            both what ``_fusion_doc_key`` prefers and what the vector leg
            carries (spread from the ingestion document into every chunk);
            ``doc_id`` keeps the same value for the media leg's existing
            consumers.
        """
        return {
            "doc_id": str(item["id"]),
            # The keyword leg used to stamp only `doc_id`, leaving
            # `_semantic_row` to fall back to the PREFIXED row id
            # (`media_15`) as the row's source id -- unlike every vector-leg
            # row, whose `source_id` is the bare id. Stamping it here makes
            # the two legs agree for fusion and gives the Library's "open
            # source" action an id it can actually resolve.
            "source_id": str(item["id"]),
            "doc_title": item.get("title", "Untitled"),
            # The display key. The vector leg gets `title` for free
            # (the indexing call spreads the document's own metadata
            # into every chunk); this leg builds its metadata from
            # scratch, so without this a keyword-leg row reaches the
            # Library evidence list as "Untitled source" -- observed
            # live under Hybrid Full with an empty semantic leg.
            # `_semantic_row` (library_local_rag_search_service) reads
            # `title`/`document_title` and never `doc_title`.
            "title": item.get("title") or "",
            "media_type": item.get("type"),
            "url": item.get("url"),
            "author": item.get("author"),
            "ingestion_date": item.get("ingestion_date"),
            "text_preview": content[:200],
            "source_type": source_type,
            "source": source_type,
        }

    def _process_keyword_results_basic(
        self,
        search_results: List[Dict],
        filter_metadata: Optional[Dict[str, Any]],
        top_k: int,
        source_type: str = SOURCE_TYPE_MEDIA,
    ) -> List[SearchResult]:
        """Process keyword search results without citations.

        Args:
            search_results: Raw sub-leg rows, best first.
            filter_metadata: Optional metadata equality filters.
            top_k: Maximum rows to return.
            source_type: Which sub-leg produced these rows.
        """
        results = []

        for item in search_results:
            # Apply metadata filters if provided
            if filter_metadata:
                item_meta = {
                    "media_type": item.get("type"),
                    "source": source_type,
                    "source_type": source_type,
                    "author": item.get("author"),
                }
                if not all(
                    item_meta.get(k) == v
                    for k, v in filter_metadata.items()
                    if k in item_meta
                ):
                    continue

            # Create base SearchResult
            content = item.get("content", "")[:1000]  # Limit content size

            base_result = SearchResult(
                id=f"{source_type}_{item['id']}",
                score=KEYWORD_SEARCH_SCORE,  # FTS5 doesn't provide normalized scores
                document=content,
                metadata=self._keyword_row_metadata(item, content, source_type),
            )
            results.append(base_result)

            if len(results) >= top_k:
                break

        return results

    async def _process_keyword_results_with_citations(
        self,
        search_results: List[Dict],
        query: str,
        filter_metadata: Optional[Dict[str, Any]],
        top_k: int,
        source_type: str = SOURCE_TYPE_MEDIA,
    ) -> List[SearchResultWithCitations]:
        """Process keyword search results with citations - batch processing for efficiency.

        Args:
            search_results: Raw sub-leg rows, best first.
            query: Raw user query (used to locate citation spans).
            filter_metadata: Optional metadata equality filters.
            top_k: Maximum rows to return.
            source_type: Which sub-leg produced these rows.
        """
        import asyncio

        results = []

        # Process in batches for efficiency
        batch_size = KEYWORD_BATCH_SIZE

        for i in range(0, len(search_results), batch_size):
            batch = search_results[i : i + batch_size]

            # Process batch concurrently
            batch_results = await asyncio.gather(
                *[
                    self._create_keyword_result_with_citations(
                        item, query, filter_metadata, source_type=source_type
                    )
                    for item in batch
                ]
            )

            # Filter out None results and add to results
            for result in batch_results:
                if result is not None:
                    results.append(result)
                    if len(results) >= top_k:
                        return results

        return results

    async def _create_keyword_result_with_citations(
        self,
        item: Dict,
        query: str,
        filter_metadata: Optional[Dict[str, Any]],
        source_type: str = SOURCE_TYPE_MEDIA,
    ) -> Optional[SearchResultWithCitations]:
        """Create a single keyword result with citations.

        Citation spans come from ``_keyword_citation_spans``, which reads
        the SAME token list ``_escape_fts5_query`` built this search's MATCH
        expression from. Locating the raw query as one contiguous substring
        instead (what this did before) assumed phrase semantics the keyword
        leg no longer has, so every multi-token hit whose tokens are
        scattered lost its citations entirely.

        Args:
            item: One raw sub-leg row.
            query: Raw user query (used to locate citation spans).
            filter_metadata: Optional metadata equality filters.
            source_type: Which sub-leg produced this row.

        Returns:
            The row, or ``None`` when the metadata filters exclude it.
        """
        # Apply metadata filters
        if filter_metadata:
            item_meta = {
                "media_type": item.get("type"),
                "source": source_type,
                "source_type": source_type,
                "author": item.get("author"),
            }
            if not all(
                item_meta.get(k) == v
                for k, v in filter_metadata.items()
                if k in item_meta
            ):
                return None

        # Create base result
        content = item.get("content", "")[:1000]
        base_metadata = self._keyword_row_metadata(item, content, source_type)

        # Find citations from the query's own tokens (see the docstring).
        full_content = item.get("content", "")
        tokens = self._fts5_query_tokens(query)
        spans = self._keyword_citation_spans(full_content, tokens)

        # Cap the spans, preferring coverage: a span evidencing a token no
        # earlier span did comes first, so a two-token query whose first
        # token repeats does not spend every slot on that one token. Ties
        # and leftover slots fall back to document order, which is what the
        # single-token case (and the old whole-query lookup) produced.
        selected: List[Tuple[int, int, frozenset]] = []
        covered_tokens: set = set()
        for span in spans:
            if len(selected) >= MAX_CITATION_MATCHES:
                break
            if span[2] - covered_tokens:
                selected.append(span)
                covered_tokens |= span[2]
        for span in spans:
            if len(selected) >= MAX_CITATION_MATCHES:
                break
            if span not in selected:
                selected.append(span)
        selected.sort()

        citations = []
        all_token_indices = frozenset(range(len(tokens)))
        for start, end, span_tokens in selected:
            start_context = max(0, start - CITATION_CONTEXT_CHARS)
            end_context = min(len(full_content), end + CITATION_CONTEXT_CHARS)
            whole_query = span_tokens == all_token_indices

            citation = Citation(
                document_id=str(item["id"]),
                document_title=item.get("title", "Untitled"),
                chunk_id=f"{source_type}_{item['id']}_kw_{start}",
                text=full_content[start_context:end_context],
                start_char=start,
                end_char=end,
                # A span covering every query token is the exact match the
                # pre-implicit-AND builder used to emit; a span covering
                # only some of them is real but weaker evidence.
                confidence=1.0 if whole_query else PARTIAL_CITATION_CONFIDENCE,
                match_type=CitationType.EXACT if whole_query else CitationType.KEYWORD,
                metadata={
                    "query": query,
                    "match_text": full_content[start:end],
                    "media_type": item.get("type"),
                },
            )
            citations.append(citation)

        # No span at all -- the row still matched FTS5, on an indexed column
        # this content does not carry (`media_fts` indexes the title too),
        # so fall back to a document-level citation rather than hand the
        # evidence list a keyword-backed row with nothing to show.
        if not citations and full_content:
            citation = Citation(
                document_id=str(item["id"]),
                document_title=item.get("title", "Untitled"),
                chunk_id=f"{source_type}_{item['id']}_general",
                text=content,
                start_char=0,
                end_char=len(content),
                confidence=0.7,
                match_type=CitationType.KEYWORD,
                metadata={"query": query, "media_type": item.get("type")},
            )
            citations.append(citation)

        return SearchResultWithCitations(
            id=f"{source_type}_{item['id']}",
            score=KEYWORD_SEARCH_SCORE,
            document=content,
            metadata=base_metadata,
            citations=citations,
        )

    @staticmethod
    def _fts5_query_tokens(query: str) -> List[str]:
        """Tokenize a raw user query -- the ONE tokenization of the keyword leg.

        Two consumers must agree on this list or the leg contradicts itself:
        ``_escape_fts5_query`` builds the FTS5 MATCH expression from it, and
        ``_keyword_citation_spans`` locates the citation spans from it. They
        used to tokenize independently (per-token quoting on one side, a raw
        whole-query substring lookup on the other), which is exactly how a
        row could match the query and then be reported with no evidence for
        it -- see ``_keyword_citation_spans``.

        Tokens are whitespace-separated runs that contain at least one
        alphanumeric character. FTS5's default tokenizer indexes only
        alphanumeric runs, so a pure-punctuation token ("!!!") can never
        match anything and is dropped rather than carried as a no-op.

        Args:
            query: Raw search query.

        Returns:
            The query's searchable tokens, in query order; empty when the
            query is empty, whitespace-only or all punctuation.
        """
        if not query:
            return []

        # Bound total processing length before tokenizing (DoS guard). The
        # warning belongs to `_escape_fts5_query`, which runs once per
        # search; this helper also runs once per RESULT ROW.
        query = query[:MAX_QUERY_LENGTH]

        return [
            token for token in query.split() if any(ch.isalnum() for ch in token)
        ]

    def _escape_fts5_query(self, query: str) -> str:
        """
        Build a safe FTS5 MATCH expression for a raw user query.

        TASK-3995: wrapping the *entire* query in one pair of double quotes
        (the previous approach) makes FTS5 treat it as a phrase query,
        which requires every token to appear as one contiguous run in the
        document. That is strictly stronger than AND-of-terms, not
        equivalent to it -- a document containing all the query's tokens
        but not adjacent to each other never matches. Verified directly
        against a real corpus document (task-3995's description): the
        phrase form of a multi-token query matched 0 rows against text
        that plainly contained the relevant terms, just not contiguously.

        This implementation quotes each token individually (doubling any
        embedded quote characters) and joins the quoted tokens with a
        single space, which FTS5 interprets as an implicit AND: every
        token must appear somewhere in the document, in any order, at any
        distance. This keeps the safety property that made whole-query
        phrase-quoting attractive in the first place -- a bare token FTS5
        would otherwise parse as column-filter or operator syntax (e.g.
        the hyphenated-numeric token "Obsidian-3", which raises
        `OperationalError('no such column: 3')` unquoted) is safe once
        quoted, because FTS5 treats a quoted token as a literal string
        with no operator semantics. A single-token query degenerates to
        the exact same MATCH expression as before (one quoted token), so
        single-token search behavior is unchanged.

        Tokenization (including the pure-punctuation drop and the length
        guard) lives in ``_fts5_query_tokens``, shared with the citation
        builder. If every token is dropped, the result is "" -- callers
        must treat "" as "no results" and skip the FTS5 query entirely
        rather than run a MATCH expression that can only ever match
        nothing.

        Args:
            query: Raw search query

        Returns:
            A safe, per-token-quoted FTS5 MATCH expression, or "" if the
            query has no FTS5-searchable tokens (empty, whitespace-only,
            or all punctuation).
        """
        if query and len(query) > MAX_QUERY_LENGTH:
            logger.warning(
                f"Query truncated from {len(query)} to {MAX_QUERY_LENGTH} characters"
            )

        quoted_tokens = [
            '"{}"'.format(token.replace('"', '""'))
            for token in self._fts5_query_tokens(query)
        ]

        # FTS5 joins space-separated quoted terms with an implicit AND.
        return " ".join(quoted_tokens)

    @staticmethod
    def _keyword_citation_spans(
        content: str, tokens: List[str]
    ) -> List[Tuple[int, int, frozenset]]:
        """Locate the citation spans for a keyword hit, from the SAME tokens.

        TASK-3996 follow-up (Qodo, PR #1469). Before TASK-3995 the keyword
        leg used phrase semantics, so a hit guaranteed the raw query was one
        contiguous substring of the document and the citation builder could
        just look that raw query up. Per-token implicit AND deleted that
        guarantee: documents now match with the tokens scattered, the raw
        lookup found nothing, and the rows the fix had just made reachable
        came back with ``citations=[]``.

        Spans are located per token, case-insensitively, from the token list
        ``_escape_fts5_query`` built the MATCH expression from. A token is
        matched as its alphanumeric runs separated by non-alphanumerics
        ("Obsidian-3" -> ``Obsidian`` then ``3``), which is how FTS5 reads a
        quoted token: a phrase over the runs, adjacency required.

        Spans that overlap, or that are separated only by non-alphanumeric
        characters, are merged -- so a query whose tokens ARE contiguous in
        the document ("spindle runout") still yields the single whole-phrase
        span the pre-TASK-3995 lookup produced, rather than one citation per
        token.

        Args:
            content: The document text the offsets must index.
            tokens: ``_fts5_query_tokens(query)`` for the same query.

        Returns:
            Merged, non-overlapping ``(start, end, token_indices)`` spans in
            document order, where ``token_indices`` is the set of ``tokens``
            positions the span evidences. Empty when no token appears in the
            content -- a real case, since a row can match on an indexed
            column the caller never sees (``media_fts`` indexes the title
            too).
        """
        if not content or not tokens:
            return []

        raw_spans: List[Tuple[int, int, int]] = []
        for index, token in enumerate(tokens):
            runs = re.findall(r"[^\W_]+", token, re.UNICODE)
            if not runs:
                continue
            # Runs separated by any non-alphanumeric run: FTS5 treats a
            # quoted token as a phrase over exactly these runs.
            pattern = r"[\W_]+".join(re.escape(run) for run in runs)
            raw_spans.extend(
                (match.start(), match.end(), index)
                for match in re.finditer(pattern, content, re.IGNORECASE)
            )

        if not raw_spans:
            return []

        merged: List[Tuple[int, int, set]] = []
        for start, end, index in sorted(raw_spans):
            if merged:
                previous_start, previous_end, covered = merged[-1]
                gap = content[previous_end:start] if start > previous_end else ""
                if start <= previous_end or (
                    len(gap) <= CITATION_SPAN_MERGE_GAP_CHARS
                    and not any(ch.isalnum() for ch in gap)
                ):
                    # Overlapping, or separated only by a little
                    # punctuation/whitespace: one piece of evidence.
                    covered.add(index)
                    merged[-1] = (
                        previous_start,
                        max(previous_end, end),
                        covered,
                    )
                    continue
            merged.append((start, end, {index}))

        return [(start, end, frozenset(covered)) for start, end, covered in merged]

    def _perform_fts5_search(
        self, pool, query: str, limit: int
    ) -> List[Dict[str, Any]]:
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
            limit = DEFAULT_FTS5_LIMIT  # Safe default
        limit = min(limit, MAX_FTS5_LIMIT)  # Cap maximum results

        # Note: `Media` has no `tags` column and `media_fts` only indexes
        # (title, content) -- see Client_Media_DB_v2's `_FTS_TABLES_SQL` and
        # its `Media` CREATE TABLE. An earlier version of this query selected
        # a nonexistent `m.tags`, which raised `OperationalError: no such
        # column: m.tags` on every real DB (silently swallowed by the outer
        # exception handler, so keyword search always returned []).
        # Review finding (Task 3 PR): this used to SELECT `-rank as rank` and
        # `ORDER BY rank` -- ascending on the NEGATED alias, i.e.
        # worst-match-first (fts5's raw `rank` column is smaller/more
        # negative = better; negating it and sorting ascending flips that).
        # Order on the raw `media_fts.rank` column instead -- ascending is
        # already best-match-first, the canonical fts5 usage. `rank` is not
        # read from the result rows anywhere downstream (keyword results use
        # a fixed KEYWORD_SEARCH_SCORE, not a rank-derived score), so it does
        # not need to be selected at all, only ordered on.
        sql = """
        SELECT
            m.id,
            m.title,
            m.content,
            m.url,
            m.type,
            m.author,
            m.ingestion_date
        FROM Media m
        JOIN media_fts ON m.id = media_fts.rowid
        WHERE media_fts MATCH ?
        AND m.is_trash = 0
        ORDER BY media_fts.rank
        LIMIT ?
        """

        results = []
        try:
            # Use transaction for consistent read
            with pool.transaction() as conn:
                cursor = conn.cursor()
                # Use parameterized query - the escaped_query is already safe
                cursor.execute(sql, (escaped_query, limit))

                for row in cursor:
                    results.append(
                        {
                            "id": row["id"],
                            "title": row["title"],
                            "content": row["content"],
                            "url": row["url"],
                            "type": row["type"],
                            "author": row["author"],
                            "ingestion_date": row["ingestion_date"],
                        }
                    )
        except Exception as e:
            logger.error(f"FTS5 search failed for query '{query}': {e}")
            # Re-raise with more context
            raise RuntimeError(f"Database search failed: {str(e)}") from e

        return results

    def _extract_context_from_results(
        self,
        results: List[Union[SearchResult, SearchResultWithCitations]],
        max_length: int = 10000,
    ) -> str:
        """Extract a context string from search results for caching."""
        context_parts = []
        total_chars = 0

        for result in results:
            # Format result
            title = result.metadata.get("title", "Untitled")
            source = result.metadata.get("source", "unknown")

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
                "last_index_time": self._last_index_time,
            },
            "config": {
                "embedding_model": self.config.embedding_model,
                "vector_store_type": self.config.vector_store_type,
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "default_top_k": self.config.default_top_k,
            },
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
            # Shutdown thread pool executor
            if hasattr(self, "_executor"):
                self._executor.shutdown(wait=True, cancel_futures=True)
                logger.info("Shut down thread pool executor")

            # Close embeddings service
            self.embeddings.close()

            # Release vector-store clients, including persistent SQLite handles.
            self.vector_store.close()

            # Close all database connection pools
            from .db_connection_pool import close_all_pools

            close_all_pools()

            # Clear cache
            if hasattr(self, "cache"):
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
    show_progress: bool = True,
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


# NOTE (task-655): a `create_rag_service(embedding_model=None, vector_store=
# "chroma", persist_dir=None, **kwargs)` convenience function used to live
# here. `simplified/__init__.py` never imported it -- it only imports the
# same-named `rag_factory.create_rag_service(profile_name="hybrid_basic",
# ...)`, which is what the public seam and every real caller (production and
# tests) actually reach. The version in this module was therefore dead code
# with a different, misleading signature; it was removed rather than kept as
# an unreachable duplicate. See Tests/RAG/simplified/test_create_rag_service_seam.py.
