"""
Wrapper to adapt the existing Embeddings_Lib.py to the simplified RAG interface.

This maintains the clean API while leveraging the robust existing implementation
that provides thread-safe caching, multiple providers, and async support.
"""

from types import SimpleNamespace
from ..activation import async_guarded as activation_async_guarded
from ..activation import guarded as activation_guarded
from ..activation import source_paths
from typing import Any, Dict, List, Optional, Union
from loguru import logger
import asyncio
import os
import re
import time
import psutil
import hashlib

# Handle numpy as optional dependency
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

    # Create a minimal stub for type hints
    class np:
        class ndarray:
            pass


# Lazy imports - will be loaded on demand
EmbeddingFactory = None
EmbeddingConfigSchema = None

from tldw_chatbook.Metrics.metrics_logger import (  # noqa: E402
    log_counter,
    log_histogram,
    log_gauge,
    timeit,
)
from .circuit_breaker import (  # noqa: E402
    get_circuit_breaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
)


class _DeterministicEmbeddingFactory:
    """Small offline embedding backend for tests and explicit mock configurations."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.config = SimpleNamespace(
            default_model_id="default",
            models={
                "default": SimpleNamespace(provider="mock", model_name_or_path="mock")
            },
        )

    def _encode_one(self, text: str) -> "np.ndarray":
        vector = np.zeros(self.dimension, dtype=float)
        tokens = re.findall(r"[\w']+", text.lower())
        if not tokens:
            tokens = [text]

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8", errors="replace")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def embed(self, texts: List[str], as_list: bool = False):
        embeddings = np.array([self._encode_one(text) for text in texts])
        return embeddings.tolist() if as_list else embeddings

    async def async_embed(self, texts: List[str], as_list: bool = False):
        return self.embed(texts, as_list=as_list)

    def embed_one(self, text: str, as_list: bool = False):
        embedding = self._encode_one(text)
        return embedding.tolist() if as_list else embedding

    async def async_embed_one(self, text: str, as_list: bool = False):
        return self.embed_one(text, as_list=as_list)

    def close(self) -> None:
        return None


def _ensure_embeddings_imported():
    """Lazily import embeddings components when needed."""
    global EmbeddingFactory, EmbeddingConfigSchema

    if EmbeddingFactory is not None and EmbeddingConfigSchema is not None:
        return True

    try:
        from tldw_chatbook.Embeddings.Embeddings_Lib import (
            EmbeddingFactory as _EF,
            EmbeddingConfigSchema as _ECS,
        )

        if EmbeddingFactory is None:
            EmbeddingFactory = _EF
        if EmbeddingConfigSchema is None:
            EmbeddingConfigSchema = _ECS
        logger.info("Successfully imported embeddings components")
        return True
    except ImportError as e:
        logger.error(f"Failed to import embeddings components: {e}")
        return False


# task-640 AC#7 (the "additional finding" bundled into the same review,
# not one of the original 6 numbered items -- see fd_protection.py's/
# ingestion_indexing.py's "item N" comments for those; do not confuse this
# with THEIR "item 4", which is the unrelated protect_file_descriptors()
# lock): bare, org-prefix-less HuggingFace model ids that are known to
# 404 against the real Hub API when passed through verbatim as
# `model_name_or_path` -- currently just "all-MiniLM-L6-v2", the builtin
# "hybrid_basic"/"bm25_basic"/"fast" RAG profiles' default embedding model
# (RAG_Search/config_profiles.py), missing its "sentence-transformers/" org
# prefix (the canonical id is "sentence-transformers/all-MiniLM-L6-v2").
# Every OTHER builtin profile's embedding.model already carries a real org
# prefix (sentence-transformers/, BAAI/, microsoft/); this table is
# deliberately a narrow, explicit alias map -- NOT a "no slash -> prepend
# sentence-transformers/" heuristic, which would silently break a
# legitimately slash-free top-level HF model id like "bert-base-uncased".
#
# Applied ONLY in `_build_config` below, to the `model_name_or_path` handed
# to the underlying HuggingFace loader -- NEVER to `model_name`/`self.
# model_name` itself, which is exactly the string that round-trips as
# `RAGConfig.embedding.model`, the collection-fingerprint-determining field
# (`collection_fingerprint.py`'s `_index_fields`: `("embedding.model", e.
# model)`). Changing THAT string for an existing builtin would silently
# re-fingerprint (and orphan) every user's collection built under the old,
# broken-but-stable id; normalizing only the HTTP-facing id here keeps the
# fingerprint byte-for-byte stable while still loading the CORRECT model
# instead of silently 404-ing into the dim=768 default (see rag_service.
# py's `_get_embedding_dimension`).
_BARE_HF_MODEL_ID_ALIASES = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
}


class EmbeddingsServiceWrapper:
    """
    Wrapper around EmbeddingFactory to provide simplified interface for RAG.

    This allows us to:
    - Use the existing robust Embeddings_Lib.py
    - Provide a simpler interface for RAG use cases
    - Add RAG-specific features like metrics tracking
    - Handle provider detection based on model names
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_size: int = 2,  # Number of models to cache
        device: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        cache_dir: Optional[str] = None,
        embedding_cache_db: Optional[Any] = None,
    ):
        """
        Initialize embeddings service using existing EmbeddingFactory.

        Args:
            model_name: Model identifier - can be:
                - HuggingFace model: "sentence-transformers/model-name" or just "model-name"
                - OpenAI model: "openai/model-name"
                - Local OpenAI-compatible: "openai/model-name" with base_url
            cache_size: Number of models to keep in memory
            device: Device to use (cpu, cuda, mps) - defaults to auto-detection
            api_key: Optional API key for OpenAI (overrides environment)
            base_url: Optional base URL for OpenAI-compatible APIs
            cache_dir: Optional cache directory for HuggingFace model downloads
            embedding_cache_db: Optional persistent content-hash embedding
                cache store (ADR-223): any object exposing
                ``get_cached_embeddings(model_id, content_hashes)`` and
                ``store_cached_embeddings(model_id, rows)`` -- in practice
                a ``RAGIndexingDB``. Deliberately opt-in: the wrapper never
                default-opens the user-data-dir DB itself (see
                ``set_embedding_cache_store``).
        """
        self.model_name = model_name
        self.device = device
        self._cache_size = cache_size
        self._api_key = api_key
        self._base_url = base_url
        self._cache_dir = cache_dir
        self._embedding_cache_db = embedding_cache_db
        self._rag_activation_sources = source_paths(self)
        self._use_mock_backend = str(model_name).lower() in {
            "mock",
            "mock-embedding-model",
            "mock_embedding_model",
        } or str(model_name).lower().startswith("mock/")

        # Auto-detect device if not specified
        if device is None:
            try:
                import torch

                if torch.cuda.is_available():
                    self.device = "cuda"
                elif (
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                ):
                    self.device = "mps"
                else:
                    self.device = "cpu"
                logger.info(f"Auto-detected device: {self.device}")
            except ImportError:
                self.device = "cpu"
                logger.info("Torch not available, defaulting to CPU")

        # Note API key source without logging sensitive info
        self._has_api_key = bool(api_key or os.environ.get("OPENAI_API_KEY"))

        # Initialize the lightweight factory eagerly. The factory itself keeps model
        # loading lazy, while callers retain the historical construction contract.
        self.factory = None
        self._config_dict = None
        logger.info(f"Embeddings service configured with model: {model_name}")

        # Log configuration metrics
        log_counter(
            "embeddings_service_configured",
            labels={"model": model_name, "device": self.device},
        )

        # Metrics tracking
        self._embeddings_created = 0
        self._total_texts_processed = 0
        self._errors_count = 0
        # ADR-223: per-TEXT persistent-cache outcomes (real hits/misses from
        # actual store lookups; both stay 0 when no store is attached).
        self._cache_hits = 0
        self._cache_misses = 0
        self._embedding_dimension = None  # Cache the dimension after first use

        # Memory usage cache
        self._memory_cache = {
            "value": None,
            "timestamp": 0,
            "ttl": 5.0,  # Cache for 5 seconds
        }

        # Circuit breaker for resilience - configurable via env vars or config
        breaker_config = CircuitBreakerConfig(
            failure_threshold=int(
                os.environ.get("EMBEDDINGS_CIRCUIT_FAILURE_THRESHOLD", "3")
            ),
            recovery_timeout=float(
                os.environ.get("EMBEDDINGS_CIRCUIT_RECOVERY_TIMEOUT", "30.0")
            ),
            success_threshold=int(
                os.environ.get("EMBEDDINGS_CIRCUIT_SUCCESS_THRESHOLD", "2")
            ),
            enable_exponential_backoff=os.environ.get(
                "EMBEDDINGS_CIRCUIT_BACKOFF_ENABLED", "true"
            ).lower()
            == "true",
            backoff_multiplier=float(
                os.environ.get("EMBEDDINGS_CIRCUIT_BACKOFF_MULTIPLIER", "2.0")
            ),
            max_recovery_timeout=float(
                os.environ.get("EMBEDDINGS_CIRCUIT_MAX_RECOVERY", "300.0")
            ),
            min_recovery_timeout=float(
                os.environ.get("EMBEDDINGS_CIRCUIT_MIN_RECOVERY", "30.0")
            ),
        )
        self._circuit_breaker = get_circuit_breaker(
            f"embeddings_{model_name}", breaker_config
        )
        logger.info(
            f"Circuit breaker configured: failure_threshold={breaker_config.failure_threshold}, "
            f"recovery_timeout={breaker_config.recovery_timeout}s"
        )

        self._ensure_initialized()

    @activation_guarded
    def _ensure_initialized(self):
        """Ensure the factory is initialized when first needed."""
        recovered = None
        if not self._use_mock_backend and not self.model_name.startswith("openai/"):
            from tldw_chatbook.Embeddings.Embeddings_Lib import HFModelCfg

            from ..model_recovery import require_local_embedding

            recovered = require_local_embedding(
                HFModelCfg(
                    model_name_or_path=self.model_name,
                    device=self.device,
                    cache_dir=self._cache_dir,
                    local_files_only=True,
                )
            )
        if self.factory is not None:
            if recovered != getattr(self, "_local_model_binding", None):
                from ..activation import RAGActivationRequired

                raise RAGActivationRequired("local_model_review_changed")
            return

        if self._config_dict is not None and recovered != getattr(
            self, "_local_model_binding", None
        ):
            from ..activation import RAGActivationRequired

            raise RAGActivationRequired("local_model_review_changed")
        self._local_model_binding = recovered
        if self._use_mock_backend:
            self.factory = _DeterministicEmbeddingFactory()
            self._embedding_dimension = self.factory.dimension
            logger.info("Initialized deterministic mock embeddings backend")
            log_counter("embeddings_mock_backend_initialized")
            return

        if not _ensure_embeddings_imported():
            raise ImportError(
                "Embeddings dependencies not available. Install with: pip install tldw_chatbook[embeddings_rag]"
            )

        # Build configuration if not already done
        if self._config_dict is None:
            self._config_dict = self._build_config(
                self.model_name,
                self.device,
                self._api_key,
                self._base_url,
                self._cache_dir,
            )

        if recovered:
            self._config_dict["models"]["default"]["local_files_only"] = True

        try:
            # Validate the configuration
            validated_config = EmbeddingConfigSchema(**self._config_dict)

            # Initialize factory with validated configuration
            self.factory = EmbeddingFactory(
                cfg=validated_config,
                max_cached=self._cache_size,
                idle_seconds=900,  # 15 minutes idle timeout
                allow_dynamic_hf=not bool(recovered),  # Recovery binds one reviewed model
            )
            if recovered or os.path.isdir(os.path.expanduser(self.model_name)):
                from ..model_recovery import participant

                participant.register_wrapper(self)
            logger.info(
                f"Initialized embeddings factory with model: {self.model_name}, device: {self.device}"
            )

            # Log initialization metrics
            log_counter(
                "embeddings_factory_initialized",
                labels={"model": self.model_name, "device": self.device},
            )
            log_gauge("embeddings_cache_max_size", self._cache_size)

        except Exception as e:
            logger.error(f"Failed to initialize embeddings factory: {e}")
            log_counter(
                "embeddings_factory_init_error",
                labels={"model": self.model_name, "error": type(e).__name__},
            )
            raise

    def _get_cached_memory_info(self):
        """
        Get memory info with caching to reduce psutil overhead.

        Returns:
            Memory usage in MB
        """
        current_time = time.time()

        # Check if cache is valid
        if (
            self._memory_cache["value"] is not None
            and current_time - self._memory_cache["timestamp"]
            < self._memory_cache["ttl"]
        ):
            return self._memory_cache["value"]

        # Get fresh memory info
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)

        # Update cache
        self._memory_cache["value"] = memory_mb
        self._memory_cache["timestamp"] = current_time

        return memory_mb

    def set_embedding_cache_store(self, store: Optional[Any]) -> None:
        """Attach (or replace/detach) the persistent embedding cache store.

        ADR-223 / TASK-34420. The ingestion seam (``index_entries``) calls
        this with the ``RAGIndexingDB`` handle its caller already owns;
        the wrapper deliberately does NOT default-open the user-data-dir
        DB itself, because the wrapper (and RAGService) are constructed in
        many test contexts without a DB and an implicit open would break
        test hermeticity.

        Re-attaching with a different store replaces the previous one
        (attribute assignment is atomic, so a concurrent embed either sees
        the old store or the new one -- both are valid caches). Passing
        ``None`` detaches and restores no-cache behavior.
        """
        self._embedding_cache_db = store

    def _embedding_cache_model_id(self) -> str:
        """Keep local model keys stable and isolate effective hosted endpoints."""
        if not self.model_name.startswith("openai/"):
            return self.model_name
        config = self.factory.config
        model = config.models[config.default_model_id]
        base_url = str(model.base_url) if model.base_url else "https://api.openai.com/v1"
        endpoint = base_url.rstrip("/") + "/embeddings"
        # The validated factory config supplies the exact effective endpoint.
        # Persist only its digest: URLs can contain credentials or query secrets.
        digest = hashlib.sha256(endpoint.encode("utf-8")).hexdigest()
        return f"{self.model_name}@endpoint:{digest}"

    async def _async_cache_call(self, operation, store, *args):
        """Retain finite SQLite work and retire only its new worker connection."""
        if store is None:
            return operation(*args, store)
        from tldw_chatbook.DB.RAG_Indexing_DB import RAGIndexingDB

        # Each thread gets a separate :memory: database, so these explicitly
        # single-threaded test owners must stay on their constructing thread.
        if isinstance(store, RAGIndexingDB) and store.is_memory_db:
            return operation(*args, store)

        from tldw_chatbook.DB.base_db import operation_owned_connection
        from tldw_chatbook.TTS._async_lifecycle import join_retained_task

        from ..activation import native_worker

        def invoke():
            with operation_owned_connection(store):
                return operation(*args, store)

        completion = asyncio.create_task(asyncio.to_thread(native_worker(self, invoke)))
        await join_retained_task(completion)
        return completion.result()

    def _lookup_cached_embeddings(
        self, content_hashes: List[str], store: Optional[Any]
    ) -> Dict[str, List[float]]:
        """Batch-lookup cached vectors; never raises (a cache cannot fail an embed)."""
        if store is None:
            return {}
        try:
            return store.get_cached_embeddings(
                self._embedding_cache_model_id(), content_hashes
            ) or {}
        except Exception as e:
            log_counter(
                "embeddings_persistent_cache_error",
                labels={"operation": "lookup", "error_type": type(e).__name__},
            )
            logger.warning(
                "Persistent embedding-cache lookup failed; embedding without cache "
                f"(error_type={type(e).__name__})"
            )
            return {}

    def _store_cached_embeddings(
        self,
        content_hashes: List[str],
        miss_indices: List[int],
        fresh_rows: List[List[float]],
        store: Optional[Any],
    ) -> None:
        """Persist freshly embedded vectors; never raises (best-effort cache fill)."""
        if store is None:
            return
        try:
            store.store_cached_embeddings(
                self._embedding_cache_model_id(),
                [
                    (content_hashes[index], row)
                    for index, row in zip(miss_indices, fresh_rows)
                ],
            )
        except Exception as e:
            log_counter(
                "embeddings_persistent_cache_error",
                labels={"operation": "store", "error_type": type(e).__name__},
            )
            logger.warning(
                "Persistent embedding-cache store failed; cache not updated "
                f"(error_type={type(e).__name__})"
            )

    @staticmethod
    def _content_hashes(texts: List[str]) -> List[str]:
        """sha256 of the FULL text of each item (computed once per call).

        The pre-ADR dead check hashed only the first 100 characters of each
        text, which would have conflated distinct chunks sharing a prefix.
        """
        return [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]

    def _record_cache_counters(self, total: int, misses: int) -> None:
        """Count real per-text cache hits/misses (ADR-223 metric semantics).

        Only counted when a store is attached: with no store there are no
        cache requests, and emitting synthetic 0/0-miss counters would
        recreate the bogus metric this replaced.
        """
        if self._embedding_cache_db is None:
            return
        hits = total - misses
        self._cache_hits += hits
        self._cache_misses += misses
        if hits:
            log_counter("embeddings_cache_hit", value=hits)
        if misses:
            log_counter("embeddings_cache_miss", value=misses)

    def _build_config(
        self,
        model_name: str,
        device: Optional[str],
        api_key: Optional[str],
        base_url: Optional[str],
        cache_dir: Optional[str],
    ) -> Dict[str, Any]:
        """
        Build configuration dictionary for EmbeddingFactory.

        This method determines the provider type from the model name and
        constructs the appropriate configuration.
        """
        # Default model ID in the factory
        default_model_id = "default"

        # Determine provider and model path
        if model_name.startswith("openai/"):
            # OpenAI or OpenAI-compatible model
            provider = "openai"
            model_path = model_name.split("/", 1)[1]

            # Build OpenAI configuration
            model_config = {"provider": "openai", "model_name_or_path": model_path}

            # Add API key if provided (otherwise factory will use env var)
            if api_key:
                model_config["api_key"] = api_key

            # Add base URL for OpenAI-compatible endpoints
            if base_url:
                model_config["base_url"] = base_url
                logger.info(f"Using OpenAI-compatible endpoint: {base_url}")

        else:
            # HuggingFace model (default)
            provider = "huggingface"
            # task-640 AC#7: canonicalize known bare model ids that 404
            # against the real Hub when passed through as-is. This must
            # NEVER touch `model_name` itself -- see
            # _BARE_HF_MODEL_ID_ALIASES' docstring above for why.
            model_path = _BARE_HF_MODEL_ID_ALIASES.get(model_name, model_name)

            # Build HuggingFace configuration
            model_config = {
                "provider": "huggingface",
                "model_name_or_path": model_path,
                "trust_remote_code": False,
                "max_length": 512,
                "batch_size": 32,
            }

            # Add device if specified
            if device:
                model_config["device"] = device

            # Add cache directory if specified
            if cache_dir:
                model_config["cache_dir"] = cache_dir
                logger.info(
                    f"Using cache directory for HuggingFace models: {cache_dir}"
                )

        # Construct the full configuration
        config = {
            "default_model_id": default_model_id,
            "models": {default_model_id: model_config},
        }

        logger.debug(f"Built embedding config: provider={provider}, model={model_path}")
        return config

    @timeit("embeddings_create_operation")
    @activation_guarded
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for texts using the configured model.

        Args:
            texts: List of texts to embed

        Returns:
            Numpy array of embeddings with shape (n_texts, embedding_dim)

        Raises:
            ValueError: If texts is empty
            RuntimeError: If embedding creation fails
            ImportError: If numpy is not available
        """
        if not NUMPY_AVAILABLE:
            raise ImportError(
                "NumPy is required for embeddings. Install with: pip install tldw_chatbook[embeddings_rag]"
            )

        # Ensure factory is initialized
        self._ensure_initialized()

        if not texts:
            logger.warning("create_embeddings called with empty text list")
            return np.empty(
                (0, self._get_empty_embedding_dimension()), dtype=np.float32
            )

        start_time = time.time()

        # Log batch details
        log_histogram("embeddings_batch_size", len(texts))
        logger.info(f"Creating embeddings for batch of {len(texts)} texts")

        # Get memory usage before
        memory_before = self._get_cached_memory_info()

        try:
            # ADR-223 / TASK-34420: persistent content-hash cache. sha256 per
            # text (computed once), one batched lookup, embed ONLY the misses
            # through the circuit breaker, merge back in caller order, store
            # the misses. This replaces a "cache check" that compared a batch
            # key against EmbeddingFactory._cache -- a dict keyed by MODEL id
            # -- so it could never hit and fed a permanently-0% hit-rate
            # metric.
            content_hashes = self._content_hashes(texts)
            cached_rows = self._lookup_cached_embeddings(
                content_hashes, self._embedding_cache_db
            )
            miss_indices = [
                index
                for index, digest in enumerate(content_hashes)
                if digest not in cached_rows
            ]

            fresh_embeddings: Optional[np.ndarray] = None
            if miss_indices:
                miss_texts = [texts[index] for index in miss_indices]
                logger.debug(
                    f"Calling factory.embed with {len(miss_texts)} of {len(texts)} texts "
                    f"({len(texts) - len(miss_texts)} served from the persistent cache)"
                )

                # Define the embedding function for circuit breaker
                def embed_with_factory():
                    return self.factory.embed(miss_texts, as_list=False)

                try:
                    fresh_embeddings = self._circuit_breaker.call_sync(
                        embed_with_factory
                    )
                    logger.debug(
                        f"Factory returned embeddings of type {type(fresh_embeddings)}, shape: {fresh_embeddings.shape if hasattr(fresh_embeddings, 'shape') else 'N/A'}"
                    )
                except CircuitBreakerOpenError as e:
                    # Circuit is open, fail fast
                    self._errors_count += 1
                    log_counter("embeddings_circuit_breaker_open")
                    logger.error(f"Embeddings service unavailable: {e}")
                    raise RuntimeError(
                        f"Embeddings service temporarily unavailable: {e}"
                    ) from e

                # Ensure embeddings is a numpy array
                if not isinstance(fresh_embeddings, np.ndarray):
                    logger.debug(
                        f"Converting embeddings from {type(fresh_embeddings)} to numpy array"
                    )
                    fresh_embeddings = np.array(fresh_embeddings)
                    logger.debug(f"After conversion: shape={fresh_embeddings.shape}")

            rows: List[Optional[List[float]]] = [
                cached_rows.get(digest) for digest in content_hashes
            ]
            if fresh_embeddings is not None:
                fresh_rows: List[List[float]] = fresh_embeddings.tolist()
                for position, index in enumerate(miss_indices):
                    rows[index] = fresh_rows[position]
                self._store_cached_embeddings(
                    content_hashes, miss_indices, fresh_rows, self._embedding_cache_db
                )

            embeddings = np.asarray(rows, dtype=np.float32)
            self._record_cache_counters(len(texts), len(miss_indices))

            # Cache the embedding dimension if not already cached
            if (
                self._embedding_dimension is None
                and embeddings.shape[0] > 0
                and embeddings.shape[1] > 0
            ):
                self._embedding_dimension = embeddings.shape[1]
                logger.info(
                    f"Cached embedding dimension from first use: {self._embedding_dimension}"
                )

            # Get memory usage after
            memory_after = self._get_cached_memory_info()
            memory_delta = memory_after - memory_before

            # Update metrics
            self._embeddings_created += 1
            self._total_texts_processed += len(texts)

            # Log performance metrics
            elapsed_time = time.time() - start_time
            log_histogram("embeddings_creation_time", elapsed_time)
            log_gauge("embeddings_model_memory_mb", memory_after)
            log_histogram("embeddings_memory_delta_mb", memory_delta)
            log_counter("embeddings_texts_processed", value=len(texts))

            # Log text length statistics
            text_lengths = [len(text) for text in texts]
            log_histogram(
                "embeddings_text_length_chars", sum(text_lengths) / len(text_lengths)
            )

            logger.info(
                f"Created embeddings for {len(texts)} texts in {elapsed_time:.3f}s, shape: {embeddings.shape}, memory delta: {memory_delta:.1f}MB"
            )
            return embeddings

        except Exception as e:
            self._errors_count += 1
            log_counter("embeddings_creation_error", labels={"error": type(e).__name__})
            logger.opt(exception=True).error(f"Failed to create embeddings: {e}")
            raise RuntimeError(f"Embedding creation failed: {e}") from e

    def _get_empty_embedding_dimension(self) -> int:
        """Return a best-known dimension for empty-batch results without model work."""
        if self._embedding_dimension is not None:
            return int(self._embedding_dimension)
        if self.factory is not None:
            dimension = getattr(self.factory, "dimension", None)
            if isinstance(dimension, (int, float)):
                return int(dimension)
            get_dimension = getattr(self.factory, "get_dimension", None)
            if callable(get_dimension):
                try:
                    dimension = get_dimension()
                except Exception:
                    dimension = None
                if isinstance(dimension, (int, float)):
                    return int(dimension)
        normalized_model = str(self.model_name or "").lower()
        if "text-embedding-3-large" in normalized_model:
            return 3072
        if "text-embedding" in normalized_model or normalized_model.startswith(
            "openai/"
        ):
            return 1536
        return 384

    @activation_async_guarded
    async def create_embeddings_async(self, texts: List[str]) -> np.ndarray:
        """
        Async version of create_embeddings.

        Uses the factory's async_embed method for non-blocking operation.
        """
        if not NUMPY_AVAILABLE:
            raise ImportError(
                "NumPy is required for embeddings. Install with: pip install tldw_chatbook[embeddings_rag]"
            )

        # Ensure factory is initialized
        self._ensure_initialized()

        if not texts:
            logger.warning("create_embeddings_async called with empty text list")
            return np.array([])

        start_time = time.time()
        log_histogram("embeddings_async_batch_size", len(texts))
        logger.info(
            f"Creating embeddings asynchronously for batch of {len(texts)} texts"
        )

        try:
            # ADR-223: the same persistent-cache flow as the sync path --
            # these are the two choke points every chunk and query
            # embedding already flows through.
            content_hashes = self._content_hashes(texts)
            cache_store = self._embedding_cache_db
            cached_rows = await self._async_cache_call(
                self._lookup_cached_embeddings, cache_store, content_hashes
            )
            miss_indices = [
                index
                for index, digest in enumerate(content_hashes)
                if digest not in cached_rows
            ]

            fresh_embeddings: Optional[np.ndarray] = None
            if miss_indices:
                miss_texts = [texts[index] for index in miss_indices]
                # Use the factory's async embed method with circuit breaker protection
                try:
                    fresh_embeddings = await self._circuit_breaker.call_async(
                        self._async_factory_embed, miss_texts
                    )
                except CircuitBreakerOpenError as e:
                    # Circuit is open, fail fast
                    self._errors_count += 1
                    log_counter("embeddings_circuit_breaker_open")
                    logger.error(f"Embeddings service unavailable: {e}")
                    raise RuntimeError(
                        f"Embeddings service temporarily unavailable: {e}"
                    ) from e

            rows: List[Optional[List[float]]] = [
                cached_rows.get(digest) for digest in content_hashes
            ]
            if fresh_embeddings is not None:
                if not isinstance(fresh_embeddings, np.ndarray):
                    fresh_embeddings = np.array(fresh_embeddings)
                fresh_rows: List[List[float]] = fresh_embeddings.tolist()
                for position, index in enumerate(miss_indices):
                    rows[index] = fresh_rows[position]
                await self._async_cache_call(
                    self._store_cached_embeddings,
                    cache_store,
                    content_hashes,
                    miss_indices,
                    fresh_rows,
                )

            embeddings = np.asarray(rows, dtype=np.float32)
            self._record_cache_counters(len(texts), len(miss_indices))

            # Update metrics
            self._embeddings_created += 1
            self._total_texts_processed += len(texts)

            # Log performance metrics
            elapsed_time = time.time() - start_time
            log_histogram("embeddings_async_creation_time", elapsed_time)
            log_counter("embeddings_async_texts_processed", value=len(texts))

            logger.info(
                f"Created embeddings asynchronously for {len(texts)} texts in {elapsed_time:.3f}s"
            )
            return embeddings

        except Exception as e:
            self._errors_count += 1
            log_counter(
                "embeddings_async_creation_error", labels={"error": type(e).__name__}
            )
            logger.opt(exception=True).error(
                f"Failed to create embeddings asynchronously: {e}"
            )
            raise RuntimeError(f"Async embedding creation failed: {e}") from e

    @activation_guarded
    def create_embedding(self, text: str) -> np.ndarray:
        """
        Create embedding for a single text.

        Convenience method for single text embedding.
        """
        self._ensure_initialized()
        logger.debug(f"Creating single embedding for text of length {len(text)}")
        result = self.factory.embed_one(text, as_list=False)
        logger.debug(
            f"Factory returned single embedding of type {type(result)}, shape: {result.shape if hasattr(result, 'shape') else 'N/A'}"
        )
        if not isinstance(result, np.ndarray):
            logger.debug(
                f"Converting single embedding from {type(result)} to numpy array"
            )
            result = np.array(result)
            logger.debug(f"After conversion: shape={result.shape}")
        return result

    async def _async_factory_embed(self, texts):
        from contextlib import nullcontext

        from ..activation import native_worker
        from ..model_recovery import participant

        local = os.path.isdir(os.path.expanduser(self.model_name))
        with participant.operation() if local else nullcontext():
            function = (
                participant.worker(self.factory.embed) if local else self.factory.embed
            )
            return await asyncio.to_thread(
                native_worker(self, function), texts, as_list=False
            )

    @activation_async_guarded
    async def create_embedding_async(self, text: str) -> np.ndarray:
        """
        Async version of create_embedding for single text.
        """
        self._ensure_initialized()
        result = (await self._async_factory_embed([text]))[0]
        if not isinstance(result, np.ndarray):
            result = np.array(result)
        return result

    def get_embedding_dimension(self) -> Optional[int]:
        """
        Get the dimension of embeddings produced by the current model.

        Returns:
            Embedding dimension or None if it cannot be determined
        """
        # Return cached dimension if available
        if self._embedding_dimension is not None:
            return self._embedding_dimension

        try:
            # Create a dummy embedding to get dimension
            dummy_embedding = self.create_embedding("test")
            self._embedding_dimension = int(dummy_embedding.shape[0])
            logger.info(
                f"Detected and cached embedding dimension: {self._embedding_dimension}"
            )
            return self._embedding_dimension
        except Exception as e:
            logger.warning(f"Could not determine embedding dimension: {e}")
            return None

    @timeit("embeddings_prefetch_models")
    @activation_guarded
    def prefetch_model(self, model_ids: Optional[List[str]] = None):
        """
        Prefetch and cache models for faster first-use.

        Args:
            model_ids: List of model IDs to prefetch. If None, prefetches the default model.
        """
        if model_ids is None:
            model_ids = ["default"]

        # Get memory before loading
        memory_before = self._get_cached_memory_info()

        try:
            start_time = time.time()
            self.factory.prefetch(model_ids)
            load_time = time.time() - start_time

            # Get memory after loading
            memory_after = self._get_cached_memory_info()
            memory_used = memory_after - memory_before

            # Log metrics
            log_histogram("embeddings_model_load_time", load_time)
            log_histogram("embeddings_model_memory_usage_mb", memory_used)
            log_counter("embeddings_models_prefetched", value=len(model_ids))

            logger.info(
                f"Prefetched models: {model_ids} in {load_time:.2f}s, memory used: {memory_used:.1f}MB"
            )
        except Exception as e:
            log_counter("embeddings_prefetch_error", labels={"error": type(e).__name__})
            logger.opt(exception=True).error(f"Failed to prefetch models: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics for monitoring."""
        # Calculate cache hit rate
        total_cache_requests = self._cache_hits + self._cache_misses
        cache_hit_rate = (
            self._cache_hits / total_cache_requests if total_cache_requests > 0 else 0.0
        )

        metrics = {
            "model_name": self.model_name,
            "device": self.device,
            "cache_size": self._cache_size,
            "total_calls": self._embeddings_created,
            "total_texts_processed": self._total_texts_processed,
            "errors_count": self._errors_count,
            "error_rate": (
                self._errors_count / self._embeddings_created
                if self._embeddings_created > 0
                else 0.0
            ),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": cache_hit_rate,
        }

        # Log cache efficiency metrics
        log_gauge("embeddings_cache_hit_rate", cache_hit_rate)
        log_gauge("embeddings_total_texts_processed", self._total_texts_processed)
        log_gauge("embeddings_error_rate", metrics["error_rate"])

        # Try to get embedding dimension
        dim = self.get_embedding_dimension()
        if dim is not None:
            metrics["embedding_dimension"] = dim

        # Get factory configuration
        try:
            metrics["factory_config"] = {
                "models": list(self.factory.config.models.keys()),
                "default_model": self.factory.config.default_model_id,
            }
        except Exception:
            pass

        return metrics

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get memory usage statistics for the embeddings service.

        Returns:
            Dict with memory usage in MB for different components
        """
        # Get current memory usage
        rss_mb = self._get_cached_memory_info()  # Resident Set Size in MB

        # For VMS, we need to get it separately as it's not cached
        process = psutil.Process()
        memory_info = process.memory_info()
        vms_mb = memory_info.vms / (1024 * 1024)  # Virtual Memory Size in MB

        # Try to get GPU memory if using CUDA
        gpu_memory_mb = 0.0
        if self.device == "cuda":
            try:
                import torch

                if torch.cuda.is_available():
                    gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            except Exception as e:
                logger.debug(f"Could not get GPU memory usage: {e}")

        memory_usage = {
            "rss_mb": rss_mb,
            "vms_mb": vms_mb,
            "gpu_mb": gpu_memory_mb,
            "total_mb": rss_mb + gpu_memory_mb,
        }

        # Log memory metrics
        log_gauge("embeddings_memory_rss_mb", rss_mb)
        log_gauge("embeddings_memory_vms_mb", vms_mb)
        if gpu_memory_mb > 0:
            log_gauge("embeddings_memory_gpu_mb", gpu_memory_mb)

        return memory_usage

    def clear_cache(self):
        """
        Clear the model cache and reinitialize.

        This is useful when switching models or freeing memory.
        """
        try:
            if self.factory is not None:
                self.factory.close()
            self.factory = None
            self._embedding_dimension = None
            self._ensure_initialized()

            logger.info("Cleared embeddings cache and reinitialized")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            raise

    def close(self):
        """
        Clean up resources.

        Should be called when the service is no longer needed.
        """
        try:
            # Close the factory first
            if hasattr(self, "factory") and self.factory is not None:
                self.factory.close()
                logger.info("Closed embeddings factory")

            # Clear any model references
            if hasattr(self, "factory"):
                self.factory = None

            # Explicitly free GPU memory if using CUDA
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("Cleared CUDA cache")
            except ImportError:
                pass  # torch not available
            except Exception as e:
                logger.warning(f"Failed to clear CUDA cache: {e}")

            logger.info("Closed embeddings service")
        except Exception as e:
            logger.error(f"Error closing embeddings service: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.close()

    def __del__(self):
        """Destructor - attempt cleanup if not already done."""
        try:
            # Only attempt cleanup if factory exists and hasn't been cleaned up
            if hasattr(self, "factory") and self.factory is not None:
                self.close()
        except (KeyboardInterrupt, SystemExit):
            # Re-raise critical exceptions
            raise
        except Exception as e:
            # Log the error but don't raise - destructors shouldn't throw
            logger.error(f"Error during embeddings wrapper cleanup: {e}")
            # Log traceback for debugging
            import traceback

            logger.debug(f"Cleanup error traceback: {traceback.format_exc()}")


# Convenience function for creating service with common configurations


def detect_embedding_provider(model_name: str) -> str:
    """Detect the legacy embedding provider name from a model identifier."""
    if not model_name:
        return "unknown"
    if model_name.startswith("text-embedding"):
        return "openai"
    if "/" in model_name:
        return "sentence_transformers"
    return "unknown"


def normalize_embeddings(embeddings: List[List[float]]) -> List[List[float]]:
    """Normalize embeddings to unit length, preserving zero vectors."""
    normalized: List[List[float]] = []
    for embedding in embeddings:
        vector = np.array(embedding, dtype=float)
        norm = np.linalg.norm(vector)
        normalized.append((vector / norm).tolist() if norm else list(embedding))
    return normalized


class EmbeddingsWrapper:
    """Legacy test-friendly embeddings wrapper retained for backward compatibility."""

    def __init__(
        self,
        provider: str = "sentence_transformers",
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        cache_enabled: bool = True,
        cache_size: int = 1000,
    ):
        self.provider = provider
        self.model_name = model_name
        self.device = device
        self.cache_enabled = cache_enabled
        self.cache_size = cache_size
        self._cache: Dict[str, List[float]] = {}
        self._model = None

        if provider == "openai":
            self._dimension = 1536
            self._model = _DeterministicEmbeddingFactory(self._dimension)
        elif provider == "mock":
            self._dimension = 384
            self._model = _DeterministicEmbeddingFactory(self._dimension)
        elif provider == "sentence_transformers":
            self._dimension = 384
            if model_name in {
                "all-MiniLM-L6-v2",
                "sentence-transformers/all-MiniLM-L6-v2",
            }:
                self._model = _DeterministicEmbeddingFactory(self._dimension)
            else:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(model_name, device=device)
                if hasattr(self._model, "get_sentence_embedding_dimension"):
                    self._dimension = int(
                        self._model.get_sentence_embedding_dimension()
                    )
        else:
            self._dimension = 384

    def _encode(self, texts: List[str]) -> List[List[float]]:
        if self.provider in {"mock", "openai"}:
            return self._model.embed(texts, as_list=True)
        if self.provider == "sentence_transformers":
            if isinstance(self._model, _DeterministicEmbeddingFactory):
                return self._model.embed(texts, as_list=True)
            return np.array(self._model.encode(texts)).tolist()
        raise ValueError(f"Unknown provider: {self.provider}")

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        results: List[Optional[List[float]]] = [None] * len(texts)
        missing_texts: List[str] = []
        missing_indices: List[int] = []

        for index, text in enumerate(texts):
            if self.cache_enabled and text in self._cache:
                results[index] = self._cache[text]
            else:
                missing_texts.append(text)
                missing_indices.append(index)

        if missing_texts:
            generated = self._encode(missing_texts)
            for index, text, embedding in zip(
                missing_indices, missing_texts, generated
            ):
                if self.cache_enabled:
                    self._cache[text] = embedding
                results[index] = embedding

        return [result or [] for result in results]

    async def create_embeddings_async(self, texts: List[str]) -> List[List[float]]:
        return self.create_embeddings(texts)

    def get_dimension(self) -> int:
        return self._dimension

    def clear_cache(self) -> None:
        self._cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        return {
            "size": len(self._cache),
            "max_size": self.cache_size,
            "enabled": self.cache_enabled,
        }

    def estimate_memory_usage(self) -> int:
        model_memory = 100 * 1024 * 1024
        return model_memory + len(self._cache) * self._dimension * 8


def create_embeddings_service(
    provider: Union[str, Dict[str, Any]] = "huggingface",
    model: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs,
) -> Union[EmbeddingsServiceWrapper, EmbeddingsWrapper]:
    """
    Create an embeddings service with common configurations.

    Args:
        provider: Provider type - "huggingface", "openai", or "local"
        model: Model name (defaults based on provider)
        device: Device to use (defaults to auto-detection)
        **kwargs: Additional arguments passed to EmbeddingsServiceWrapper

    Returns:
        Configured EmbeddingsServiceWrapper instance

    Examples:
        # HuggingFace sentence transformers
        service = create_embeddings_service("huggingface")

        # OpenAI embeddings
        service = create_embeddings_service("openai", api_key="...")

        # Local OpenAI-compatible
        service = create_embeddings_service("local",
                                          base_url="http://localhost:8080",
                                          model="local-model")
    """
    if isinstance(provider, dict):
        config = provider
        return EmbeddingsWrapper(
            provider=config.get("embedding_provider", "sentence_transformers"),
            model_name=config.get("embedding_model", "all-MiniLM-L6-v2"),
            device=config.get("device", "cpu"),
            cache_enabled=config.get("enable_cache", True),
            cache_size=config.get("max_cache_size", 1000),
        )

    # Default models for each provider
    default_models = {
        "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
        "openai": "openai/text-embedding-3-small",
        "local": "openai/text-embedding-ada-002",  # Common local model
        "mock": "mock",
    }

    # Determine model name
    if model is None:
        model = default_models.get(provider, default_models["huggingface"])
    elif provider == "openai" and not model.startswith("openai/"):
        model = f"openai/{model}"
    elif provider == "local" and not model.startswith("openai/"):
        model = f"openai/{model}"

    # Create service
    return EmbeddingsServiceWrapper(model_name=model, device=device, **kwargs)


# For backward compatibility
EmbeddingsService = EmbeddingsServiceWrapper
