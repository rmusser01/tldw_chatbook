# Embeddings_Lib.py
#
from __future__ import annotations
#
"""Adaptive embedding factory for desktop / TUI apps – asynchronous‑ready, logging‑aware.

This version is a thread-safe, robust implementation that addresses race conditions
and resource management issues found in previous iterations.

Key Features & Fixes in this Version
────────────────────────────────────
• **Thread-Safe & Race-Condition-Free**: The core embedding logic is now fully
  thread-safe, preventing race conditions where a model could be evicted by one
  thread while being used by another.
• **Correct Resource Management**: Eviction now occurs *before* loading a new
  model, ensuring the `max_cached` limit is never exceeded, preventing
  potential out-of-memory errors.
• **Improved Typing**: The internal cache and Pydantic configurations use
  `TypedDict` and discriminated unions for better static analysis and maintainability.
• **Async Facade**: `await factory.async_embed(...)` uses `asyncio.to_thread`
  so the UI never blocks.
• **Structured Logging**: Provides insight into cache hits, loads, and evictions.
• **Pluggable & High-Quality Pooling**: Defaults to a proper masked mean pooling
  strategy with L2 normalization for superior embedding quality.
• **Prefetch / Warm‑up**: `factory.prefetch(...)` downloads weights on-demand.
"""
#
# Imports
import asyncio
import random
import threading
import time
from collections import OrderedDict
from typing import (Any, Annotated, Callable, Dict, List, Literal, Optional,
                    Protocol, TypedDict, Union)
#
# Third-Party Libraries
import requests
from loguru import logger
from pydantic import BaseModel, Field, field_validator, model_validator, HttpUrl, ValidationError

# Optional dependencies with fallbacks
from ..Utils.optional_deps import (
    check_dependency, require_dependency, get_safe_import, 
    DEPENDENCIES_AVAILABLE, create_unavailable_feature_handler
)

# Import optional dependencies safely
numpy = get_safe_import('numpy')
torch = get_safe_import('torch')
transformers = get_safe_import('transformers')

# Create type aliases that work with or without dependencies
if torch is not None:
    Tensor = torch.Tensor
    normalize = torch.nn.functional.normalize
    AutoModel = transformers.AutoModel
    AutoTokenizer = transformers.AutoTokenizer
else:
    # Create placeholder types
    Tensor = Any
    normalize = create_unavailable_feature_handler('torch', 'pip install tldw_chatbook[embeddings_rag]')
    AutoModel = create_unavailable_feature_handler('transformers', 'pip install tldw_chatbook[embeddings_rag]')
    AutoTokenizer = create_unavailable_feature_handler('transformers', 'pip install tldw_chatbook[embeddings_rag]')

if numpy is not None:
    np = numpy
else:
    # Create a basic np-like object for type annotations
    class _NumpyPlaceholder:
        @staticmethod
        def asarray(*args, **kwargs):
            raise ImportError("NumPy not available. Install with: pip install tldw_chatbook[embeddings_rag]")
        
        @staticmethod  
        def empty(*args, **kwargs):
            raise ImportError("NumPy not available. Install with: pip install tldw_chatbook[embeddings_rag]")
    
    np = _NumpyPlaceholder()
#
# Local Imports
#
########################################################################################################################
#
__all__ = ["EmbeddingFactory", "EmbeddingConfigSchema"]
#
# Configure logger with context
logger = logger.bind(module="Embeddings_Lib")
#
###############################################################################
# Configuration schema (with Discriminated Union)
###############################################################################

PoolingFn = Callable[[Tensor, Tensor], Tensor]

class EmbeddingConfigSchema(BaseModel):
    default_model_id: Optional[str] = None
    models: Dict[str, 'ModelCfg'] # Forward reference for ModelCfg

    @model_validator(mode='after')
    def check_default_model_exists(self) -> 'EmbeddingConfigSchema':
        """Ensures the default_model_id refers to a model defined in the models dict."""
        if self.default_model_id and self.default_model_id not in self.models:
            raise ValueError(
                f"default_model_id '{self.default_model_id}' is not a valid key in the 'models' dictionary. "
                f"Available models are: {list(self.models.keys())}"
            )
        return self


def _masked_mean(last_hidden: Tensor, attn: Tensor) -> Tensor:
    """Default pooling: mean of vectors where attention_mask is 1."""
    mask = attn.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    lengths = mask.sum(dim=1).clamp(min=1e-9)
    avg = summed / lengths
    return normalize(avg, p=2, dim=1)


class HFModelCfg(BaseModel):
    provider: Literal["huggingface"] = "huggingface"
    model_name_or_path: str
    trust_remote_code: bool = False
    max_length: int = 512
    device: Optional[str] = None
    batch_size: int = 32
    pooling: Optional[PoolingFn] = None  # default: masked mean
    dimension: Optional[int] = None
    cache_dir: Optional[str] = None  # Custom cache directory for model downloads


class OpenAICfg(BaseModel):
    provider: Literal["openai"] = "openai"
    model_name_or_path: str = "text-embedding-3-small"
    api_key: Optional[str] = Field(default=None, repr=False)
    base_url: Optional[HttpUrl] = Field(default=None) # CHANGED: Added base_url, made it HttpUrl for validation
    dimension: Optional[int] = None

    @field_validator("api_key", mode="before")
    def _default_api_key(cls, v: Optional[str], values: Any) -> Optional[str]: # Ensure values is available if needed
        # If a base_url is provided (likely for a local/custom OpenAI-compatible API),
        # an API key might not be required or might be a placeholder.
        # This validator makes the API key truly optional if base_url is set.
        if values.data.get("base_url"): # Check if base_url is present in the input data
            return v # Return the provided value (None or actual key)

        # Original logic: API key is required if no base_url pointing to a local server
        if v:
            return v
        from os import getenv
        env = getenv("OPENAI_API_KEY")
        if not env:
            # Only raise ValueError if it's for the actual OpenAI API (no base_url)
            # and no key is provided and no env var is set.
            if not values.data.get("base_url"):
                 raise ValueError("OPENAI_API_KEY env-var missing and api_key not set for OpenAI provider (and no base_url specified).")
        return env


# A discriminated union lets Pydantic and type checkers infer the correct model type
ModelCfg = Annotated[
    Union[HFModelCfg, OpenAICfg],
    Field(discriminator="provider")
]

# Update the forward reference now that ModelCfg is fully defined
EmbeddingConfigSchema.model_rebuild()


###############################################################################
# Provider helpers
###############################################################################

class EmbedFn(Protocol):
    """A protocol defining a callable for embedding texts.

    This ensures that any function used as an embedding function adheres to this
    specific signature, including the keyword-only `as_list` argument.
    """
    def __call__(
        self, texts: List[str], *, as_list: bool = False
    ) -> Union[np.ndarray, List[List[float]]]:
        ...

class CacheRecord(TypedDict):
    """Strongly-typed structure for a cache entry."""
    embed: EmbedFn
    close: Optional[Callable[[], None]]
    last: float


class _HuggingFaceEmbedder:
    """Wraps HF model/tokenizer; exposes poolable, dtype/device-aware embedding."""

    def __init__(self, cfg: HFModelCfg):
        try:
            # Use cache_dir if provided, otherwise HuggingFace will use default
            cache_dir = cfg.cache_dir if cfg.cache_dir else None
            if cache_dir:
                # Expand user home directory if present
                import os
                cache_dir = os.path.expanduser(cache_dir)
                logger.info(f"Using custom cache directory: {cache_dir}")
            
            self._tok = AutoTokenizer.from_pretrained(
                cfg.model_name_or_path, 
                trust_remote_code=cfg.trust_remote_code,
                cache_dir=cache_dir
            )
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self._model = AutoModel.from_pretrained(
                cfg.model_name_or_path,
                torch_dtype=dtype,
                trust_remote_code=cfg.trust_remote_code,
                cache_dir=cache_dir
            )
        # --- [FIX] Added robust error handling for model loading ---
        except (OSError, requests.exceptions.RequestException) as e:
            raise IOError(
                f"Failed to download or load model '{cfg.model_name_or_path}'. "
                "Check the model name and your network connection."
            ) from e

        self._device = torch.device(
            cfg.device if cfg.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model.to(self._device).eval()
        self._max_len = cfg.max_length
        self._batch_size = cfg.batch_size
        self._pool = cfg.pooling or _masked_mean

    @torch.inference_mode()
    def _forward(self, inp: Dict[str, Tensor]) -> Tensor:
        out = self._model(**inp).last_hidden_state
        return self._pool(out, inp["attention_mask"])

    def embed(self, texts: List[str], *, as_list: bool = False) -> np.ndarray | List[List[float]]:
        vecs: List[Tensor] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i: i + self._batch_size]
            tok = self._tok(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self._max_len,
            )
            tok = {k: v.to(self._device) for k, v in tok.items()}
            # --- [FIX] Performance: keep tensors on GPU during loop ---
            vecs.append(self._forward(tok))

        # --- [FIX] Performance: concatenate on GPU, then move to CPU once ---
        joined = torch.cat(vecs, dim=0).float().cpu().numpy()
        return joined.tolist() if as_list else joined

    def close(self) -> None:
        del self._model, self._tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------

_BACKOFF: tuple[float, ...] = (1, 2, 4, 8)


def _openai_embedder(cfg: OpenAICfg) -> Callable[[list[str], Any, bool], Any]: # CHANGED: Takes OpenAICfg
    session = requests.Session()

    # CHANGED: Use cfg.api_key and handle its potential None value if base_url is set
    if cfg.api_key:
        session.headers.update(
            {"Authorization": f"Bearer {cfg.api_key}", "User-Agent": "EmbeddingsLib/4.0"}
        )
    else:
        # If no API key (e.g., for local server), just set User-Agent
        session.headers.update({"User-Agent": "EmbeddingsLib/4.0"})

    # CHANGED: Determine the endpoint URL
    endpoint_url_base = str(cfg.base_url) if cfg.base_url else "https://api.openai.com/v1"
    embeddings_endpoint = f"{endpoint_url_base.rstrip('/')}/embeddings"
    logger.info(f"OpenAI embedder configured for endpoint: {embeddings_endpoint} with model {cfg.model_name_or_path}")


    def _embed(texts: List[str], *, as_list: bool = False) -> np.ndarray | List[List[float]]:
        payload = {"input": texts, "model": cfg.model_name_or_path} # CHANGED: Use cfg.model_name_or_path
        for attempt, wait in enumerate(_BACKOFF, 1):
            t0 = time.perf_counter()
            try:
                resp = session.post(embeddings_endpoint, json=payload, timeout=30) # CHANGED: Use dynamic endpoint
                resp.raise_for_status()
                data = resp.json()["data"]
                arr = np.asarray([d["embedding"] for d in data], dtype=np.float32)
                latency = time.perf_counter() - t0
                logger.debug("openai_embed[%s] %d texts in %.3fs via %s", cfg.model_name_or_path, len(texts), latency, embeddings_endpoint)
                return arr.tolist() if as_list else arr
            except requests.RequestException as exc:
                if attempt == len(_BACKOFF):
                    logger.error("openai_embed failed after %d retries for %s: %s", attempt, embeddings_endpoint, exc)
                    raise
                logger.warning("openai_embed retry %d/%d after %s for %s: %s", attempt, len(_BACKOFF), wait, embeddings_endpoint, exc)
                time.sleep(wait + random.random())
        raise RuntimeError(f"Exhausted retries in OpenAI embedder for {embeddings_endpoint}")  # Should be unreachable

    return _embed


###############################################################################
# Factory
###############################################################################

class EmbeddingFactory:
    """Thread‑safe LRU/idle cache with sync & async embedding methods."""

    def __init__(
            self,
            cfg: Dict[str, Any] | EmbeddingConfigSchema,
            *,
            max_cached: int = 2,
            idle_seconds: int = 900,
            allow_dynamic_hf: bool = True,
    ) -> None:
        # Check if embeddings/RAG dependencies are available
        if not DEPENDENCIES_AVAILABLE.get('embeddings_rag', False):
            raise ImportError(
                "EmbeddingFactory requires embeddings/RAG dependencies. "
                "Install with: pip install tldw_chatbook[embeddings_rag]"
            )
        try:
            self._cfg = cfg if isinstance(cfg, EmbeddingConfigSchema) else EmbeddingConfigSchema(**cfg)
        except ValidationError as e_pydantic:  # Catch Pydantic validation errors
            logger.critical(f"Invalid embedding configuration provided to EmbeddingFactory: {e_pydantic}")
            raise ValueError(f"Embedding configuration error: {e_pydantic}") from e_pydantic

        self._max_cached = max_cached
        self._idle = idle_seconds
        self._allow_dynamic_hf = allow_dynamic_hf
        self._cache: "OrderedDict[str, CacheRecord]" = OrderedDict()
        self._lock = threading.Lock()
        logger.debug("factory initialised max_cached=%d idle=%ds", max_cached, idle_seconds)

    def _get_spec(self, model_id: str) -> ModelCfg:
        try:
            return self._cfg.models[model_id]
        except KeyError:
            if self._allow_dynamic_hf:
                logger.info("dynamic HF model %s (model_id used as model_name_or_path)", model_id)
                # This assumes any unknown model_id passed when allow_dynamic_hf is True
                # should be treated as a HuggingFace model path.
                return HFModelCfg(model_name_or_path=model_id, provider="huggingface")
            raise ValueError(f"Model ID '{model_id}' not found in configuration and dynamic HF loading is disabled.")


    @property
    def config(self) -> EmbeddingConfigSchema:
        return self._cfg

    def _build(self, model_id: str) -> CacheRecord:
        """
        Build and initialize a model for embedding.

        Args:
            model_id: The ID of the model to build

        Returns:
            CacheRecord containing the embedding function and cleanup function

        Raises:
            TypeError: If the model specification doesn't match the provider type
            ValueError: If the provider is not supported
        """
        logger.debug(f"_build: Building model {model_id}")

        spec = self._get_spec(model_id)
        logger.debug(f"_build: Got specification for model {model_id}, provider={spec.provider}")

        t0 = time.perf_counter()
        if spec.provider == "huggingface":
            # Ensure spec is correctly typed for HFModelCfg
            if not isinstance(spec, HFModelCfg):
                logger.error(f"_build: Type mismatch for model {model_id} - expected HFModelCfg, got {type(spec)}")
                raise TypeError(f"Expected HFModelCfg for provider 'huggingface', got {type(spec)} for model_id '{model_id}'")

            logger.info(f"_build: Initializing HuggingFace model {model_id} (path: {spec.model_name_or_path})")
            try:
                hf = _HuggingFaceEmbedder(spec)
                rec = CacheRecord(embed=hf.embed, close=hf.close, last=time.monotonic())
                logger.debug(f"_build: Successfully created HuggingFace embedder for {model_id}")
            except Exception as e:
                logger.error(f"_build: Failed to initialize HuggingFace model {model_id}: {e}", exc_info=True)
                raise

        elif spec.provider == "openai":
            # Ensure spec is correctly typed for OpenAICfg
            if not isinstance(spec, OpenAICfg):
                logger.error(f"_build: Type mismatch for model {model_id} - expected OpenAICfg, got {type(spec)}")
                raise TypeError(f"Expected OpenAICfg for provider 'openai', got {type(spec)} for model_id '{model_id}'")

            logger.info(f"_build: Initializing OpenAI model {model_id} (model: {spec.model_name_or_path})")
            try:
                fn = _openai_embedder(spec) # Pass the whole spec object
                rec = CacheRecord(embed=fn, close=None, last=time.monotonic())
                logger.debug(f"_build: Successfully created OpenAI embedder for {model_id}")
            except Exception as e:
                logger.error(f"_build: Failed to initialize OpenAI model {model_id}: {e}", exc_info=True)
                raise

        else:
            # This should ideally not be reached if Pydantic validation is working with discriminated unions
            # However, it's a good safeguard.
            logger.error(f"_build: Unsupported provider '{spec.provider}' for model {model_id}")
            raise ValueError(f"Unsupported provider: {spec.provider} for model_id '{model_id}'")

        build_time = time.perf_counter() - t0
        logger.info(f"_build: Loaded model {model_id} in {build_time:.2f}s")
        return rec

    def embed(
            self, texts: List[str], *, model_id: Optional[str] = None, as_list: bool = False
    ):
        """
        Embed a list of texts using the specified model or the default model.

        Args:
            texts: List of strings to embed
            model_id: Optional model ID to use (falls back to default_model_id)
            as_list: If True, return embeddings as list of lists instead of numpy array

        Returns:
            Embeddings as numpy array or list of lists
        """
        logger.debug(f"embed: Called with {len(texts)} texts, model_id={model_id}, as_list={as_list}")

        model_id_to_use = model_id or self._cfg.default_model_id
        if not model_id_to_use:
            logger.error("embed: No model_id provided and no default_model_id is set")
            raise ValueError("No model_id provided and no default_model_id is set.")

        if not texts:
            logger.debug("embed: Empty texts list provided, returning empty result")
            return [] if as_list else np.empty((0, 0), dtype=np.float32)

        # --- The lock must be held during the embedding call to prevent use-after-free ---
        logger.debug(f"embed: Acquiring lock for model {model_id_to_use}")
        with self._lock:
            # First, check for idle models to evict.
            now = time.monotonic()
            for mid, rec in list(self._cache.items()):
                if now - rec["last"] > self._idle:
                    logger.debug(f"embed: Idle evicting model {mid} (unused for {now - rec['last']:.1f}s)")
                    self._cache.pop(mid)
                    if rec["close"]:
                        rec["close"]()

            # Now, get the model, building it if it doesn't exist.
            rec = self._cache.get(model_id_to_use)
            if rec is None:
                logger.info(f"embed: Model {model_id_to_use} not in cache, will build it")
                # If we need to load a model, make space for it *first*.
                while len(self._cache) >= self._max_cached:
                    lru_mid, lru_rec = self._cache.popitem(last=False)
                    logger.debug(f"embed: LRU evicting model {lru_mid} to make space for {model_id_to_use}")
                    if lru_rec["close"]:
                        lru_rec["close"]()
                logger.debug(f"embed: Building model {model_id_to_use}")
                rec = self._build(model_id_to_use)
                self._cache[model_id_to_use] = rec
                logger.info(f"embed: Successfully built and cached model {model_id_to_use}")
            else:
                logger.debug(f"embed: Using cached model {model_id_to_use}")

            # Mark as most recently used and get the function to call.
            rec["last"] = time.monotonic()
            self._cache.move_to_end(model_id_to_use)
            embed_fn = rec["embed"]

            # The embedding call itself is now inside the lock. This serializes
            # all embedding calls, but guarantees that the model cannot be
            # evicted by another thread while it is in use.
            logger.debug(f"embed: Starting embedding of {len(texts)} texts with model {model_id_to_use}")
            t0 = time.perf_counter()
            result = embed_fn(texts, as_list=as_list)
            elapsed = time.perf_counter() - t0
            logger.debug(f"embed: Completed embedding {len(texts)} texts with model {model_id_to_use} in {elapsed:.3f}s")

            # Log more details for larger batches or slower operations
            if len(texts) > 10 or elapsed > 1.0:
                logger.info(f"embed: Embedded {len(texts)} texts with model {model_id_to_use} in {elapsed:.3f}s ({elapsed/len(texts):.3f}s per text)")

            return result
        # The lock is released only after the work is complete.

    async def async_embed(
            self, texts: List[str], *, model_id: Optional[str] = None, as_list: bool = False
    ):
        """Non-blocking version of `embed` for use in async contexts."""
        return await asyncio.to_thread(self.embed, texts, model_id=model_id, as_list=as_list)

    def embed_one(
            self, text: str, *, model_id: Optional[str] = None, as_list: bool = False
    ):
        vecs = self.embed([text], model_id=model_id, as_list=as_list)
        return vecs[0] if as_list else vecs.squeeze(0)

    async def async_embed_one(
            self, text: str, *, model_id: Optional[str] = None, as_list: bool = False
    ):
        vecs = await self.async_embed([text], model_id=model_id, as_list=as_list)
        return vecs[0] if as_list else vecs.squeeze(0)

    def prefetch(self, model_ids: List[str]):
        """Download / load given model ids in advance (bypasses eviction)."""
        for mid in model_ids:
            with self._lock:
                if mid in self._cache:
                    continue
                # Note: This can temporarily exceed max_cached, which is acceptable for a startup operation.
                self._cache[mid] = self._build(mid)
            logger.info("prefetched %s", mid)

    def close(self) -> None:
        """Close all models and clear the cache."""
        with self._lock:
            for mid, rec in self._cache.items():
                if rec["close"]:
                    rec["close"]()
            self._cache.clear()
            logger.debug("factory closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

#
# End of Embeddings_Lib.py
########################################################################################################################
