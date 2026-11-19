"""Plain RAG profile component configuration, independent of runtime engines.

Legacy engine modules reexport these exact classes. Historical pickle module/name
references remain valid through those aliases; new pickles use this module.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple


@dataclass
class RerankingConfig:
    """Configuration for reranking operations."""

    # Model settings
    model_provider: str = "openai"
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.0  # Use deterministic scoring
    # Sized for `{"score": 0.9}`. With `include_reasoning` on it is raised to
    # `REASONING_TOKEN_FLOOR` at the call site (`_effective_max_tokens`) --
    # 100 tokens cannot hold an explanation AND the JSON, and the truncated
    # object is billed but unscorable (TASK-17365).
    max_tokens: int = 100

    # Reranking settings.
    #
    # ``cross_encoder`` is implemented and selectable (TASK-16965) but is
    # NOT a recommendation: measured on the gated instrument it is net
    # harmful on the averaged row and bimodal by category. It is the
    # default of no profile and no config template; pick it only with
    # Docs/superpowers/qa/2026-08-17-cross-encoder/report.md in hand.
    strategy: Literal["pairwise", "listwise", "pointwise", "cross_encoder"] = (
        "pointwise"
    )
    top_k_to_rerank: int = 20  # Only rerank top K results
    batch_size: int = 5  # Number of results to evaluate at once
    include_reasoning: bool = False  # Whether to generate explanations

    # Scoring settings
    score_scale: Tuple[float, float] = (0.0, 1.0)  # Min and max scores
    combine_original_score: bool = True  # Combine with original retrieval score
    original_score_weight: float = 0.3  # Weight for original score (0-1)

    # Prompts
    system_prompt: Optional[str] = None
    scoring_prompt_template: Optional[str] = None

    # Performance settings
    cache_results: bool = True
    timeout_seconds: float = 30.0
    retry_on_failure: bool = True
    max_retries: int = 2


@dataclass
class ProcessingConfig:
    """Configuration for parallel processing."""

    # Worker settings
    num_workers: Optional[int] = None  # None = use CPU count
    max_workers: int = 16  # Maximum workers to prevent resource exhaustion
    min_workers: int = 2  # Minimum workers

    # Batch settings
    batch_size: int = 32
    max_batch_size: int = 256
    dynamic_batching: bool = True

    # Performance settings
    prefetch_factor: int = 2  # Number of batches to prefetch
    timeout_seconds: float = 300.0  # 5 minutes default
    show_progress: bool = True
    progress_interval: int = 10  # Report every N items

    # Memory management
    max_memory_per_worker_mb: float = 1024.0  # 1GB per worker
    monitor_memory: bool = True

    # Error handling
    continue_on_error: bool = True
    max_retries: int = 2
    retry_delay: float = 1.0
