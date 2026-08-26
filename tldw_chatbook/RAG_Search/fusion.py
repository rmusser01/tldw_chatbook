"""
Hybrid retrieval fusion: Reciprocal Rank Fusion with alpha weighting.

This is the single authoritative implementation of hybrid (FTS + vector)
result fusion for the chatbook, mirroring the tldw_server design
(tldw_Server_API .. rag_service/database_retrievers.py, RRF k=60 plus an
alpha-weighted blend of the two per-leg RRF scores):

    fts_rrf(doc)    = 1 / (k + rank_in_fts_leg)      # rank starts at 1
    vector_rrf(doc) = 1 / (k + rank_in_vector_leg)   # rank starts at 1
    final(doc)      = (1 - alpha) * fts_rrf + alpha * vector_rrf

Documents missing from a leg contribute 0 from that leg.

Alpha semantics (server-consistent): ``alpha`` weights the VECTOR side.
``alpha = 0`` -> FTS-only ordering, ``alpha = 1`` -> vector-only ordering.
The server default is ``alpha = 0.7`` (vector-weighted), ``k = 60``.

The fusion math is rank-based: only each leg's returned ordering matters,
never the leg's raw scores. This makes fusion robust to the incomparable
score scales of SQLite FTS5 and cosine similarity.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, List, Optional, Sequence, TypeVar

from loguru import logger

__all__ = [
    "DEFAULT_RRF_K",
    "DEFAULT_HYBRID_ALPHA",
    "FusedResult",
    "reciprocal_rank_fusion",
    "interleave_rankings",
    "resolve_hybrid_alpha",
    "resolve_rrf_k",
]

# Server-parity constants (tldw_server database_retrievers.py)
DEFAULT_RRF_K = 60
DEFAULT_HYBRID_ALPHA = 0.7

T = TypeVar("T")


@dataclass
class FusedResult:
    """A single fused result with per-leg provenance.

    Attributes:
        key: Deduplication key shared by both legs.
        fts_item: The item as returned by the FTS leg, if present there.
        vector_item: The item as returned by the vector leg, if present there.
        fts_rank: 1-based rank in the FTS leg, or None if absent.
        vector_rank: 1-based rank in the vector leg, or None if absent.
        fts_rrf: RRF contribution of the FTS leg (0.0 if absent).
        vector_rrf: RRF contribution of the vector leg (0.0 if absent).
        score: Final fused score: (1 - alpha) * fts_rrf + alpha * vector_rrf.
    """

    key: Hashable
    fts_item: Optional[Any]
    vector_item: Optional[Any]
    fts_rank: Optional[int]
    vector_rank: Optional[int]
    fts_rrf: float
    vector_rrf: float
    score: float

    @property
    def item(self) -> Any:
        """Primary item for this key (FTS leg wins, matching the server)."""
        return self.fts_item if self.fts_item is not None else self.vector_item

    def provenance(self) -> Dict[str, Any]:
        """Cheap leg-provenance dict suitable for stashing in metadata."""
        return {
            "fts_rank": self.fts_rank,
            "vector_rank": self.vector_rank,
            "fts_rrf": self.fts_rrf,
            "vector_rrf": self.vector_rrf,
        }


def _leg_ranks(
    results: Sequence[T],
    key: Callable[[T], Hashable],
) -> Dict[Hashable, tuple]:
    """Map key -> (1-based rank, item) for a leg, keeping the best rank on dupes."""
    ranks: Dict[Hashable, tuple] = {}
    for rank, item in enumerate(results, start=1):
        k = key(item)
        if k not in ranks:  # keep the best (earliest) rank for duplicates
            ranks[k] = (rank, item)
    return ranks


def reciprocal_rank_fusion(
    fts_results: Sequence[T],
    vector_results: Sequence[T],
    *,
    key: Callable[[T], Hashable],
    alpha: float = DEFAULT_HYBRID_ALPHA,
    rrf_k: int = DEFAULT_RRF_K,
    max_results: Optional[int] = None,
) -> List[FusedResult]:
    """Fuse an FTS ranking and a vector ranking via RRF + alpha blend.

    Pure function; the input orderings are treated as the rankings (best
    first) and input items are never mutated.

    Args:
        fts_results: FTS/keyword leg results, best first.
        vector_results: Vector/semantic leg results, best first.
        key: Extracts the deduplication key (document identity) from an item.
        alpha: Weight of the vector leg; 0 = FTS only, 1 = vector only.
        rrf_k: RRF constant (typically 60).
        max_results: Optional cap on the number of fused results returned.

    Returns:
        FusedResult list sorted by fused score descending. Ties break
        deterministically: better FTS rank first, then better vector rank
        (absent legs sort last).

    Note:
        As a last-resort invariant (config-sourced values should already be
        validated via resolve_hybrid_alpha / resolve_rrf_k), out-of-range
        inputs are sanitized rather than propagated into the math: alpha is
        clamped into [0, 1] and a negative rrf_k falls back to
        DEFAULT_RRF_K, so no call site can produce negative blend weights
        or a zero/negative RRF denominator.
    """
    if not 0.0 <= alpha <= 1.0:
        clamped = min(1.0, max(0.0, alpha))
        logger.warning(
            f"reciprocal_rank_fusion: alpha {alpha} outside [0, 1]; clamping to {clamped}"
        )
        alpha = clamped
    if rrf_k < 0:
        logger.warning(
            f"reciprocal_rank_fusion: rrf_k {rrf_k} is negative; using {DEFAULT_RRF_K}"
        )
        rrf_k = DEFAULT_RRF_K

    fts_ranks = _leg_ranks(fts_results, key)
    vector_ranks = _leg_ranks(vector_results, key)

    fused: List[FusedResult] = []
    # FTS keys first, then vector-only keys, preserving leg order (deterministic).
    all_keys = list(fts_ranks.keys())
    all_keys.extend(k for k in vector_ranks.keys() if k not in fts_ranks)

    for k in all_keys:
        fts_entry = fts_ranks.get(k)
        vector_entry = vector_ranks.get(k)
        fts_rank = fts_entry[0] if fts_entry else None
        vector_rank = vector_entry[0] if vector_entry else None
        fts_rrf = (1.0 / (rrf_k + fts_rank)) if fts_rank is not None else 0.0
        vector_rrf = (1.0 / (rrf_k + vector_rank)) if vector_rank is not None else 0.0
        fused.append(
            FusedResult(
                key=k,
                fts_item=fts_entry[1] if fts_entry else None,
                vector_item=vector_entry[1] if vector_entry else None,
                fts_rank=fts_rank,
                vector_rank=vector_rank,
                fts_rrf=fts_rrf,
                vector_rrf=vector_rrf,
                score=(1.0 - alpha) * fts_rrf + alpha * vector_rrf,
            )
        )

    infinity = float("inf")
    fused.sort(
        key=lambda f: (
            -f.score,
            f.fts_rank if f.fts_rank is not None else infinity,
            f.vector_rank if f.vector_rank is not None else infinity,
        )
    )

    if max_results is not None:
        fused = fused[:max_results]
    return fused


def interleave_rankings(
    rankings: Sequence[Sequence[T]],
    *,
    key: Callable[[T], Hashable],
) -> List[T]:
    """Round-robin merge several per-source rankings into one ranking.

    Used to collapse the pipeline's parallel FTS5 legs (media, conversations,
    notes) into a single rank-fair FTS leg before fusion: the sources' raw
    FTS5 scores are not comparable, so rank position is the only meaningful
    cross-source signal. Duplicates (by key) keep their earliest position.

    Args:
        rankings: Per-source rankings, each best first.
        key: Extracts the deduplication key from an item.

    Returns:
        A single ranking, best first.
    """
    merged: List[T] = []
    seen: set = set()
    longest = max((len(r) for r in rankings), default=0)
    for position in range(longest):
        for ranking in rankings:
            if position >= len(ranking):
                continue
            item = ranking[position]
            k = key(item)
            if k in seen:
                continue
            seen.add(k)
            merged.append(item)
    return merged


def resolve_hybrid_alpha(explicit: Optional[float] = None) -> float:
    """Resolve the hybrid alpha from the single authoritative config knob.

    Precedence: explicit value -> the active profile's
    ``search.hybrid_alpha`` (via ``resolve_active_rag_config``) ->
    ``DEFAULT_HYBRID_ALPHA`` (0.7, server parity). Semantics: 0 = FTS
    only, 1 = vector only.

    Invalid or out-of-range values fall back to the default with a warning.

    Args:
        explicit: Caller-supplied alpha override, if any.

    Returns:
        A validated alpha in [0.0, 1.0].
    """
    value = explicit
    if value is None:
        try:
            from .simplified.active_config import resolve_active_rag_config

            value = resolve_active_rag_config().search.hybrid_alpha
        except Exception as e:  # config loading must never break search
            logger.warning(f"Could not read hybrid_alpha from active profile: {e}")
            value = None
    if value is None:
        return DEFAULT_HYBRID_ALPHA
    try:
        alpha = float(value)
    except (TypeError, ValueError):
        logger.warning(
            f"Invalid hybrid_alpha {value!r}; falling back to {DEFAULT_HYBRID_ALPHA}"
        )
        return DEFAULT_HYBRID_ALPHA
    if not 0.0 <= alpha <= 1.0:
        logger.warning(
            f"hybrid_alpha {alpha} outside [0, 1]; falling back to {DEFAULT_HYBRID_ALPHA}"
        )
        return DEFAULT_HYBRID_ALPHA
    return alpha


def _shipped_rrf_k() -> int:
    """The app's shipped RRF constant, for every fallback in ``resolve_rrf_k``.

    ``DEFAULT_RRF_K`` (60) is deliberately NOT used as a fallback by the
    config resolver (TASK-4110 Task 5, review round 2). Every input this
    resolver sanitizes is reachable from user configuration --
    ``pipeline_builder_simple`` passes a TOML pipeline's
    ``steps[].config.rrf_k`` and ``RAGService`` passes
    ``config.search.rrf_k``, either of which a hand-edited or
    round-tripped config can turn into ``"oops"``, ``-1`` or ``None`` --
    so falling back to the server-parity 60 would silently revert THAT path
    to the weighting TASK-4110 measured away from, while every other path
    stayed at 5. One resolver, one fallback.

    ``DEFAULT_RRF_K`` remains the no-config fallback of the pure function
    (``reciprocal_rank_fusion``'s own signature default and its
    negative-``rrf_k`` sanitization). That is a different thing: a library
    invariant for a caller who supplied nothing, not an app-config decision,
    and it is unreachable from the live call sites, which always resolve
    through here first.

    Returns:
        ``config.DEFAULT_HYBRID_RRF_K``. Imported lazily -- this module must
        not depend on the config package at import time, since
        ``simplified.config`` imports ``DEFAULT_HYBRID_ALPHA`` from here --
        and defensively: a last resort must itself be total, so an import
        failure degrades to ``DEFAULT_RRF_K`` rather than raising out of a
        function whose whole contract is that it never breaks search.
    """
    try:
        from .simplified.config import DEFAULT_HYBRID_RRF_K

        return DEFAULT_HYBRID_RRF_K
    except Exception:  # pragma: no cover - importing a module constant
        return DEFAULT_RRF_K


def resolve_rrf_k(value: Optional[Any] = None) -> int:
    """Resolve the RRF constant k from an untrusted (config) value.

    Precedence: explicit value -> the active profile's ``search.rrf_k``
    (via ``resolve_active_rag_config``) -> the shipped default
    (``_shipped_rrf_k()``, i.e. ``config.DEFAULT_HYBRID_RRF_K`` = 5).
    Mirrors ``resolve_hybrid_alpha``'s precedence exactly (TASK-4110
    review): the two live fusion call sites --
    ``RAGService._fuse_hybrid_results`` (Library hybrid) and
    ``pipeline_builder_simple._rrf_merge_parallel_results`` (Chat RAG
    hybrid) -- must resolve k the same way they already resolve alpha, or a
    default change moves one live path and silently strands the other.

    THIS IS LOAD-BEARING (TASK-4110 Task 5): the shipped default moved to
    ``SearchConfig.rrf_k = DEFAULT_HYBRID_RRF_K`` (5, measured for
    chatbook's ~20-row candidate window), so the profile fallback is what
    carries the measured value into the legacy pipeline path as well --
    **both live fusion paths moved together**. Unifying or consciously
    diverging the two paths belongs to TASK-3501.

    ``DEFAULT_RRF_K`` (60) is NOT part of this resolver's fallback chain --
    see ``_shipped_rrf_k`` for why. It survives only as
    ``reciprocal_rank_fusion``'s own signature default and that function's
    negative-``rrf_k`` sanitization: a pure-library invariant for a caller
    who supplied nothing, unreachable from the live call sites, which
    always resolve through here first.

    Args:
        value: Caller/config-supplied k, if any (e.g. a TOML pipeline's
            ``steps[].config.rrf_k``). ``None`` means "no explicit
            override" -- the active profile is consulted next, exactly as
            for alpha.

    Returns:
        A non-negative int. **Every** fallback in this resolver is the app's
        shipped default (``config.DEFAULT_HYBRID_RRF_K``, 5): a profile that
        cannot be read, a profile with nothing to say, a non-numeric value
        and a negative value all land there. ``DEFAULT_RRF_K`` (60) is NOT a
        fallback here -- see ``_shipped_rrf_k`` for why. Invalid values are
        logged, never raised: a misconfigured pipeline must not abort search
        at merge time.
    """
    shipped_k = _shipped_rrf_k()
    if value is None:
        try:
            from .simplified.active_config import resolve_active_rag_config

            value = resolve_active_rag_config().search.rrf_k
        except Exception:  # config loading must never break search
            logger.warning(
                "Could not read rrf_k from active profile; using shipped default"
            )
            return shipped_k
    if value is None:
        return shipped_k
    try:
        k = int(float(value))
    except (TypeError, ValueError, OverflowError):
        # OverflowError: `float(value)` can itself land on an infinite
        # float ("1e309" overflows float's range, and TOML accepts a
        # literal `inf`/`-inf`), and `int()` on an infinite float raises
        # OverflowError rather than ValueError -- Qodo PR-1487. Reachable
        # straight from a hand-edited config's `rrf_k`, so it must fall
        # back like every other invalid value, not crash hybrid search.
        logger.warning("Invalid rrf_k; using shipped default")
        return shipped_k
    if k < 0:
        logger.warning(f"rrf_k {k} is negative; falling back to {shipped_k}")
        return shipped_k
    return k
