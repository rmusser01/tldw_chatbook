"""
Simplified RAG configuration management.

This module provides configuration handling for the simplified RAG implementation,
integrating with the existing tldw_cli configuration system while providing
sensible defaults and easy overrides.
"""

from dataclasses import dataclass, field, fields, asdict
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import os
from loguru import logger

# Import the main config module to access existing configuration
from tldw_chatbook.config import (
    get_cli_setting,
    load_cli_config_and_ensure_existence,
    get_user_data_dir,
)
from tldw_chatbook.Utils.path_validation import validate_path_simple

# Hybrid fusion defaults. `hybrid_alpha` keeps tldw_server's 0.7 (it weights
# the vector leg); the RRF constant k does NOT keep the server's 60 -- see
# `DEFAULT_HYBRID_RRF_K` below.
from ..fusion import DEFAULT_HYBRID_ALPHA


# Canonical vector store type values used by the selection logic below.
# "auto" is a sentinel meaning "resolve from env/config/installed deps".
VECTOR_STORE_TYPE_AUTO = "auto"
VECTOR_STORE_TYPE_CHROMA = "chroma"
VECTOR_STORE_TYPE_MEMORY = "memory"

# The single source of truth for SearchConfig.hybrid_pool_multiplier's
# default (TASK-4110 review, minor b): `rag_service._resolve_hybrid_pool_
# multiplier`'s invalid-value fallback imports this same constant rather
# than the module-level `SEARCH_RESULT_MULTIPLIER` -- those two used to
# collapse to the same number (2) by coincidence, not by any shared
# definition, so a user who had tuned the undocumented
# `[rag.service] search_result_multiplier` TOML knob (which still governs
# `_semantic_search`'s own internal over-fetch on every search path) would
# have silently gotten THEIR number back out of an invalid
# hybrid_pool_multiplier, rather than this field's own default. Release
# note: hybrid legs previously honored `search_result_multiplier` for their
# over-fetch; they now honor `hybrid_pool_multiplier` instead -- a user who
# set `search_result_multiplier = 4` gets the hybrid legs back to 2 until
# they set `hybrid_pool_multiplier` explicitly.
DEFAULT_HYBRID_POOL_MULTIPLIER = 2

# The shipped RRF constant for chatbook's hybrid fusion (TASK-4110, Task 5).
#
# DELIBERATELY NOT `fusion.DEFAULT_RRF_K` (60). That constant is the
# tldw_server-parity value and survives only as a PURE-LIBRARY no-config
# fallback (`reciprocal_rank_fusion`'s own signature default and its
# negative-k sanitization, plus `_fuse_hybrid_results`' pre-parameter
# default); this one is the value a chatbook profile actually ships with,
# and it is what EVERY fallback in `fusion.resolve_rrf_k` -- the app-config
# resolver both live fusion paths go through -- now returns.
#
# Measured, not asserted (the full matrix is in the TASK-4110 PR): the
# server calibrates k for candidate pools of thousands, while chatbook's
# `_hybrid_search` only ever fuses `top_k * hybrid_pool_multiplier` rows per
# leg -- ~20. Over a 20-row window the k=60 RRF curve is nearly flat, so an
# FTS-only row at keyword rank 1 (score `(1-alpha)/(60+1)` = 0.00492) is
# beaten by every vector-only row down to rank ~83 and can never enter the
# fused top-k: hybrid could not rescue a document the vector leg missed. At
# k=5 an FTS-only rank-1 row strictly outranks vector-only rows from rank 10
# -- well inside the window fusion actually sees. (Rank 9 is the exact
# equality point, `3/10 x 1/6 == 7/10 x 1/14`; the keyword row still ranks
# above it, but by one ULP of float rounding rather than by the weighting, so
# 10 is the honest boundary to quote.)
#
# On the 49-document eval corpus this moved keyword recall@10 0.938 -> 1.000
# and keyword NDCG 0.938 -> 0.957 with no per-category cell regressing.
# That safety half is BOUNDED TO THAT CORPUS -- k=5 makes rank position
# matter more within each leg, which is good for a well-ordered vector leg
# and bad for a noisy one. `hybrid_alpha` (0.7) and
# `hybrid_pool_multiplier` (2) were measured alongside it and deliberately
# left alone: pool widening bought +0.005 on one metric family by re-ranking
# a document k=5 had already rescued, for a permanent +50% retrieval width.
DEFAULT_HYBRID_RRF_K = 5

# Cached result of the embeddings_rag installed-probe. Availability cannot
# change without a restart, so probe at most once per process.
_EMBEDDINGS_RAG_AVAILABLE: Optional[bool] = None


def _embeddings_rag_available() -> bool:
    """Return True when the `embeddings_rag` optional dependencies are installed.

    Wraps `Utils.optional_deps.embeddings_rag_deps_installed()` (a cheap
    `find_spec`-based probe of the 'embeddings_rag' feature group — no heavy
    imports, no side effects) and caches the result. Any failure is treated
    as "not available" so environments without the extras never error.

    Returns:
        True if all embeddings/RAG dependencies are installed.
    """
    global _EMBEDDINGS_RAG_AVAILABLE
    if _EMBEDDINGS_RAG_AVAILABLE is None:
        try:
            from tldw_chatbook.Utils.optional_deps import embeddings_rag_deps_installed

            _EMBEDDINGS_RAG_AVAILABLE = bool(embeddings_rag_deps_installed())
        except Exception as e:
            logger.debug(
                f"embeddings_rag availability check failed, assuming unavailable: {e}"
            )
            _EMBEDDINGS_RAG_AVAILABLE = False
    return _EMBEDDINGS_RAG_AVAILABLE


def _explicit_rag_setting(section: str, key: str) -> Optional[Any]:
    """Read an explicit `[AppRAGSearchConfig.rag.<section>].<key>` user setting.

    Args:
        section: Subsection name within the rag config (e.g. 'vector_store').
        key: Setting name within that subsection (e.g. 'type').

    Returns:
        The configured value, or None when not explicitly set (or on any
        config read failure).
    """
    try:
        rag_section = get_cli_setting("AppRAGSearchConfig", "rag", {}) or {}
        if not isinstance(rag_section, dict):
            return None
        subsection = rag_section.get(section, {})
        if not isinstance(subsection, dict):
            return None
        return subsection.get(key)
    except Exception as e:
        logger.debug(f"Could not read rag.{section}.{key} from user config: {e}")
        return None


def _normalized_type_setting(value: Any) -> Optional[str]:
    """Normalize an explicit vector store type setting.

    Args:
        value: Raw setting value from env/config.

    Returns:
        Stripped, lowercased type string, or None when the value is unset,
        blank, or the "auto" sentinel (which means "run auto-detection").
    """
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized or normalized == VECTOR_STORE_TYPE_AUTO:
        return None
    return normalized


def _cleaned_path_setting(value: Any) -> Optional[str]:
    """Strip an explicit path setting, treating blank values as unset.

    Args:
        value: Raw setting value from env/config.

    Returns:
        Stripped path string, or None when unset or whitespace-only.
    """
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def default_vector_store_type() -> str:
    """Resolve the default vector store type.

    Priority: RAG_VECTOR_STORE env var > explicit
    `[AppRAGSearchConfig.rag.vector_store].type` in user config > persistent
    ChromaDB when the `embeddings_rag` optional deps are installed >
    in-memory fallback. An explicit `type = "memory"` therefore always wins.
    Explicit values are normalized (stripped/lowercased); blank or "auto"
    values fall through to the next source, ending in auto-detection.

    Returns:
        Vector store type string ("chroma" or "memory", or the normalized
        explicit user-configured value).
    """
    explicit = _normalized_type_setting(
        os.getenv("RAG_VECTOR_STORE")
    ) or _normalized_type_setting(_explicit_rag_setting("vector_store", "type"))
    if explicit:
        return explicit
    return (
        VECTOR_STORE_TYPE_CHROMA
        if _embeddings_rag_available()
        else VECTOR_STORE_TYPE_MEMORY
    )


def default_chroma_persist_directory() -> Path:
    """Resolve the default persist directory for the ChromaDB store.

    Priority: RAG_PERSIST_DIR env var > explicit
    `[AppRAGSearchConfig.rag.vector_store].persist_directory` > legacy
    `[AppRAGSearchConfig.rag.chroma].persist_directory` (still honored by
    `RAGConfig.from_settings`) > `<user data dir>/chromadb` (the same
    user-data-dir convention as the application databases). Blank or
    whitespace-only values are treated as unset.

    Returns:
        Path to the ChromaDB persist directory.
    """
    explicit = (
        _cleaned_path_setting(os.getenv("RAG_PERSIST_DIR"))
        or _cleaned_path_setting(
            _explicit_rag_setting("vector_store", "persist_directory")
        )
        or _cleaned_path_setting(
            _explicit_rag_setting("chroma", "persist_directory")
        )  # Legacy location
    )
    if explicit:
        return Path(explicit).expanduser()
    return get_user_data_dir() / "chromadb"


def validate_chroma_persist_directory(persist_directory: Union[str, Path]) -> Path:
    """Validate and normalize a config-sourced Chroma ``persist_directory``.

    Both ``ChromaVectorStore`` (``vector_store.py``) and
    ``collection_indexes._client()`` construct a ``chromadb.PersistentClient``
    against this same directory. chromadb's ``SharedSystemClient`` caches one
    client per persist-directory *string* within a process and raises
    ``ValueError`` if the same directory is requested again with different
    ``Settings`` -- but even a harmless string-normalization difference
    between the two call sites (e.g. one wrapped in ``Path(...)``, one used
    as a raw string) would produce two different path strings for the same
    on-disk directory, defeating that cache and constructing two independent
    clients against it. This function is the single normalization point both
    call sites route through so they always agree on the exact string.

    ``persist_directory`` is config-sourced (TOML setting or ``RAG_PERSIST_DIR``
    env var), not untrusted network input, and is not confined to any single
    base directory -- a user may legitimately point it anywhere. That rules
    out ``path_validation.validate_path``, which requires a base directory
    and would also false-positive on the common default persist directory,
    which lives under a dotted ancestor (``~/.local/share/tldw_cli/...``).
    ``validate_path_simple`` fits instead: it rejects null bytes and other
    dangerous patterns without requiring a base directory or rejecting
    hidden/dotted path segments.

    This function is also the single point both persist_directory PRODUCERS
    route through -- ``active_config._apply_env_overrides`` (the
    ``RAG_PERSIST_DIR`` env-override layer) and ``RAGConfig.from_dict`` (a
    saved/legacy profile's stored JSON) -- not just the two client-
    construction CONSUMERS above. A producer that left a persist_directory
    unexpanded (e.g. a literal ``"~/x"`` string, never ``~``-expanded) would
    hand the consumers a raw value that diverges from what re-running this
    same function on it would produce, reopening the exact collision this
    function exists to close -- just one hop upstream, at config-resolution
    time instead of client-construction time. Idempotent by construction (an
    already-validated ``Path`` re-validates to itself), so calling it again
    downstream on an already-normalized value from a compliant producer is a
    safe no-op, not a second, possibly-divergent transformation.

    Args:
        persist_directory: The configured Chroma persist directory. May use
            ``~`` for the user's home directory.

    Returns:
        The validated, ``~``-expanded ``Path``.

    Raises:
        ValueError: If ``persist_directory`` isn't a str/Path-like value (so
            ``Path(...)``/``.expanduser()`` construction itself fails), or if
            the resulting path contains null bytes or another dangerous
            pattern (see ``validate_path_simple``).
    """
    try:
        expanded = Path(persist_directory).expanduser()
        validate_path_simple(str(expanded))
    except (TypeError, ValueError, OSError) as e:
        raise ValueError(
            f"Invalid Chroma persist_directory {str(persist_directory)!r}: {e}"
        ) from e
    return expanded


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    model: str = "mxbai-embed-large-v1"  # Default to mxbai-embed-large-v1 (high-quality embeddings)
    device: Optional[str] = "auto"  # auto-detect best device
    cache_size: int = 2
    batch_size: int = 16  # Reduced for larger model
    max_length: int = 512
    # For OpenAI or API-based models
    api_key: Optional[str] = None
    base_url: Optional[str] = None


@dataclass
class VectorStoreConfig:
    """Configuration for vector storage.

    The default `type` is the "auto" sentinel, resolved on construction to
    persistent ChromaDB when the `embeddings_rag` optional deps are installed
    (persisting under the user data dir), and the in-memory store otherwise.
    Explicit values — constructor arguments, `RAG_VECTOR_STORE` env var, or
    `[AppRAGSearchConfig.rag.vector_store]` in the user config — always win.
    """

    type: str = VECTOR_STORE_TYPE_AUTO  # resolved in __post_init__: chroma (persistent) with embeddings deps, else memory
    persist_directory: Optional[Path] = None
    collection_name: str = "default"
    distance_metric: str = "cosine"  # "cosine", "l2", "ip"
    # Collection names for different content types
    media_collection: str = "media_embeddings"
    chat_collection: str = "chat_embeddings"
    notes_collection: str = "notes_embeddings"
    character_collection: str = "character_embeddings"

    def __post_init__(self):
        if self.type == VECTOR_STORE_TYPE_AUTO:
            self.type = default_vector_store_type()
        if self.persist_directory is None and self.type == VECTOR_STORE_TYPE_CHROMA:
            self.persist_directory = default_chroma_persist_directory()


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""

    chunk_size: int = 400  # in words
    chunk_overlap: int = 100  # in words
    chunking_method: str = "words"  # "words", "sentences", "paragraphs"
    min_chunk_size: int = 50  # minimum words per chunk
    max_chunk_size: int = 1000  # maximum words per chunk

    # Parent document retrieval settings
    enable_parent_retrieval: bool = False
    parent_size_multiplier: int = 3  # Parent chunks are this many times larger

    # Structural chunking settings
    preserve_structure: bool = False
    clean_artifacts: bool = False  # Clean PDF artifacts
    preserve_tables: bool = False  # Serialize tables for better understanding


@dataclass
class SearchConfig:
    """Configuration for search operations."""

    default_top_k: int = 10
    score_threshold: float = 0.0
    include_citations: bool = True
    citation_style: str = "inline"  # "inline", "footnote", or "none"
    snippet_max_chars: int = 240
    # Search mode
    default_search_mode: str = "semantic"  # "plain", "semantic", or "hybrid"
    # Search-specific settings
    fts_top_k: int = 10  # For keyword search
    vector_top_k: int = 10  # For semantic search
    # Hybrid fusion alpha: weight of the vector leg in the RRF blend
    # (0 = FTS/keyword only, 1 = vector/semantic only). Default 0.7 matches
    # tldw_server. Authoritative TOML knob:
    # [AppRAGSearchConfig.rag.retriever] hybrid_alpha
    hybrid_alpha: float = DEFAULT_HYBRID_ALPHA
    # Hybrid fusion RRF constant k: the rank-fusion denominator
    # (1 / (k + rank)). Default 5 -- measured for chatbook's ~20-row
    # candidate window, NOT tldw_server's 60 (see
    # `DEFAULT_HYBRID_RRF_K` above for the measurement and the divergence).
    # Not range-checked here (this dataclass has no active load-time
    # validation -- see `hybrid_alpha` above); resolved at USE time via
    # `fusion.resolve_rrf_k`, exactly like `hybrid_alpha` is resolved via
    # `resolve_hybrid_alpha` at its call site -- an invalid/negative value
    # falls back to `DEFAULT_HYBRID_RRF_K` (this field's own default, 5)
    # with a warning rather than distorting or crashing the fusion math.
    rrf_k: int = DEFAULT_HYBRID_RRF_K
    # Hybrid leg over-fetch multiplier: `_hybrid_search` asks each of its
    # two legs (semantic, keyword) for `top_k * hybrid_pool_multiplier`
    # candidates before RRF narrows back down to `top_k` -- a wider pool
    # gives fusion more overlap between the legs to find. Scoped to the
    # HYBRID legs only: `_semantic_search`'s own internal over-fetch
    # multiplier (used on both the hybrid and the direct semantic-search
    # path) is the separate module-level `SEARCH_RESULT_MULTIPLIER`
    # constant and is untouched by this field. Resolved at use time via
    # `rag_service._resolve_hybrid_pool_multiplier`: floored to 1 (each leg
    # must fetch at least `top_k`), capped at a sanity ceiling, and an
    # invalid value falls back to `DEFAULT_HYBRID_POOL_MULTIPLIER` (2,
    # matching the prior shared `SEARCH_RESULT_MULTIPLIER` behavior
    # byte-for-byte at THIS field's own default -- see
    # `DEFAULT_HYBRID_POOL_MULTIPLIER`'s docstring above for the disclosure
    # on the two knobs no longer being the same one).
    #
    # NOTE for the Task 4 sweep: this does not set the semantic leg's total
    # effective over-fetch alone -- `_semantic_search` applies ITS OWN
    # `SEARCH_RESULT_MULTIPLIER` on top of whatever top_k it is handed, so
    # the semantic leg's raw vector-store fetch is
    # `top_k * hybrid_pool_multiplier * SEARCH_RESULT_MULTIPLIER`
    # (compounding), while the keyword leg's fetch is the simple
    # `top_k * hybrid_pool_multiplier` (no second multiplier applies there).
    hybrid_pool_multiplier: int = DEFAULT_HYBRID_POOL_MULTIPLIER
    # Re-ranking
    enable_reranking: bool = False
    reranker_model: Optional[str] = None
    reranker_top_k: int = 5
    # Cache settings
    enable_cache: bool = True
    cache_size: int = 100
    cache_ttl: float = 3600  # 1 hour in seconds (default for all search types)
    # Search-type specific cache TTLs (optional)
    semantic_cache_ttl: Optional[float] = None  # TTL for semantic search results
    keyword_cache_ttl: Optional[float] = None  # TTL for keyword search results
    hybrid_cache_ttl: Optional[float] = None  # TTL for hybrid search results
    # Database connection settings
    fts5_connection_pool_size: int = 3  # Connection pool size for FTS5 searches
    # Explicit override for the keyword (FTS5) leg's media database path.
    # None (the default) means "resolve via tldw_chatbook.config.get_media_db_path()"
    # -- the single authoritative resolver for the real on-disk media DB
    # (honors TLDW_CONFIG_PATH scratch profiles and any user-configured
    # custom path). Only set this to point the keyword leg at a specific
    # file (e.g. tests); never guessed/derived from other paths.
    media_db_path: Optional[Path] = None
    # Explicit override for the keyword (FTS5) leg's ChaChaNotes database
    # path -- the source of the notes and conversation sub-legs (TASK-3996).
    # None (the default) means "resolve via
    # tldw_chatbook.config.get_chachanotes_db_path()", the single
    # authoritative resolver, exactly as `media_db_path` above defers to
    # `get_media_db_path()`. The engine opens this file READ-ONLY and never
    # through `CharactersRAGDB` (whose constructor runs schema work); only
    # set this to point the keyword leg at a specific file (e.g. tests).
    chachanotes_db_path: Optional[Path] = None
    # Explicit override for the keyword (FTS5) leg's Prompts database path --
    # the source of the prompts sub-leg (TASK-15020/B2). None (the default)
    # means "resolve via tldw_chatbook.config.get_prompts_db_path()", the
    # single authoritative resolver, exactly as the two paths above defer to
    # theirs. The engine opens this file READ-ONLY and never through
    # `PromptsDatabase` (whose constructor runs schema work); only set this to
    # point the keyword leg at a specific file (e.g. tests).
    prompts_db_path: Optional[Path] = None
    # How the keyword (FTS5) leg joins a query's tokens into its MATCH
    # expression (TASK-15400, extended by TASK-15700). One of the six
    # candidates the two arcs' specs pre-register, resolved at USE time by
    # `rag_service.RAGService._fts5_match_expressions`:
    #
    #   "and"                -- the PRE-15400 construction: implicit AND over
    #                           EVERY token (a document must contain every
    #                           word the user typed).
    #   "and_stopword_trim"  -- AND over the content tokens only, falling back
    #                           to the full AND when trimming empties the
    #                           query. THE SHIPPED DEFAULT 2026-08-11 ->
    #                           2026-08-13.
    #   "or"                 -- OR over the content tokens.
    #   "and_then_or"        -- AND first; on a ZERO-row sub-leg result,
    #                           that sub-leg re-runs as the content-token
    #                           OR (a non-empty AND is never widened).
    #   "prefix"             -- the content tokens as PREFIX terms, implicitly
    #                           ANDed (`"tok"*`). Widens as the PRIMARY form.
    #   "and_then_prefix"    -- THE SHIPPED DEFAULT since 2026-08-13: AND
    #                           first; on a ZERO-row sub-leg result, that
    #                           sub-leg re-runs as the content-token PREFIX
    #                           form (a non-empty AND is never widened).
    #
    # THE DECISION RECORD FOR THIS DEFAULT (TASK-15700 Task 4, 2026-08-13).
    # Two sentences, and the second is NOT the first's conclusion:
    #
    # (1) WHAT THE RULE PRODUCED. The arc's pre-registered rule was applied
    # VERBATIM to the six-row re-run matrix (Task 3, 2026-08-13): the
    # census-maximal row `and_then_or` (29) was DISQUALIFIED on hard
    # constraint (b) -- 8 gated cells past 0.02, 5 of them past the 0.05 fail
    # band; `or` failed (a) and (b); the qualifying set was
    # {`and_stopword_trim` 21, `prefix` 23, `and_then_prefix` 23}, so max
    # census 23 TIED `prefix` against `and_then_prefix`. The two were verified
    # MEASUREMENT-IDENTICAL on every captured axis -- all 105 gated cells
    # unmoved, all 60 per-query hybrid top-10s and all 60 keyword-leg top-10s
    # identical, the same rescued queries, `lost` 0 both ways. The rule's
    # tie-break (fewest extra FTS statements, MEASURED at 240 vs 460 over the
    # 60-query golden set) therefore selected **`prefix`**.
    #
    # (2) THE OWNER RULED `and_then_prefix` SHIPS INSTEAD. The standing
    # stability-over-quick-wins ruling was applied to a dimension the
    # tie-break PREDATES: structural immunity to intra-sub-leg
    # self-displacement. `prefix` widens as the PRIMARY form, so its widened
    # rows compete for their own sub-leg's bm25-ordered, LIMITED slots BEFORE
    # the merge is consulted, and the tiered merge can protect nothing there
    # (measured synthetically: 12 prefix-competitor docs + 1 exact-match doc,
    # "wombat log" at top_k=5 -- the trimmed AND finds the exact doc, `prefix`
    # returns 5 rows without it). `and_then_prefix` cannot reach that shape: a
    # NON-EMPTY AND primary is never widened, and the widening rows are
    # confined to tier 2 of the sub-leg merge. The measured price is 220 extra
    # SQLite statements over the 60-query set (460 vs 240; 92% of sub-legs
    # fall back on the 172-document eval corpus -- an upper bound that
    # shrinks as a corpus densifies), wall time indistinguishable, and ZERO
    # measured retrieval difference.
    #
    # So `and_then_prefix` is NOT the rule's own output and must never be
    # described as such: the rule produced `prefix`, and the owner overrode
    # the tie-break between two measurement-identical qualifiers.
    #
    # What the flip buys is LEG-LEVEL: keyword-leg census 21 -> 23 of the 53
    # non-negative golden queries (+`kw-quillon-mast`, +`kw-thimble-relay`),
    # zero-row queries 39 -> 36 of 60. NO gated cell moves in any mode (0 of
    # 105), because both new census hits are queries the vector leg already
    # ranks highly -- the gain is what matters when the vector leg is blind,
    # absent or scoped away.
    #
    # NOT A SUPERSET BY CONSTRUCTION -- read this before claiming one. This
    # construction's PRIMARY is the FULL AND (`_escape_fts5_query`, every
    # token INCLUDING stopwords), not the trimmed AND the outgoing default
    # ran. Where a sub-leg's full AND returns rows, the fallback never fires,
    # so `and_stopword_trim`'s trim-only hits are not sought there. That
    # nothing was lost is a MEASURED fact on this corpus (Task 3's `lost` = 0
    # against both the control and the shipped row), never a structural
    # guarantee. `pm-vendor-chaser` illustrates it: the outgoing default
    # reached it by TRIMMING, this one reaches it by the PREFIX FALLBACK --
    # one query, two mechanisms, which is why the gated prompt cell (0.200)
    # does not move across the flip.
    #
    # Deliberately NOT wired to TOML/user config: both arcs measured their
    # candidates against the golden set and ship a default, rather than
    # handing users an unmeasured knob (spec: "the construction choice is NOT
    # a config knob in this arc"). It is a field rather than a constant only
    # so the sweep can vary it on a live SearchConfig -- which is exactly why
    # it also joins the cache key (`simple_cache._make_key`); a
    # runtime-variable retrieval parameter outside the key is how TASK-4110's
    # sweep would have reported "the parameter doesn't matter". A consequence
    # of BOTH flips, unchanged in kind: the cache key is VALUE-keyed on this
    # field, so the default now renders a `fts:and_then_prefix` key part where
    # it rendered `fts:and_stopword_trim` before 2026-08-13 and nothing at all
    # pre-15400. Entries cached under a previous construction are keyed APART
    # from the new one rather than invalidated in place -- correct by
    # construction, at the cost of a one-time run of cold misses after the
    # upgrade, which is pre-existing semantics and not new to this flip.
    #
    # That guarantee holds for the ASYNC cache path only. `SimpleRAGCache`'s
    # SYNC twins (`_sync_get_impl`/`_sync_put_impl`) never pass this
    # parameter, so they still render the legacy construction-less key --
    # and before the flip that key was CORRECT for a default-config search,
    # while after it a sync-path entry is labelled as if the full AND
    # produced it. No production code calls the sync API today (verified by
    # grep, and re-verified at the close of TASK-15400), which is the only
    # reason this is a note rather than a defect; wiring anything to the
    # sync twins requires passing the construction first. Escalated as
    # **TASK-15701**, which covers all three dimensions the sync key omits
    # (this one, `keyword_source_types` and `hybrid_fusion`) rather than
    # only the construction.
    #
    # An unrecognized value warns once and behaves as "and" (fail-safe to the
    # PRE-ARC behaviour, which is the one every escaping/pushdown pin still
    # describes), matching how `hybrid_alpha`/`rrf_k` degrade.
    #
    # TASK-15700 (2026-08-13) added the last TWO values -- `prefix` and
    # `and_then_prefix` -- for its re-run of the sweep under the form-tiered
    # sub-leg merge, and the re-run was NOT null: it moved this default, by
    # the decision recorded above (the rule's winner `prefix`, overridden to
    # `and_then_prefix` by owner ruling).
    fts_match_construction: str = "and_then_prefix"

    # Maximum total context size in characters. LIVE: the Settings > Library >
    # RAG defaults screen reads and writes it (settings_library_rag_defaults.py).
    #
    # RETIRED HERE (TASK-16174, Phase K): `include_parent_docs`,
    # `parent_size_threshold` and `parent_inclusion_strategy` used to sit on this
    # block. They were shipped, user-switchable, switched ON by three profiles --
    # and read by NOTHING (grep-verified: 1 definition + 3 profile writes, 0
    # reads). The decision rule was pre-registered in the arc's spec: wire them
    # only if the capability lands engine-side. It did not -- expansion ships as a
    # pull-based agent tool that runs AFTER retrieval, so wiring them would have
    # changed what retrieval returns, which this arc's gate forbids. See
    # Docs/superpowers/specs/2026-08-15-rag-agentic-expansion-design.md.
    #
    # Saved configs that still carry the retired keys keep loading: `from_dict`
    # drops unknown search keys with a logged notice instead of raising TypeError.
    max_context_size: int = 16000


@dataclass
class QueryExpansionConfig:
    """Configuration for query expansion/rewriting.

    NOTE (task-252): The QueryExpander module (RAG_Search/query_expansion.py) that
    consumed this config was removed as dead code. This dataclass is intentionally
    kept so the [AppRAGSearchConfig.rag.query_expansion] TOML section and RAGConfig
    to_dict/from_dict round-trips keep working; no runtime code performs query
    expansion with it today.
    """

    enabled: bool = False
    method: str = "llm"  # "llm", "local_llm", "llamafile", "keywords"
    max_sub_queries: int = 3
    llm_provider: str = "openai"  # Which LLM provider to use
    llm_model: str = "gpt-3.5-turbo"  # Model for query expansion
    local_model: str = "Qwen3-0.6B-Q6_K.gguf"  # For Ollama/local models/llamafile
    expansion_prompt_template: str = "default"  # Template name or custom prompt
    combine_results: bool = True  # Combine results from all sub-queries
    cache_expansions: bool = True  # Cache expanded queries


@dataclass
class PipelineConfig:
    """Configuration for pipeline selection and behavior."""

    default_pipeline: str = "hybrid"
    enable_pipeline_metrics: bool = True
    pipeline_timeout_seconds: float = 30.0
    max_concurrent_pipelines: int = 3
    cache_pipeline_results: bool = True
    pipeline_config_file: Optional[Path] = None

    # Pipeline-specific overrides
    pipeline_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class RAGConfig:
    """Complete RAG configuration."""

    # Component configurations
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    query_expansion: QueryExpansionConfig = field(default_factory=QueryExpansionConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    # Convenience shortcuts for common settings
    @property
    def embedding_model(self) -> str:
        return self.embedding.model

    @property
    def vector_store_type(self) -> str:
        return self.vector_store.type

    @property
    def persist_directory(self) -> Optional[Path]:
        return self.vector_store.persist_directory

    @property
    def collection_name(self) -> str:
        return self.vector_store.collection_name

    @property
    def distance_metric(self) -> str:
        return self.vector_store.distance_metric

    @property
    def chunk_size(self) -> int:
        return self.chunking.chunk_size

    @property
    def chunk_overlap(self) -> int:
        return self.chunking.chunk_overlap

    @property
    def chunking_method(self) -> str:
        return self.chunking.chunking_method

    @property
    def default_top_k(self) -> int:
        return self.search.default_top_k

    @property
    def score_threshold(self) -> float:
        return self.search.score_threshold

    @property
    def include_citations(self) -> bool:
        return self.search.include_citations

    @property
    def device(self) -> Optional[str]:
        return self.embedding.device

    @property
    def embedding_cache_size(self) -> int:
        return self.embedding.cache_size

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RAGConfig":
        """Create configuration from dictionary."""
        # Extract sub-configurations
        embedding_data = data.get("embedding", {})
        vector_store_data = data.get("vector_store", {})
        chunking_data = data.get("chunking", {})
        search_data = data.get("search", {})
        query_expansion_data = data.get("query_expansion", {})
        pipeline_data = data.get("pipeline", {})

        # Handle path conversion for persist_directory. Routed through the
        # SAME validate_chroma_persist_directory() the two Chroma client-
        # construction sites use (not a bare Path(...)) -- a saved/legacy
        # profile JSON is a persist_directory PRODUCER, and a stored "~/x"
        # left unexpanded here would reach the consumers as a literal,
        # un-expanded path string that diverges from what they'd compute
        # themselves. See validate_chroma_persist_directory's docstring.
        if (
            "persist_directory" in vector_store_data
            and vector_store_data["persist_directory"]
        ):
            vector_store_data["persist_directory"] = validate_chroma_persist_directory(
                vector_store_data["persist_directory"]
            )

        # Handle path conversion for pipeline_config_file
        if (
            "pipeline_config_file" in pipeline_data
            and pipeline_data["pipeline_config_file"]
        ):
            pipeline_data["pipeline_config_file"] = Path(
                pipeline_data["pipeline_config_file"]
            )

        # `search_data` comes straight from a user-editable TOML/profile JSON, so
        # an unknown key here is a hostile-dict problem, not a programming error:
        # a plain `SearchConfig(**search_data)` raises TypeError and takes the
        # whole config load down. That is exactly what a config saved BEFORE
        # TASK-16174 retired `include_parent_docs` / `parent_size_threshold` /
        # `parent_inclusion_strategy` would do. Drop unknown keys with a notice
        # naming each one, so a retired or mistyped key degrades to "ignored and
        # reported". (Same defensive posture as `rag_service.py`'s
        # `_resolve_fts_match_construction`, for the same reason: this dict is
        # user input.) Scope is deliberately the SEARCH section only -- the other
        # sections have no retired fields and are out of this arc's scope.
        known_search_fields = {f.name for f in fields(SearchConfig)}
        known_search_data = {
            key: value
            for key, value in search_data.items()
            if key in known_search_fields
        }
        for dropped in search_data:
            if dropped not in known_search_fields:
                logger.warning(
                    f"Ignoring unknown RAG search config key '{dropped}' "
                    "(retired or misspelled); it has no effect."
                )

        return cls(
            embedding=EmbeddingConfig(**embedding_data),
            vector_store=VectorStoreConfig(**vector_store_data),
            chunking=ChunkingConfig(**chunking_data),
            search=SearchConfig(**known_search_data),
            query_expansion=QueryExpansionConfig(**query_expansion_data),
            pipeline=PipelineConfig(**pipeline_data),
        )

    @classmethod
    def from_settings(
        cls,
        override_embedding_model: Optional[str] = None,
        override_persist_dir: Optional[Union[str, Path]] = None,
    ) -> "RAGConfig":
        """Load the active-profile RAG config + env overrides.

        Delegates to `active_config.resolve_active_rag_config`, which reads the
        active profile's `rag_config` (deep copy) and applies the same
        env/override layer this method used to apply directly. This makes the
        active profile the single config source for both the search path (this
        method) and the ingestion path, so they never diverge.

        Args:
            override_embedding_model: Override the embedding model
            override_persist_dir: Override the persist directory

        Returns:
            RAGConfig instance with resolved settings
        """
        # Imported at call time (not module scope) to avoid a config.py <->
        # active_config.py import cycle (active_config imports RAGConfig from
        # this module).
        from .active_config import resolve_active_rag_config

        return resolve_active_rag_config(override_embedding_model, override_persist_dir)

    def validate(self) -> List[str]:
        """
        Validate the configuration and return any issues.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Validate chunk sizes
        if self.chunking.chunk_size <= 0:
            errors.append("chunk_size must be positive")

        if self.chunking.chunk_overlap >= self.chunking.chunk_size:
            errors.append("chunk_overlap must be less than chunk_size")

        if self.chunking.chunk_overlap < 0:
            errors.append("chunk_overlap cannot be negative")

        # Validate search settings
        if self.search.default_top_k <= 0:
            errors.append("default_top_k must be positive")

        if self.search.score_threshold < 0 or self.search.score_threshold > 1:
            errors.append("score_threshold must be between 0 and 1")

        if self.search.hybrid_alpha < 0 or self.search.hybrid_alpha > 1:
            errors.append("hybrid_alpha must be between 0 and 1")

        # Validate vector store
        if self.vector_store.type not in [
            VECTOR_STORE_TYPE_CHROMA,
            VECTOR_STORE_TYPE_MEMORY,
        ]:
            errors.append(f"Unknown vector store type: {self.vector_store.type}")

        if (
            self.vector_store.type == VECTOR_STORE_TYPE_CHROMA
            and not self.vector_store.persist_directory
        ):
            errors.append("persist_directory is required for chroma vector store")

        if self.vector_store.distance_metric not in ["cosine", "l2", "ip"]:
            errors.append(
                f"Unknown distance metric: {self.vector_store.distance_metric}"
            )

        # Validate embedding settings
        if self.embedding.cache_size <= 0:
            errors.append("embedding cache_size must be positive")

        if self.embedding.batch_size <= 0:
            errors.append("embedding batch_size must be positive")

        return errors


# Convenience functions for common configuration patterns


def create_config_for_collection(
    collection_type: str,
    embedding_model: Optional[str] = None,
    persist_dir: Optional[Union[str, Path]] = None,
) -> RAGConfig:
    """
    Create a RAG configuration for a specific collection type.

    Args:
        collection_type: One of "media", "chat", "notes", "character"
        embedding_model: Optional embedding model override
        persist_dir: Optional persist directory override

    Returns:
        RAGConfig configured for the specified collection
    """
    config = RAGConfig.from_settings(embedding_model, persist_dir)

    # Set the appropriate collection name based on type
    collection_map = {
        "media": config.vector_store.media_collection,
        "chat": config.vector_store.chat_collection,
        "notes": config.vector_store.notes_collection,
        "character": config.vector_store.character_collection,
    }

    if collection_type in collection_map:
        config.vector_store.collection_name = collection_map[collection_type]
    else:
        logger.warning(f"Unknown collection type: {collection_type}, using default")

    return config


def create_config_for_testing(
    use_memory_store: bool = True, embedding_model: str = "mock"
) -> RAGConfig:
    """
    Create a RAG configuration suitable for testing.

    Args:
        use_memory_store: If True, use in-memory vector store
        embedding_model: Embedding model to use

    Returns:
        RAGConfig configured for testing
    """
    config = RAGConfig()
    config.embedding.model = embedding_model
    config.embedding.cache_size = 1  # Minimal cache for testing

    if use_memory_store:
        config.vector_store.type = "memory"
        config.vector_store.persist_directory = None
    else:
        config.vector_store.type = "chroma"
        config.vector_store.persist_directory = Path("/tmp/test_rag_chromadb")

    # Use smaller chunks for testing
    config.chunking.chunk_size = 100
    config.chunking.chunk_overlap = 20

    # Faster search for tests
    config.search.default_top_k = 5

    return config


# Example TOML configuration structure for documentation
EXAMPLE_TOML_CONFIG = """
# Example RAG configuration in config.toml

[AppRAGSearchConfig.rag]
# Embedding configuration
[AppRAGSearchConfig.rag.embedding]
model = "mxbai-embed-large-v1"  # Uses model from [embedding_config.models.mxbai-embed-large-v1]
device = "auto"  # Auto-detect best device ("auto", "cpu", "cuda", "mps")
cache_size = 2
batch_size = 16  # Reduced for larger model
max_length = 512
# For API-based models (optional):
# api_key = "your-api-key"  # Or use OPENAI_API_KEY env var
# base_url = "http://localhost:8080/v1"  # For local servers

# Vector store configuration
[AppRAGSearchConfig.rag.vector_store]
type = "chroma"  # or "memory"
persist_directory = "~/.local/share/tldw_cli/chromadb"
collection_name = "default"
distance_metric = "cosine"  # or "l2", "ip"

# Chunking configuration
[AppRAGSearchConfig.rag.chunking]
chunk_size = 400
chunk_overlap = 100
method = "words"  # or "sentences", "paragraphs"

# Search configuration
[AppRAGSearchConfig.rag.search]
default_top_k = 10
score_threshold = 0.0
include_citations = true
citation_style = "inline"  # "inline", "footnote", or "none"
snippet_max_chars = 240
default_search_mode = "semantic"  # "plain", "semantic", or "hybrid"
max_context_size = 16000
fts_top_k = 10
vector_top_k = 10
cache_size = 100
cache_ttl = 3600  # 1 hour default for all search types
# Optional search-type specific cache TTLs
# semantic_cache_ttl = 7200  # 2 hours for semantic search
# keyword_cache_ttl = 1800   # 30 minutes for keyword search
# hybrid_cache_ttl = 3600    # 1 hour for hybrid search
fts5_connection_pool_size = 3  # Adjust based on concurrent search load

# Retriever configuration (authoritative location for hybrid_alpha)
[AppRAGSearchConfig.rag.retriever]
# Hybrid fusion alpha: weight of the vector leg in the RRF blend
# (0 = FTS only, 1 = vector only). Default 0.7 matches tldw_server.
hybrid_alpha = 0.7
media_collection = "media_embeddings"
chat_collection = "chat_embeddings"
notes_collection = "notes_embeddings"
character_collection = "character_embeddings"

[AppRAGSearchConfig.rag.processor]
enable_reranking = false
reranker_model = null
reranker_top_k = 5

# Query expansion configuration
[AppRAGSearchConfig.rag.query_expansion]
enabled = false
method = "llm"  # "llm", "local_llm", "llamafile", "keywords"
max_sub_queries = 3
llm_provider = "openai"
llm_model = "gpt-3.5-turbo"
local_model = "Qwen3-0.6B-Q6_K.gguf"  # For Ollama/llamafile
expansion_prompt_template = "default"
combine_results = true
cache_expansions = true
"""
