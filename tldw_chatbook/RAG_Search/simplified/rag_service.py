"""
Main RAG service coordinator with citations support.

This is the main entry point for the simplified RAG implementation, coordinating
embeddings, vector stores, chunking, and search operations.
"""

import asyncio
import json
import re
import sqlite3
import time
import uuid
from collections import abc
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    FrozenSet,
    Hashable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
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
from tldw_chatbook.Utils.fts5_match_forms import (
    FTS5_STOPWORDS,
    build_prefix_match_expression,
    fts5_query_tokens,
    fts5_token_runs,
    is_fts5_stopword,
    quote_fts5_token,
)
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
# TASK-15020/B2. Unlike the three above, this one has no vector-leg twin to
# agree with: nothing indexes prompts semantically. It is the singular
# spelling anyway, because the OTHER consumers of this string are shared with
# the three that do -- `_fusion_doc_key`, the Library's provenance
# canonicalization (`_prompt_row` already stamps `prompt`) and the eval
# harness's `SOURCE_TYPE_ALIASES`. A plural here would leave prompt rows
# present but unable to merge or to be post-filtered by the Prompts toggle.
SOURCE_TYPE_PROMPT = "prompt"

# Every source type the keyword (FTS5) leg has a sub-leg for, and the two of
# those that live in the ChaChaNotes database. A caller's
# ``keyword_source_types`` selection is expressed in THIS vocabulary (the
# engine's singular spelling), never in a UI's plural scope identifiers.
KEYWORD_LEG_SOURCE_TYPES = frozenset(
    {
        SOURCE_TYPE_MEDIA,
        SOURCE_TYPE_NOTE,
        SOURCE_TYPE_CONVERSATION,
        SOURCE_TYPE_PROMPT,
    }
)
CHACHA_KEYWORD_SOURCE_TYPES = frozenset({SOURCE_TYPE_NOTE, SOURCE_TYPE_CONVERSATION})


# --- The keyword leg's MATCH construction (TASK-15400, TASK-15700) ---------
#
# The candidates the two arcs' specs pre-register, selected by
# `SearchConfig.fts_match_construction` and resolved by
# `RAGService._fts5_match_expressions`. See that method for what each one
# builds and why; see the config field for why this is not a user knob.
#
# The last two are TASK-15700's additions for the sweep's re-run under the
# form-tiered merge: the 15400 sweep ran prefix matching as a REPORT-ONLY
# probe (it rescued 3 of the 40 zero-row golden queries, the best of the two
# probes) and the spec promotes it to a full matrix row, plus the obvious
# composition of it with the AND primary.
FTS_MATCH_CONSTRUCTION_AND = "and"
FTS_MATCH_CONSTRUCTION_AND_STOPWORD_TRIM = "and_stopword_trim"
FTS_MATCH_CONSTRUCTION_OR = "or"
FTS_MATCH_CONSTRUCTION_AND_THEN_OR = "and_then_or"
FTS_MATCH_CONSTRUCTION_PREFIX = "prefix"
FTS_MATCH_CONSTRUCTION_AND_THEN_PREFIX = "and_then_prefix"
FTS_MATCH_CONSTRUCTIONS = (
    FTS_MATCH_CONSTRUCTION_AND,
    FTS_MATCH_CONSTRUCTION_AND_STOPWORD_TRIM,
    FTS_MATCH_CONSTRUCTION_OR,
    FTS_MATCH_CONSTRUCTION_AND_THEN_OR,
    FTS_MATCH_CONSTRUCTION_PREFIX,
    FTS_MATCH_CONSTRUCTION_AND_THEN_PREFIX,
)

# The values `metadata["fts_match"]` can take. They name the FORM that
# matched a row and NOTHING else: `and` is an implicit-AND expression (full
# or stopword-trimmed), `or` is the content-token OR form -- whether that
# form ran as `and_then_or`'s fallback or as the `or` construction's own
# primary -- and `prefix` is the content-token PREFIX form, likewise either
# `and_then_prefix`'s fallback or `prefix`'s own primary. Deliberately NOT
# `or_fallback`: under the `or` construction every row comes from the primary
# query, so a name carrying "fallback" would weld a false position claim onto
# a true form fact, and Task 5's mechanism prose reads this key verbatim.
# Fallback-ness is derivable whenever it is wanted (construction + form), so
# it gets no second field. That naming is what let the probe axis (`near`,
# prefix) become a shippable construction here without a rename.
FTS_MATCH_AND = "and"
FTS_MATCH_OR = "or"
FTS_MATCH_PREFIX = "prefix"

# Which FORM each construction's PRIMARY and FALLBACK expressions run --
# deliberately shaped as a pair, parallel to `_fts5_match_expressions`'
# ``(primary, fallback)`` return, with ``None`` meaning "this construction
# has no fallback".
#
# ONE table (TASK-15700). Three consumers read it and must never disagree:
# the row stamp in `_fts_rows_with_fallback`, `_keyword_search`'s tier
# partition (primary-form sub-legs ahead of fallback ones), and
# `_keyword_row_metadata`'s default for a row that arrived unstamped. Before
# this table the FALLBACK branch hardcoded ``FTS_MATCH_OR`` while only the
# primary branch was construction-aware, so a construction whose fallback is
# NOT the OR form would have stamped its rows ``or``: the tiering would
# still have worked (any non-primary value lands tier 2) but the sweep's
# negative-composition table would have attributed those rows to a form
# they never ran. A construction added here names both of its forms in one
# edit, or `test_every_construction_names_the_forms_it_can_run` reds.
FTS_MATCH_FORMS_BY_CONSTRUCTION: Dict[str, Tuple[str, Optional[str]]] = {
    FTS_MATCH_CONSTRUCTION_AND: (FTS_MATCH_AND, None),
    FTS_MATCH_CONSTRUCTION_AND_STOPWORD_TRIM: (FTS_MATCH_AND, None),
    FTS_MATCH_CONSTRUCTION_OR: (FTS_MATCH_OR, None),
    FTS_MATCH_CONSTRUCTION_AND_THEN_OR: (FTS_MATCH_AND, FTS_MATCH_OR),
    FTS_MATCH_CONSTRUCTION_PREFIX: (FTS_MATCH_PREFIX, None),
    FTS_MATCH_CONSTRUCTION_AND_THEN_PREFIX: (FTS_MATCH_AND, FTS_MATCH_PREFIX),
}

# A small fixed English function-word list, consulted by every FORM except
# the full AND (`and`, the pre-TASK-15400 construction) -- so it is consulted
# on the trimmed AND of `and_stopword_trim`, on the OR form used by
# `or`/`and_then_or`'s fallback, and on the PREFIX form used by
# `prefix`/`and_then_prefix`'s fallback (TASK-15700).
#
# Where that lands on the SHIPPED DEFAULT is worth being exact about, since
# 2026-08-13 shipped `and_then_prefix`: its PRIMARY is the full AND, which
# does NOT consult this list, and its FALLBACK is the prefix form, which
# does. So on the default path today the list is consulted only for sub-legs
# whose primary returned zero rows -- not on every search, as it was under
# `and_stopword_trim`.
#
# The PREFIX form is where this list is MOST load-bearing, by a distance. An
# untrimmed OR admits every document containing "the"; an untrimmed prefix
# term is worse still, because `"the"*` matches "the", "then", "there",
# "their", "these" -- i.e. nearly a whole corpus -- and the prefix form ANDs
# its terms, so one such term does not merely add noise, it makes the whole
# expression match almost everything the other terms allow. That is why the
# prefix constructions answer `""` (no rows) when trimming empties the token
# list rather than falling back to the full AND the way `and_stopword_trim`
# does. The FULL AND
# never consults it: an implicit AND over function words is harmless, and
# TASK-15400's census measured that trimming them rescues 1 of the 40
# zero-row golden queries (`pm-vendor-chaser`, blocked solely by "about") --
# the +1 that made `and_stopword_trim` that sweep's winner and the default
# from 2026-08-11 to 2026-08-13. (TASK-15700's re-run reaches that same
# query through the PREFIX fallback instead, so the rescue survived the
# flip on a different mechanism.) Where the list is more load-bearing still
# is the
# OR form: a raw OR of every token matches every document containing "the",
# and bm25's IDF discounts a ubiquitous term in the RANKING but not in the
# row COUNT, so the junk rows still enter fusion. Fixed and small on purpose
# -- a large list starts deleting content words (the census's real blockers
# were `template`, `building`, `rough`, `turns`, `pulls`, `builds`, which no
# stopword list removes). The exact size is pinned by
# `test_stopword_list_is_lowercase_and_covers_the_measured_blocker`.
#
# TASK-17755 MOVED the literal to `Utils/fts5_match_forms.py` and imports it
# back under this name. It did not copy it. The Library's four-seam plain
# Search path adopted `and_then_prefix` in that task, so a second consumer
# now needs this list and the prefix builder below -- and the Library screen
# runs BOTH paths (Search is the four-seam one, RAG Answer is this engine),
# which is precisely the situation where two lists that agree today and
# diverge next quarter present as a retrieval bug nobody can locate. There
# is one list; `_FTS5_STOPWORDS` is still the name the engine's tests and
# the RAG_Eval probes read it by.
_FTS5_STOPWORDS: FrozenSet[str] = FTS5_STOPWORDS


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
        logger.debug("Ignoring unknown keyword-leg source types.")
    return requested & KEYWORD_LEG_SOURCE_TYPES


async def _no_keyword_rows() -> List[Any]:
    """A skipped sub-leg's contribution: nothing, and no query to get it.

    Used so ``_keyword_search`` can keep gathering a fixed pair of awaitables
    while an unselected sub-leg never touches a database.
    """
    return []


#: The two metadata dimensions the FTS sub-legs can actually enforce. They
#: are exactly the keys `rag_scope.build_semantic_allowlists` produces, and
#: an entry carrying anything else is a scoping request no sub-leg can honor
#: (see `_keyword_allowlist_ids`).
ALLOWLIST_SOURCE_TYPE_KEY = "source_type"
ALLOWLIST_SOURCE_ID_KEY = "source_id"
ENFORCEABLE_ALLOWLIST_KEYS = frozenset(
    {ALLOWLIST_SOURCE_TYPE_KEY, ALLOWLIST_SOURCE_ID_KEY}
)

#: One allowlist entry (its keys AND-ed together), or a union of them.
MetadataAllowlist = Union[
    Mapping[str, Collection[str]], Sequence[Mapping[str, Collection[str]]]
]


def _rereadable_entry(
    entry: Mapping[str, Collection[str]],
) -> Mapping[str, Collection[str]]:
    """Freeze an entry's one-shot values, leaving re-readable ones alone.

    Same hazard as ``_allowlist_entries``' own, one level down: a generator
    passed as ``source_id`` is drained by whoever reads it first (the cache
    key, in ``RAGService.search``) and every later reader sees an EMPTY set
    of allowed ids. ``abc.Collection`` is exactly the "can be read again"
    test -- a set/frozenset/list/tuple/str is one, a generator is not.

    Args:
        entry: One AND-group.

    Returns:
        The entry with any non-re-readable value materialized. Re-readable
        values are passed through untouched, so the common case allocates
        nothing new beyond the dict.
    """
    return {
        key: values if isinstance(values, abc.Collection) else tuple(values)
        for key, values in entry.items()
    }


def _allowlist_entries(
    metadata_allowlist: Optional[MetadataAllowlist],
) -> Tuple[Mapping[str, Collection[str]], ...]:
    """Normalize either allowlist shape into a tuple of AND-group entries.

    ``RAGService.search`` accepts ONE mapping (every key AND-ed -- the shape
    every pre-TASK-15020 caller passes, one per store query) or a SEQUENCE of
    them (a union of AND-groups -- what
    ``rag_scope.build_semantic_allowlists`` returns, deliberately one entry
    per source type because a flat dict cannot express "media in A OR note in
    B"). Both reduce to the same thing here.

    **The result is re-readable.** A scope is read at least three times per
    search (the cache key, then each leg), so a one-shot iterable would be
    drained by the first reader and reach the legs EMPTY -- which fails OPEN:
    an empty allowlist is "no scoping request", so both legs would run
    unscoped and the unscoped rows would then be cached under the SCOPED key.
    Reproduced with a generator expression during review; materializing here
    (and in ``_rereadable_entry``, for the values) is what makes the shape
    safe for the new seams Tasks 5/7 thread allowlists through.

    Args:
        metadata_allowlist: The caller's allowlist, in either shape.

    Returns:
        One entry per AND-group; empty for ``None``, an empty mapping, or an
        empty sequence (all of which mean "no scoping request").

    Raises:
        ValueError: If a SEQUENCE contains an empty entry. An empty
            AND-group restricts nothing, so silently dropping it would let
            ``[{}]`` read as "no allowlist" (fail-open) while ``[{media},
            {}]`` reads as "media only" -- two different answers to the same
            malformed input. ``EffectiveScope`` carries only non-empty
            entries, so this is unreachable from ``build_semantic_allowlists``
            and is a caller defect wherever it appears.
    """
    if not metadata_allowlist:
        return ()
    if isinstance(metadata_allowlist, abc.Mapping):
        return (_rereadable_entry(metadata_allowlist),)
    entries = tuple(metadata_allowlist)
    if any(not entry for entry in entries):
        raise ValueError(
            "metadata_allowlist entries must each be a non-empty mapping; an "
            "empty entry restricts nothing and would silently widen the scope"
        )
    return tuple(_rereadable_entry(entry) for entry in entries)


def _keyword_allowlist_ids(
    metadata_allowlist: Optional[MetadataAllowlist],
) -> Optional[Dict[str, Optional[List[str]]]]:
    """Translate an allowlist into per-sub-leg id filters for the FTS leg.

    The translation is the whole of B1: each entry names one or more source
    types and the ids allowed for them, and each FTS sub-leg is the keyword
    half of exactly one of those types. A type the allowlist never names has
    no entry to run under, so its sub-leg is SKIPPED -- fail-closed, matching
    the semantic side, where a store query AND-scoped on ``source_type``
    simply cannot return rows of a type it does not name. "Skip" and "run
    unfiltered" produce identical row sets on a single-type corpus and wildly
    different ones on a real corpus, which is why the direction is pinned by
    spies rather than by row counts alone.

    Args:
        metadata_allowlist: The caller's allowlist, in either shape.

    Returns:
        ``None`` when there is no allowlist -- the leg runs exactly as it did
        before this parameter existed. Otherwise a mapping from source type
        to that sub-leg's allowed ids (stringified and sorted, ready for
        ``json_each``), or to ``None`` when the entry restricts the type but
        not the ids. **A source type ABSENT from the mapping must not be
        queried at all**; an empty mapping therefore means "no sub-leg may
        run", which is the caller's own degrade path.
    """
    entries = _allowlist_entries(metadata_allowlist)
    if not entries:
        return None

    ids_by_type: Dict[str, Optional[set]] = {}
    unservable_types: set = set()
    for entry in entries:
        unenforceable = set(entry) - ENFORCEABLE_ALLOWLIST_KEYS
        if unenforceable:
            # Honoring the enforceable half of a scoping request would run an
            # UNDER-restricted query and return rows the caller asked to
            # exclude. The keyword leg contributes nothing for this entry
            # instead; the semantic leg still applies the whole entry.
            logger.warning(
                "Keyword leg cannot enforce allowlist key(s) {}; this scope "
                "entry contributes no keyword sub-leg (the semantic leg "
                "still applies it in full).",
                sorted(str(key) for key in unenforceable),
            )
            continue

        raw_types = entry.get(ALLOWLIST_SOURCE_TYPE_KEY)
        if raw_types is None:
            # No type restriction: the entry's ids apply to every sub-leg.
            source_types: Collection[str] = KEYWORD_LEG_SOURCE_TYPES
        else:
            source_types = {str(value) for value in raw_types}

        raw_ids = entry.get(ALLOWLIST_SOURCE_ID_KEY)
        entry_ids = (
            None if raw_ids is None else {str(value) for value in raw_ids}
        )

        for source_type in source_types:
            if source_type not in KEYWORD_LEG_SOURCE_TYPES:
                unservable_types.add(source_type)
                continue  # No sub-leg serves this type (e.g. a vector-only type).
            if source_type not in ids_by_type:
                ids_by_type[source_type] = (
                    None if entry_ids is None else set(entry_ids)
                )
            elif ids_by_type[source_type] is not None and entry_ids is not None:
                # Two entries naming one type are a union, not an
                # intersection: each is an independent AND-group.
                ids_by_type[source_type].update(entry_ids)
            else:
                # One of them restricts nothing, so their union restricts
                # nothing.
                ids_by_type[source_type] = None

    if unservable_types:
        # Named, not swallowed. The two sibling paths
        # (`_resolve_keyword_source_types`, the Library's translation map)
        # both say which values they dropped, and this one matters MORE: a
        # plural typo ("notes") or a source type that reaches the scope
        # vocabulary before it has an FTS sub-leg (exactly B2's prompt
        # sequencing) empties the keyword leg for that type with no other
        # symptom than missing rows.
        logger.warning(
            "Allowlist names source type(s) {} that no keyword sub-leg "
            "serves; the leg serves {}. Items of those types are reachable "
            "through the vector leg only.",
            sorted(str(value) for value in unservable_types),
            sorted(KEYWORD_LEG_SOURCE_TYPES),
        )

    return {
        # An entry that names a type with ZERO ids can only match nothing --
        # the same answer the semantic leg gives (`str(id) in set()` is False
        # for every candidate), reached without a query. Dropping it here is
        # what makes that skip, rather than an empty `json_each` round trip.
        source_type: (None if ids is None else sorted(ids))
        for source_type, ids in ids_by_type.items()
        if ids is None or ids
    }


def _json_id_param(allowed_ids: Collection[str]) -> str:
    """Bind an id allowlist as ONE parameter, the way the ORM already does.

    ``ChaChaNotes_DB.search_notes``' ``id_allowlist`` and
    ``Client_Media_DB_v2``'s large-``media_ids_filter`` branch both encode
    the ids as a single JSON array consumed by ``json_each``, so a ~1k-item
    scope (an ordinary collection) cannot hit SQLite's bound-parameter cap.
    The ids are stringified to match the scope's own type
    (``EffectiveScope`` carries ``str`` ids) and the vector store's
    comparison (``str(metadata[key]) in values``); SQLite applies the left
    operand's NUMERIC affinity for an INTEGER id column (``Media.id``), the
    same reliance both precedents already have.

    Args:
        allowed_ids: The ids this sub-leg may return.

    Returns:
        A JSON array literal, sorted for a deterministic query.
    """
    return json.dumps(sorted(str(value) for value in allowed_ids))


#: The ``prompts_fts`` columns that make up a prompt's DOCUMENT, in the
#: order the ORM and the Prompts UI present them. The other two indexed
#: columns are this row's metadata, not its text: ``name`` becomes the
#: title and ``author`` the author field, exactly as the media sub-leg
#: treats them.
PROMPT_DOCUMENT_COLUMNS = ("details", "system_prompt", "user_prompt")


def _prompt_document_text(row: Any) -> str:
    """Render a prompt row's body as this sub-leg's document text.

    A saved prompt is not one text field: `prompts_fts` indexes five
    columns, and a query can match any of them. Concatenating the three
    BODY columns (skipping the empty ones -- the writer stores ``""`` for a
    missing field, so most prompts have two of the three blank) is the
    honest answer to "show me what matched", and it is also the text the
    citation/snippet builders then search for the query's tokens.

    A match on ``name`` or ``author`` alone therefore yields a row whose
    document does not contain the query term. That is not a defect and is
    not special-cased: the media sub-leg has had exactly that property
    since it existed (a title match returns the body), and the title is on
    the row for the user to see.

    Args:
        row: A ``sqlite3.Row``/mapping carrying the body columns.

    Returns:
        The non-empty body columns joined by a blank line, in
        ``PROMPT_DOCUMENT_COLUMNS`` order; ``""`` when all three are empty
        (a prompt with only a name), which the row processing renders as an
        empty document rather than dropping the row.
    """
    parts = []
    for column in PROMPT_DOCUMENT_COLUMNS:
        value = row[column]
        if value and str(value).strip():
            parts.append(str(value).strip())
    return "\n\n".join(parts)


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
        logger.warning("Invalid hybrid_pool_multiplier; using shipped default")
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
    ``conversation``); it keeps note 15 and media 15 apart. The keyword leg
    adds a fourth value, ``prompt`` (TASK-15020/B2), which no indexer writes
    -- prompts have no vector twin to merge with, so a prompt row is always
    an unmerged FTS-only row. It still has to be in this key's vocabulary:
    without ``source_type``, prompt 15 and note 15 would collide on
    ``source_id`` alone and fusion would merge two different documents.

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

        # Unrecognized `fts_match_construction` values already seen, so the
        # use-time resolver warns once per service rather than once per
        # search (TASK-15400).
        self._warned_fts_constructions: set = set()

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
        metadata_allowlist: Optional[MetadataAllowlist] = None,
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
                into each leg's own candidate selection instead of filtered
                afterward. Accepts ONE mapping (its keys AND-ed) or a
                SEQUENCE of them (a union of AND-groups -- what
                ``rag_scope.build_semantic_allowlists`` returns, one entry
                per source type, because a flat dict cannot express "media
                in A OR note in B").

                Supported for ``search_type="semantic"`` (the vector store
                filters candidates before ranking; a multi-entry allowlist
                runs one store query per entry and merges by score, the
                convention its callers already follow) and, since
                TASK-15020/B1, for ``search_type="hybrid"``, where it reaches
                BOTH legs: the semantic leg exactly as above, and the FTS leg
                as a per-sub-leg ``id IN (SELECT value FROM json_each(?))``
                restriction (see ``_keyword_allowlist_ids``). A sub-leg whose
                source type the allowlist never names is SKIPPED rather than
                run unfiltered, and an allowlist that leaves no sub-leg
                runnable degrades the keyword leg to ``[]`` -- hybrid then
                falls back to its semantic leg through the same path an empty
                FTS result already takes.

                ``search_type="keyword"`` still raises ``ValueError`` rather
                than silently ignoring the scoping request: it has no
                semantic leg to scope, and the app's scoped plain-profile
                search runs through the Library's own four-seam path, which
                is scope-aware at each database.
            keyword_source_types: Source types the keyword (FTS5) leg should
                budget for, in the engine's vocabulary
                (``media``/``note``/``conversation``/``prompt``). ``None``
                -- the default, and every pre-TASK-14751 caller -- serves
                all four. The mirror image of ``metadata_allowlist``: it
                scopes the
                KEYWORD leg only, so passing it with
                ``search_type="semantic"`` raises ``ValueError`` rather than
                silently ignoring the scoping request. It is part of the
                cache key, so two selections of the same query never share
                an entry.

        Returns:
            List of search results (with or without citations)

        Raises:
            ValueError: If ``metadata_allowlist`` is provided with
                ``search_type="keyword"``, or if ``keyword_source_types`` is
                provided with ``search_type="semantic"``.
        """
        # Freeze the scope BEFORE anything reads it. This method reads it
        # twice (the cache key, then the legs) and hybrid reads it twice
        # more, so a one-shot iterable would reach the legs drained -- i.e.
        # unscoped -- and the unscoped rows would be stored under the SCOPED
        # cache key. `_allowlist_entries` also rejects malformed entries, so
        # both happen once, here, before any guard reads a value.
        metadata_allowlist = _allowlist_entries(metadata_allowlist) or None

        if metadata_allowlist and search_type == "keyword":
            raise ValueError(
                "metadata_allowlist is not supported for search_type='keyword' "
                "(use search_type='hybrid' for a scoped search that keeps the "
                "keyword leg, or the scope-aware per-database search path)"
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

        # The keyword leg's companion to `hybrid_fusion_key` (TASK-15400):
        # the MATCH construction decides which rows the FTS leg returns at
        # all, so every search that READS that leg keys it. `None` for
        # semantic (no keyword leg). The resolved value, for the same reason
        # the fusion params are resolved: an unrecognized construction
        # behaves as `and` and must therefore key as `and`.
        fts_match_construction_key: Optional[str] = None
        if search_type in ("hybrid", "keyword"):
            fts_match_construction_key = self._resolved_fts_match_construction()

        # Check cache first
        cached_result = await self.cache.get_async(
            query,
            search_type,
            top_k,
            filter_metadata,
            metadata_allowlist,
            keyword_source_types=keyword_source_types,
            hybrid_fusion=hybrid_fusion_key,
            fts_match_construction=fts_match_construction_key,
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
                results = await self._semantic_search_scoped(
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
                    metadata_allowlist=metadata_allowlist,
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
                fts_match_construction=fts_match_construction_key,
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

    async def _semantic_search_scoped(
        self,
        query: str,
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]] = None,
        include_citations: bool = True,
        score_threshold: float = 0.0,
        *,
        metadata_allowlist: Optional[MetadataAllowlist] = None,
    ) -> Union[List[SearchResult], List[SearchResultWithCitations]]:
        """The semantic leg under an allowlist of either shape.

        A union allowlist cannot be one store query -- every key inside one
        ``metadata_allowlist`` is AND-ed, and a media id and a note id are
        not guaranteed distinct -- so each entry is its own query and the
        results merge by score, which is exactly what
        ``library_local_rag_search_service`` and
        ``pipeline_functions_simple`` already do for scoped semantic search.
        Moving that convention in here is what lets HYBRID scope its semantic
        leg the same way, with one fusion over the union rather than one
        fusion per source type.

        Zero or one entry takes the single-call path unchanged, so every
        pre-TASK-15020 caller (all of which pass one mapping) behaves exactly
        as before -- same store call, same arguments, same trimming.

        Args:
            query: Search query text.
            top_k: Number of results to return.
            filter_metadata: Metadata equality filters applied after the
                store call.
            include_citations: Whether to fetch citations from the store.
            score_threshold: Minimum similarity score to keep.
            metadata_allowlist: One AND-group, or a union of them.

        Returns:
            Up to ``top_k`` results, most similar first.
        """
        entries = _allowlist_entries(metadata_allowlist)
        if len(entries) <= 1:
            return await self._semantic_search(
                query,
                top_k,
                filter_metadata,
                include_citations,
                score_threshold,
                metadata_allowlist=entries[0] if entries else None,
            )

        merged: List[Any] = []
        for entry in entries:
            merged.extend(
                await self._semantic_search(
                    query,
                    top_k,
                    filter_metadata,
                    include_citations,
                    score_threshold,
                    metadata_allowlist=entry,
                )
            )
        # No cross-entry dedup, deliberately: `build_semantic_allowlists`
        # emits ONE entry per source_type, so the entries are disjoint and no
        # chunk can appear in two of them. (The FTS side does dedup, via
        # `_fusion_doc_key`, because its sub-legs can genuinely overlap.) A
        # future allowlist whose entries are NOT disjoint would need one here.
        merged.sort(key=lambda result: result.score, reverse=True)
        return merged[:top_k]

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
        metadata_allowlist: Optional[MetadataAllowlist] = None,
    ) -> Union[List[SearchResult], List[SearchResultWithCitations]]:
        """
        Perform keyword (FTS5) search across media, notes, conversations and
        saved prompts.

        TASK-3996: this leg used to join ``media_fts`` and nothing else, so
        the keyword half of hybrid search could only ever return media rows
        -- notes and conversations were structurally unreachable through it
        no matter what the query said (29 of the P1 fixture corpus's 49
        documents). It is now four sub-legs over three databases:

        * media -- ``media_fts`` in the media DB, via the connection pool;
        * notes -- ``notes_fts`` in the ChaChaNotes DB;
        * conversations -- ``messages_fts`` in the ChaChaNotes DB, one row
          per matching conversation;
        * prompts -- ``prompts_fts`` in the Prompts DB (TASK-15020/B2).

        The ChaChaNotes and Prompts sub-legs run over READ-ONLY raw
        connections (never ``CharactersRAGDB``/``PromptsDatabase``, whose
        constructors do schema work), and each sub-leg degrades
        independently: a missing chacha DB costs the notes/conversation rows
        and leaves media and prompts untouched, and so on. The leg is empty
        only when every sub-leg is empty or unavailable.

        **Prompts are the one type with no other path.** Media, notes and
        conversations are all indexed semantically, so a hybrid search can
        reach them through either leg; nothing indexes prompts, so this
        sub-leg is the ONLY way a prompt ever enters hybrid results, and it
        gets there as an FTS-only row rescued by the fusion weighting
        (``config.DEFAULT_HYBRID_RRF_K``).

        The sub-legs are merged rank-fairly (``interleave_rankings``, round
        robin by rank position) rather than concatenated: FTS5 scores from
        different tables are not comparable, and concatenation would let one
        well-stocked source consume every ``top_k`` slot.

        TASK-15700 tiers that merge by FORM: sub-legs whose rows came from
        the construction's primary expression round-robin among themselves
        first, and sub-legs that FELL BACK round-robin after them, filling
        only the slots the primary tier left. Fusion consumes this leg's
        RANK, so before the tiering a widening sub-leg's row count could
        demote another sub-leg's untouched rank-1 row purely by source
        order -- see the merge site's comment for the measured incident.
        Under a construction with no fallback there is exactly one tier and
        the order is unchanged -- which WAS every shipped default up to
        2026-08-13, and is no longer: ``and_then_prefix`` defines a fallback,
        so this partition is live on the default path and the tiering is
        load-bearing rather than a no-op waiting for a future construction.

        TASK-14751 narrows *which* sub-legs run without touching that
        merge. ``keyword_source_types`` names the types the caller will
        actually keep (``None`` = all four, i.e. unchanged for every caller
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
                vocabulary (``media``/``note``/``conversation``/``prompt``).
                ``None`` serves all four; an empty collection serves none and
                returns ``[]`` without a database lookup; unrecognized
                values are dropped (see ``_resolve_keyword_source_types``).
            metadata_allowlist: A retrieval scope (TASK-15020/B1). Each
                sub-leg the allowlist names runs with its entry's ids bound
                as ``id IN (SELECT value FROM json_each(?))``; a sub-leg the
                allowlist does NOT name is skipped rather than run
                unfiltered, so the scope fails closed exactly as the semantic
                leg's does. It composes with ``keyword_source_types`` by
                intersection -- both narrow, neither widens. ``None`` (every
                caller that predates it) leaves the leg untouched.

        Returns:
            The merged leg, best first, capped at ``top_k``.
        """
        allowlist_ids = _keyword_allowlist_ids(metadata_allowlist)
        selected = _resolve_keyword_source_types(keyword_source_types)
        if allowlist_ids is not None:
            selected = selected & frozenset(allowlist_ids)
        if not selected:
            # An explicitly empty selection is "no keyword leg" -- an
            # answer, not a failure. So is a scope that leaves no selected
            # sub-leg runnable. Hybrid degrades to its semantic leg through
            # the same disclosed path an empty FTS result already takes.
            logger.debug(
                "Keyword search has no runnable sub-legs (selection={}, "
                "scoped id count={}); returning no results without a database "
                "lookup.",
                "all" if keyword_source_types is None else sorted(
                    str(value) for value in keyword_source_types
                ),
                None if allowlist_ids is None else len(allowlist_ids),
            )
            return []

        # TASK-3995: a query with no FTS5-searchable tokens (empty,
        # whitespace-only, or all punctuation) escapes to "" and can only
        # ever match nothing. Short-circuit before resolving any DB
        # path or acquiring a connection -- no FTS5 call, no DB touch.
        # TASK-15400: read the ACTIVE construction's primary expression, so
        # the `or` construction's all-stopword emptiness short-circuits here
        # too. Under `and` -- and so under the shipped `and_then_prefix`,
        # whose primary IS that full AND -- an all-stopword query still
        # produces a non-empty expression, so neither short-circuits on
        # emptiness; only the widening PRIMARIES (`or`, `prefix`) can.
        if not self._fts5_match_expressions(query)[0]:
            logger.debug(
                "Keyword search query has no FTS5-searchable tokens after "
                "escaping; returning no results without a database lookup."
            )
            return []

        chacha_types = selected & CHACHA_KEYWORD_SOURCE_TYPES
        media_ranking, chacha_rankings, prompts_ranking = await asyncio.gather(
            self._media_keyword_subleg(
                query,
                top_k,
                filter_metadata,
                include_citations,
                allowed_ids=(
                    None if allowlist_ids is None
                    else allowlist_ids.get(SOURCE_TYPE_MEDIA)
                ),
            )
            if SOURCE_TYPE_MEDIA in selected
            else _no_keyword_rows(),
            self._chacha_keyword_sublegs(
                query,
                top_k,
                filter_metadata,
                include_citations,
                source_types=chacha_types,
                allowed_ids=(
                    None if allowlist_ids is None
                    else {
                        source_type: allowlist_ids.get(source_type)
                        for source_type in chacha_types
                    }
                ),
            )
            if chacha_types
            else _no_keyword_rows(),
            self._prompts_keyword_subleg(
                query,
                top_k,
                filter_metadata,
                include_citations,
                allowed_ids=(
                    None if allowlist_ids is None
                    else allowlist_ids.get(SOURCE_TYPE_PROMPT)
                ),
            )
            if SOURCE_TYPE_PROMPT in selected
            else _no_keyword_rows(),
        )

        rankings = [
            ranking
            for ranking in (media_ranking, *chacha_rankings, prompts_ranking)
            if ranking
        ]
        if not rankings:
            return []

        # THE MERGE RULE (TASK-15700). Sub-legs whose rows came from the
        # construction's PRIMARY form are merged FIRST, as one tier;
        # sub-legs that FELL BACK are merged after them, as a second tier.
        # Round robin runs within each tier exactly as it always has (raw
        # FTS5 scores are not comparable across sources, so rank position
        # stays the only cross-source signal); the tiers themselves are
        # concatenated, and the `[:top_k]` truncation happens AFTER that --
        # so a fallback row can only ever occupy a slot the primary tier
        # left empty. Tier 2 FILLS; it never displaces.
        #
        # THE INCIDENT (TASK-15400 Task 3, measured over the 172-doc golden
        # corpus). This merge feeds hybrid fusion, which consumes the LEG
        # rank -- so before the tiering, one sub-leg's ROW COUNT decided
        # every other sub-leg's leg rank. Under `and_then_or`,
        # `kw-plant-maintenance-record` -> `note-saltmarsh-hide` had the
        # notes sub-leg's untouched rank-1 AND row; media and conversations
        # found zero AND rows, fell back to OR, injected 10 rows each, and
        # the round robin -- media first, by source order alone -- demoted
        # the untouched notes row to leg rank 2. At alpha 0.7 / rrf_k 5 the
        # vector rank-9 row then beat it by 6.94e-18 and the fixture lost
        # its hybrid rescue. The scoped category decomposed the same way to
        # the digit: the four NOTE-targeted scoped queries each fell behind
        # a media fallback row while the three MEDIA-targeted ones kept leg
        # rank 1 -- 3 of 7 = 0.429, recall 1.000 -> 0.429.
        #
        # Fallback-ness is f(construction, form), never the stamp alone:
        # under `or` the OR form IS the primary and belongs in tier 1. That
        # is `_fts5_primary_form()`, reading the one
        # `FTS_MATCH_FORMS_BY_CONSTRUCTION` table that also produces the row
        # stamps in `_fts_rows_with_fallback`, so the two cannot drift apart
        # about which form was the primary. An unstamped row defaults to the
        # primary and so lands in TIER 1 -- but the `.get` below is the
        # second half of that fail-safe, not the whole of it:
        # `_keyword_row_metadata` substitutes its own default first, and
        # that default is the same `_fts5_primary_form()` for exactly this
        # reason (it was a hardcoded `and`, which under `or` demoted the
        # unstamped row -- the opposite of the fail-safe).
        #
        # The partition is over SUB-LEGS, not rows: the fallback fires only
        # when a sub-leg's primary returns zero rows, so within one query a
        # sub-leg's rows are all-primary or all-fallback (pinned per sub-leg
        # by `test_notes_sub_leg_falls_back_independently`,
        # `test_conversations_sub_leg_falls_back_independently`,
        # `test_media_sub_leg_falls_back_independently` and the prompts pair
        # `test_a_matching_and_never_runs_the_fallback` /
        # `test_a_zero_row_and_falls_back_to_the_or_form_exactly_once` --
        # one per sub-leg, for a four-sub-leg fact). Reading row 0 is
        # therefore reading the whole sub-leg.
        #
        # Cross-tier deduplication is STRUCTURALLY VACUOUS and deliberately
        # absent: each sub-leg emits exactly one `source_type` and
        # `_fusion_doc_key` keys on it, so no document can appear in both
        # tiers. Within a tier, `interleave_rankings`' own `seen` set still
        # deduplicates on that same document identity fusion uses, so a
        # document appearing in two sub-legs occupies one slot.
        primary_form = self._fts5_primary_form()
        primary_tier: List[Any] = []
        fallback_tier: List[Any] = []
        for ranking in rankings:
            tier = (
                primary_tier
                if ranking[0].metadata.get("fts_match", primary_form) == primary_form
                else fallback_tier
            )
            tier.append(ranking)

        results = (
            interleave_rankings(primary_tier, key=_fusion_doc_key)
            + interleave_rankings(fallback_tier, key=_fusion_doc_key)
        )[:top_k]
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
        *,
        allowed_ids: Optional[Collection[str]] = None,
    ) -> Union[List[SearchResult], List[SearchResultWithCitations]]:
        """The media sub-leg of the keyword search: FTS5 over the media DB.

        Args:
            query: Raw user query (escaped for FTS5 downstream).
            top_k: Maximum rows this sub-leg contributes.
            filter_metadata: Optional metadata equality filters.
            include_citations: Whether to build citation-carrying rows.
            allowed_ids: Media ids this sub-leg may return (TASK-15020/B1).
                ``None`` is unrestricted -- today's behavior for every caller
                without a scope. The caller decides whether this sub-leg runs
                at all; reaching here means the scope named the media type.

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
                        allowed_ids,
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
        allowed_ids: Optional[Mapping[str, Optional[Collection[str]]]] = None,
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
            allowed_ids: Per-source-type id restrictions (TASK-15020/B1),
                e.g. ``{"note": [...]}``. ``None`` -- or a type mapped to
                ``None`` -- leaves that sub-query unrestricted. Which
                sub-legs run at all is ``source_types``' decision; the
                caller has already intersected it with the scope.

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
                allowed_ids,
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
        allowed_ids: Optional[Mapping[str, Optional[Collection[str]]]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Run the selected ChaChaNotes FTS sub-queries on one connection.

        Args:
            db_path: Validated path to the ChaChaNotes database.
            query: Raw user query.
            limit: Maximum rows per sub-query.
            source_types: Which of ``note``/``conversation`` to query
                (TASK-14751); ``None`` queries both. An unselected sub-query
                is never issued and its key stays empty.
            allowed_ids: Per-source-type id restrictions (TASK-15020/B1);
                ``None``, or a type mapped to ``None``, is unrestricted.

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

        # One construction, resolved once for both sub-queries; each of them
        # then decides independently whether its own zero-row result widens
        # (TASK-15400).
        expressions = self._fts5_match_expressions(query)
        if not expressions[0]:
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
                note_ids = (
                    None if allowed_ids is None else allowed_ids.get(SOURCE_TYPE_NOTE)
                )
                rows[SOURCE_TYPE_NOTE] = self._fts_rows_with_fallback(
                    lambda expression: self._chacha_notes_fts(
                        conn, expression, limit, note_ids
                    ),
                    expressions,
                )
            if SOURCE_TYPE_CONVERSATION in selected:
                conversation_ids = (
                    None
                    if allowed_ids is None
                    else allowed_ids.get(SOURCE_TYPE_CONVERSATION)
                )
                rows[SOURCE_TYPE_CONVERSATION] = self._fts_rows_with_fallback(
                    lambda expression: self._chacha_conversations_fts(
                        conn, expression, limit, conversation_ids
                    ),
                    expressions,
                )
        return rows

    @staticmethod
    def _chacha_notes_fts(
        conn: sqlite3.Connection,
        escaped_query: str,
        limit: int,
        allowed_ids: Optional[Collection[str]] = None,
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

        The id filter mirrors ``search_notes``' own ``id_allowlist`` clause,
        json_each and all -- the ORM's scoped notes search and this leg must
        restrict the same table the same way.

        Args:
            conn: Read-only ChaChaNotes connection.
            escaped_query: A per-token-quoted FTS5 MATCH expression.
            limit: Maximum rows.
            allowed_ids: Optional note ids this sub-leg may return
                (TASK-15020/B1); ``None`` is unrestricted.

        Returns:
            Row dicts (``id``/``title``/``content``), best match first.
        """
        params: List[Any] = [escaped_query]
        id_filter_sql = ""
        if allowed_ids is not None:
            id_filter_sql = "AND main.id IN (SELECT value FROM json_each(?))"
            params.append(_json_id_param(allowed_ids))
        params.append(limit)

        sql = f"""
        SELECT
            main.id AS id,
            main.title AS title,
            main.content AS content
        FROM notes_fts fts
        JOIN notes main ON fts.rowid = main.rowid
        WHERE fts.notes_fts MATCH ?
          AND main.deleted = 0
          {id_filter_sql}
        ORDER BY rank
        LIMIT ?
        """
        try:
            with closing(conn.execute(sql, tuple(params))) as cursor:
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
        conn: sqlite3.Connection,
        escaped_query: str,
        limit: int,
        allowed_ids: Optional[Collection[str]] = None,
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
            allowed_ids: Optional conversation ids this sub-leg may return
                (TASK-15020/B1); ``None`` is unrestricted. It restricts the
                CONVERSATION, which is this sub-leg's unit of retrieval and
                the id the vector leg carries -- the second statement below
                is already restricted to whatever conversations survive here.

        Returns:
            Row dicts (``id``/``title``/``content``), best match first.
        """
        params: List[Any] = [escaped_query]
        id_filter_sql = ""
        if allowed_ids is not None:
            id_filter_sql = "AND c.id IN (SELECT value FROM json_each(?))"
            params.append(_json_id_param(allowed_ids))
        params.append(limit)

        conversations_sql = f"""
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
          {id_filter_sql}
        GROUP BY c.id
        ORDER BY best_rank
        LIMIT ?
        """
        try:
            with closing(
                conn.execute(conversations_sql, tuple(params))
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

    async def _prompts_keyword_subleg(
        self,
        query: str,
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]] = None,
        include_citations: bool = True,
        *,
        allowed_ids: Optional[Collection[str]] = None,
    ) -> Union[List[SearchResult], List[SearchResultWithCitations]]:
        """The saved-prompts sub-leg, over a read-only Prompts DB.

        TASK-15020/B2, built to the chacha sub-legs' pattern exactly (path
        resolution -> read-only private-SQLite open -> one FTS query ->
        the shared row processing). What is NOT the same is the stake:
        prompts have no vector index anywhere, so this sub-leg is the only
        retrieval path a saved prompt has in the engine, and the fused
        result it produces is always an FTS-only row.

        Args:
            query: Raw user query (escaped for FTS5 downstream).
            top_k: Maximum rows this sub-leg contributes.
            filter_metadata: Optional metadata equality filters.
            include_citations: Whether to build citation-carrying rows.
            allowed_ids: Prompt ids this sub-leg may return
                (TASK-15020/B1's shape). Always ``None`` in practice today:
                the retrieval scope's vocabulary is media/note only (spec
                D5), so no allowlist can NAME prompts and a scoped search
                skips this sub-leg entirely before reaching here. The
                parameter exists so the day the scope vocabulary grows a
                prompt dimension, the filter is already pushed down rather
                than bolted on -- and so this sub-leg cannot accidentally
                become the one that runs unfiltered under a scope.

        Returns:
            Prompt rows, best first; ``[]`` on any failure (this sub-leg
            never breaks the other three).
        """
        # Nothing below may raise: `_hybrid_search` gathers this leg with the
        # semantic one without `return_exceptions`, so an escaping exception
        # would fail the whole search rather than degrade one sub-leg.
        try:
            db_path = self._resolve_prompts_db_path()
            if db_path is None:
                return []

            loop = asyncio.get_event_loop()
            items = await loop.run_in_executor(
                None,
                self._prompts_fts_rows,
                db_path,
                query,
                top_k * SEARCH_RESULT_MULTIPLIER,  # Get extra for filtering
                allowed_ids,
            )
            if not items:
                return []

            if include_citations:
                rows = await self._process_keyword_results_with_citations(
                    items, query, filter_metadata, top_k, source_type=SOURCE_TYPE_PROMPT
                )
            else:
                rows = self._process_keyword_results_basic(
                    items, filter_metadata, top_k, source_type=SOURCE_TYPE_PROMPT
                )
            logger.debug("Prompts keyword sub-leg found {} results", len(rows))
            return rows
        except Exception as e:
            logger.warning(
                "Prompts keyword sub-leg failed; the other sub-legs are "
                "unaffected (error_type={})",
                type(e).__name__,
            )
            return []

    def _resolve_prompts_db_path(self) -> Optional[Path]:
        """Resolve (and validate) the Prompts DB path for the FTS leg.

        Mirrors ``_resolve_chachanotes_db_path`` exactly: an explicit config
        override wins, otherwise the single authoritative resolver
        (``get_prompts_db_path``) decides -- no guessing across candidate
        filenames, and never a create-on-miss. The config-sourced override
        is run through ``path_validation``'s traversal/injection screen plus
        lexical normalization before it reaches a filesystem check.

        Returns:
            The validated, existing path, or ``None`` (with one logged
            warning naming the reason) when the prompts sub-leg is skipped.
        """
        from tldw_chatbook.Utils.path_validation import validate_path_simple
        from tldw_chatbook.Utils.private_paths import lexical_path

        try:
            from tldw_chatbook.config import get_prompts_db_path

            db_path_raw = (
                self.config.search.prompts_db_path or get_prompts_db_path()
            )
        except Exception as e:
            logger.warning(
                "Could not resolve the Prompts database path; the prompts "
                "keyword sub-leg returns no results (error_type={})",
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
                "Rejected prompts_db_path from config; the prompts keyword "
                "sub-leg returns no results (error_type={})",
                type(e).__name__,
            )
            return None

        # Existence only. Every other filesystem question -- symlinked
        # components, untrusted parent directories, a no-follow open of the
        # file itself -- belongs to the private SQLite seam this leg opens
        # through (see `_connect_prompts_readonly`).
        if not db_path.exists() or not db_path.is_file():
            logger.warning(
                "Prompts database not found; the prompts keyword sub-leg "
                "returns no results (a search never creates a database)."
            )
            return None

        return db_path

    def _connect_prompts_readonly(
        self, db_path: Union[str, Path]
    ) -> sqlite3.Connection:
        """Open the Prompts database read-only, without the ORM.

        The same three properties ``_connect_chacha_readonly`` documents,
        for the same reasons: a ``mode=ro`` URI built by the seam (so a
        write raises rather than being trusted not to happen);
        ``PromptsDatabase``'s constructor-time schema creation, migration
        and integrity work never runs on a search path; and
        ``connect_private_sqlite`` (owner ``rag.prompts_keyword_leg``)
        walks every path component with ``O_NOFOLLOW`` and opens the file
        itself no-follow, which a final-component ``is_symlink()`` check
        would not. ``preserve_read_only_source_mode`` keeps this reader from
        reasserting permissions on a file ``db.prompts.primary`` owns.

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
            "rag.prompts_keyword_leg",
            Path(db_path),
            read_only=True,
        )
        conn.row_factory = sqlite3.Row
        return conn

    def _prompts_fts_rows(
        self,
        db_path: Path,
        query: str,
        limit: int,
        allowed_ids: Optional[Collection[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Run the prompts FTS sub-query on one read-only connection.

        Args:
            db_path: Validated path to the Prompts database.
            query: Raw user query.
            limit: Maximum rows.
            allowed_ids: Optional prompt-id restriction; ``None`` is
                unrestricted.

        Returns:
            Row dicts (``id``/``title``/``content``/``author``), best match
            first -- an unopenable database or a failing sub-query yields
            ``[]`` plus one logged warning, never an exception.
        """
        expressions = self._fts5_match_expressions(query)
        if not expressions[0]:
            return []

        if not isinstance(limit, int) or limit < 1:
            limit = DEFAULT_FTS5_LIMIT
        limit = min(limit, MAX_FTS5_LIMIT)

        try:
            conn = self._connect_prompts_readonly(db_path)
        except (sqlite3.Error, ValueError, OSError) as e:
            # `PrivatePathError` is an `OSError`, so a path the seam refuses
            # (symlinked component, untrusted parent) lands here alongside a
            # genuinely unopenable file -- both degrade this sub-leg only.
            logger.warning(
                "Could not open the Prompts database read-only; the prompts "
                "keyword sub-leg returns no results (error_type={})",
                type(e).__name__,
            )
            return []

        with closing(conn):
            return self._fts_rows_with_fallback(
                lambda expression: self._prompts_fts(
                    conn, expression, limit, allowed_ids
                ),
                expressions,
            )

    @staticmethod
    def _prompts_fts(
        conn: sqlite3.Connection,
        escaped_query: str,
        limit: int,
        allowed_ids: Optional[Collection[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Prompts sub-query: mirrors ``PromptsDatabase``'s own prompt search.

        ``search_prompts`` and ``search_prompts_by_text`` both resolve
        ``prompts_fts`` rowids and then read ``Prompts`` with
        ``deleted = 0``; this is that pair of statements collapsed into one
        join, with the ORM's ``deleted`` filter kept verbatim.

        Two deliberate departures from the ORM, both because this is a
        RETRIEVAL leg rather than a list view:

        * **``ORDER BY rank``**, not ``last_modified DESC`` /
          ``name COLLATE NOCASE``. The other three sub-legs hand fusion a
          relevance ranking, and RRF fuses *positions* -- feeding it a
          recency order would make a prompt's fused score a function of when
          it was last edited.
        * **``prompt_keywords_fts`` is not consulted.** ``search_prompts``
          unions keyword matches in when the caller asks for the ``keywords``
          field; the keyword table is a separate index with its own rowid
          space (``PromptKeywordsTable``), so a union would need a second
          query and a merge with no comparable rank. Out of scope by the
          spec, and named here so its absence is a decision rather than an
          oversight.

        The ``deleted = 0`` predicate reads as redundant -- ``_delete_fts_
        prompt`` evicts the row from the index on soft delete -- and is not:
        an external-content ``'rebuild'`` re-indexes the content table,
        deleted rows included, and this predicate is then the only thing
        keeping a deleted prompt out of search results (pinned by
        ``test_deleted_prompts_are_excluded``, which rebuilds the index
        first; without the rebuild, dropping this line changes nothing).

        The join reads ``fts.rowid = main.id`` because ``prompts_fts``
        declares ``content_rowid='id'`` -- the id is the FTS rowid by
        construction, which is also why ``source_id`` below is directly
        comparable with the id every other prompt surface uses.

        Args:
            conn: Read-only Prompts connection.
            escaped_query: A per-token-quoted FTS5 MATCH expression.
            limit: Maximum rows.
            allowed_ids: Optional prompt ids this sub-leg may return; the
                filter mirrors the notes sub-leg's ``json_each`` clause.

        Returns:
            Row dicts (``id``/``title``/``content``/``author``), best first.
        """
        params: List[Any] = [escaped_query]
        id_filter_sql = ""
        if allowed_ids is not None:
            id_filter_sql = "AND main.id IN (SELECT value FROM json_each(?))"
            params.append(_json_id_param(allowed_ids))
        params.append(limit)

        sql = f"""
        SELECT
            main.id AS id,
            main.name AS name,
            main.author AS author,
            main.details AS details,
            main.system_prompt AS system_prompt,
            main.user_prompt AS user_prompt
        FROM prompts_fts fts
        JOIN Prompts main ON fts.rowid = main.id
        WHERE fts.prompts_fts MATCH ?
          AND main.deleted = 0
          {id_filter_sql}
        ORDER BY rank
        LIMIT ?
        """
        try:
            with closing(conn.execute(sql, tuple(params))) as cursor:
                return [
                    {
                        "id": row["id"],
                        "title": row["name"] or f"Prompt {row['id']}",
                        "content": _prompt_document_text(row),
                        "author": row["author"] or None,
                    }
                    for row in cursor
                ]
        except sqlite3.Error as e:
            logger.warning(
                "Prompts keyword sub-leg failed; returning no prompt rows "
                "(error_type={})",
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
        metadata_allowlist: Optional[MetadataAllowlist] = None,
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
        exactly as it was. It does not scope the semantic leg.

        ``metadata_allowlist`` (TASK-15020/B1) is the retrieval SCOPE, and it
        reaches BOTH legs: the vector store filters its candidates by it (one
        store query per entry, merged by score) and each FTS sub-leg the
        allowlist names restricts its ids to that entry's. A sub-leg the
        allowlist does not name is skipped, so a scoped hybrid can never
        return an out-of-scope keyword row -- before B1 this whole
        combination raised, which is why every scoped query in the app was
        diverted to a semantic-only search and the keyword leg was
        structurally unreachable under a scope.
        """
        # One scope, two legs: freeze it here as well as in `search`, so a
        # direct caller cannot hand this method a one-shot iterable that the
        # semantic leg drains before the keyword leg ever reads it (which
        # would leave the FTS leg unscoped -- failing OPEN).
        metadata_allowlist = _allowlist_entries(metadata_allowlist) or None

        # Get results from both search types. The pool multiplier widens
        # ONLY these two leg fetches -- `_semantic_search`'s own internal
        # over-fetch (its raw vector-store call) still uses the module
        # SEARCH_RESULT_MULTIPLIER, on this path and on the direct
        # semantic-search path alike.
        pool_multiplier = _resolve_hybrid_pool_multiplier(
            self.config.search.hybrid_pool_multiplier
        )
        semantic_task = self._semantic_search_scoped(
            query,
            top_k * pool_multiplier,
            filter_metadata,
            include_citations,
            score_threshold,
            metadata_allowlist=metadata_allowlist,
        )
        keyword_task = self._keyword_search(
            query,
            top_k * pool_multiplier,
            filter_metadata,
            include_citations,
            keyword_source_types=keyword_source_types,
            metadata_allowlist=metadata_allowlist,
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

    def _keyword_row_metadata(
        self, item: Dict, content: str, source_type: str
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
            # Which MATCH form actually returned this row (TASK-15400):
            # `and` for an implicit-AND expression, `or` for the
            # content-token OR form. One query can carry both -- the
            # fallback is decided per sub-leg -- so this is a per-ROW fact,
            # not a per-search one, and the sweep's negative-composition
            # count plus the arc's mechanism prose are both read off it.
            # Whether an `or` row was a FALLBACK is derived from this plus
            # the construction, never stamped as a second fact.
            #
            # The default is the ACTIVE construction's primary form, not a
            # hardcoded `and` (TASK-15700). This substitution happens BEFORE
            # `_keyword_search`'s tier partition can see the absence, so the
            # default here IS the fail-safe direction: an unstamped row must
            # land in tier 1, never be demoted behind a widened one. Probed
            # under the `or` construction, the hardcoded `and` did exactly
            # the opposite -- unreachable while every sub-leg goes through
            # `_fts_rows_with_fallback`, but live the moment a construction
            # ships whose primary form is not `and`.
            "fts_match": item.get("fts_match", self._fts5_primary_form()),
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
        the FULL token list ``_fts5_query_tokens`` returns for the query --
        every token the user typed, not necessarily the same tokens the
        MATCH expression required. Under the shipped ``and_then_prefix``
        construction the PRIMARY match runs over that full list, while its
        prefix FALLBACK runs over a strict SUBSET of it (content tokens
        only; see ``_fts5_match_expressions``), so a row matched by the
        fallback carries citations evidencing the full typed query while the
        match itself needed only some of its tokens. Verified NOT a
        behavioural regression: the spans themselves are unchanged, and a
        row that matched is guaranteed to carry the content tokens the
        MATCH required, so a matched row's
        citations are never missing evidence for the match. Locating the raw
        query as one contiguous substring instead (what this did before)
        assumed phrase semantics the keyword leg no longer has, so every
        multi-token hit whose tokens are scattered lost its citations
        entirely.

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

        Both MATCH construction and citation building start from this ONE
        list. ``_fts5_match_expressions`` builds the MATCH expression from it
        -- the full list under ``and`` (and so for the shipped
        ``and_then_prefix``'s PRIMARY), a content-token SUBSET under
        ``and_stopword_trim``, ``or`` and ``prefix``, and that same subset in
        the fallbacks of ``and_then_or`` and ``and_then_prefix``, which is
        where the subset case arises on the default path (see that method)
        -- while ``_keyword_citation_spans`` always locates the spans from
        the FULL list, so a matched row's citations can cite tokens the
        MATCH itself did not require. They used to tokenize independently
        (per-token quoting on one side, a raw whole-query substring lookup on
        the other), which is exactly how a row could match the query and
        then be reported with no evidence for it -- see
        ``_keyword_citation_spans``.

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
        # search; this helper also runs once per RESULT ROW. The bound stays
        # HERE rather than in the shared tokenizer: it is this service's
        # configured limit, and the Library's four-seam path -- the other
        # consumer since TASK-17755 -- validates against its own.
        return fts5_query_tokens(query[:MAX_QUERY_LENGTH])

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

        **TASK-15400 (2026-08-11) took this full AND off the default; then
        TASK-15700 (2026-08-13) put it back as the default's PRIMARY.** The
        expression this method builds is now the primary form of the shipped
        ``and_then_prefix`` construction, so a default search once again ANDs
        EVERY token -- and widens to the content-token PREFIX form only in a
        sub-leg whose primary returned zero rows. It also still ships on
        three other live paths: the ``and`` construction, the fail-safe an
        unrecognized ``fts_match_construction`` degrades to, and
        ``and_stopword_trim``'s own fallback when trimming empties the query
        (see ``_fts5_match_expressions``).

        A consequence to state rather than leave to be rediscovered: because
        the default's primary is the FULL AND and not the trimmed one, the
        shipped construction is **not a superset of the 2026-08-11 default
        by construction** -- a sub-leg whose full AND returns rows never
        seeks the trim-only hits. That it loses nothing is a MEASURED result
        on the golden corpus (``lost`` = 0 against both the pre-arc control
        and the outgoing default), not a structural guarantee.

        The arc's sweep measured all four pre-registered constructions over
        the RAG_Eval golden set at the shipped fusion parameters and applied
        a pre-registered rule; these are that matrix's cells, not arithmetic
        on top of it (sweep row ``and_trim``, `Tests/RAG_Eval/harness/
        fusion_sweep.py`):

        * **What the winner bought.** Keyword-leg census 20 -> 21 of the 53
          non-negative golden queries (the one rescue is
          ``pm-vendor-chaser``; ``_FTS5_STOPWORDS``' comment records which
          function word was blocking it), and hybrid ``prompt`` recall
          0.000 -> 0.200 (mrr 0.022, ndcg 0.060) -- the category that
          surfaced the arc at all, because prompts have no semantic leg to
          hide the keyword leg's misses. NO cell moved DOWN in any category
          in any mode, and plain/semantic are byte-identical to ``and``
          (only hybrid can move). It issues zero extra FTS queries.
        * **What it did NOT buy, and who owns that.** The arc opened on
          "this leg returns ZERO rows for 40 of 60 golden queries"; the
          winner moves that to **39 of 60**. What is left is not function
          words -- the census's measured blockers are absent CONTENT words
          (``template``, ``building``, ``rough``, ``turns``, ``pulls``,
          ``builds``), which no stopword list removes -- and it belongs to
          the RE-SCOPED merge-level follow-up filed off TASK-15400
          (**TASK-15700**; see the next point for why it is a merge problem
          rather than a MATCH problem).
        * **What else was measured, so these four rows do not read as the
          only options.** Reusing the Library four-seam path's
          ``build_fts_match_query`` was measured at authoring time and
          rescues 1 of the 40 (it is AND-joined too). Two report-only probes
          ran over the same 40 zero-row queries: proximity (``NEAR``)
          rescues 0 -- it can only narrow, so it is a subset of the trimmed
          AND by construction -- and prefix matching rescues 3, the only
          unexplored variant beating the winner, held back by the arc's
          pre-registered promotion bar and carrying the same unmeasured
          displacement risk as the OR forms.
        * **The OR forms were measured and DISQUALIFIED -- for two
          DIFFERENT reasons, one of them the merge.** ``or`` (census 28) and
          ``and_then_or`` (census 29) both scored far higher AND both lost
          the golden set's vector-blind fixture's hybrid rescue, with scoped
          recall 1.000 -> 0.429 -- but not by the same mechanism, and the
          distinction is the whole finding:

          - Under ``or`` the JOIN itself loses the fixture: the leg returns
            ten OR rows and the target is not among them (leg rank 11 in the
            k=20 fusion window, fused 0.0188, an order below the cut). No
            merge behaviour is implicated.
          - Under ``and_then_or`` the fixture's own sub-leg WAS UNTOUCHED --
            its row was still an AND row, at leg rank 2 -- and it was the
            MERGE that lost it. ``_keyword_search`` merged the four FTS
            sub-legs with a single ``interleave_rankings`` round-robin (NOT
            a score merge), so when OTHER sub-legs fell back they injected
            rows that displaced the untouched AND row from leg rank 1 to 2,
            and hybrid fusion consumes LEG RANK. That one position was the
            whole distance between rescued and gone. The same displacement
            decomposed the scoped collapse exactly (the 4 note-targeted
            scoped queries fell behind a media fallback row; media was first
            in the round-robin; 3 of 7 = 0.429).

          **That merge mechanism is RETIRED (TASK-15700 Part A).** The
          second bullet is written in the past tense because
          ``_keyword_search`` now tiers its sub-legs by form: every
          primary-form sub-leg round-robins ahead of every sub-leg that fell
          back, and fallback rows fill only the slots the primary tier left,
          so a widening construction's fallback rows can no longer demote
          another sub-leg's untouched primary row. It is kept here because
          it is the measured provenance of that fix.

          The FIRST bullet is NOT retired: under ``or`` the join still loses
          the fixture before any merge is reached. So a widening
          construction that keeps its own AND rows no longer has to fix the
          merge -- the merge is fixed -- while one that widens as its
          PRIMARY form (``or``, and any prefix-style primary) is something
          tiering cannot protect by construction, and still has to earn its
          cells in the sweep.

        **TASK-15700's RE-RUN (2026-08-13), from its own matrix.** With the
        merge fixed, the sweep re-ran as SIX rows at the shipped fusion
        parameters. Cells, not prose (census of 53 non-negative golden
        queries / rescues vs the pre-arc control / census hits LOST vs that
        control / rescue of the vector-blind fixture):

        * ``and`` 20 / 0 / 0, rescue yes -- the pre-arc control.
        * ``and_stopword_trim`` 21 / 1 / 0, rescue yes -- the outgoing
          default.
        * ``or`` 28 / 9 / **1**, rescue **NO** -- disqualified on (a) AND
          (b); the one row it loses IS the vector-blind fixture.
        * ``and_then_or`` 29 / 9 / 0, rescue yes @ slot 10 -- **the
          census-maximal row, DISQUALIFIED on (b)**: 8 gated cells past
          0.02, 5 past the 0.05 fail band (paraphrase and
          vocabulary_mismatch mrr/ndcg, overall.mrr -0.056).
        * ``prefix`` 23 / 3 / 0, rescue yes @ slot 9 -- qualifies.
        * ``and_then_prefix`` 23 / 3 / 0, rescue yes @ slot 9 -- qualifies.

        **Why the biggest census does not ship, mechanically** (the merge
        fix's boundary, and the reason this is a FUSION finding rather than
        a tiering shortfall): tier 2 confines ``and_then_or``'s fallback rows
        inside the KEYWORD LEG, but tier 2 still enters fusion, and there a
        fallback row that ALSO carries a vector rank becomes a MERGED row
        scoring far above any fts-only row. Measured on the fixture's own
        query, a brand-new merged row (fts 11 + vec 13) inserts at slot 9 and
        pushes the fixture to slot 10. Sharper still: all five regressing
        queries have an EMPTY keyword leg under the outgoing default, so
        every sub-leg falls back, the leg is 100% tier 2, tier 1 is empty and
        the partition is the IDENTITY function -- Part A is structurally
        inert on exactly the queries that disqualify the row, so no tiering
        change could have saved it. Those 8 census points are a fusion-
        weighting question, out of this arc's scope.

        **THE DECISION, and the fact that it was overridden.** The rule was
        applied verbatim: qualifiers {``and_stopword_trim`` 21, ``prefix``
        23, ``and_then_prefix`` 23}; max census 23 TIED the two prefix-
        bearing rows, which were verified measurement-identical on every
        captured axis (all 105 gated cells unmoved, all 60 per-query hybrid
        top-10s and all 60 keyword-leg top-10s identical, same rescued
        queries, ``lost`` 0 both ways); and the rule's tie-break -- fewest
        extra FTS statements, MEASURED at 240 vs 460 over the 60-query set --
        selected **``prefix``**. **The OWNER RULED ``and_then_prefix`` ships
        instead**, applying the standing stability-over-quick-wins ruling to
        a dimension the tie-break predates: ``prefix`` widens as the PRIMARY
        form, so its widened rows compete for their own sub-leg's bm25-
        ordered, LIMITED slots BEFORE the merge is consulted and tiering
        protects nothing (measured synthetically: 12 prefix-competitor docs
        + 1 exact-match doc, "wombat log" at top_k=5 -- the trimmed AND finds
        the exact doc, ``prefix`` returns 5 rows without it), while
        ``and_then_prefix`` never widens a NON-EMPTY AND primary and confines
        widening rows to tier 2. Price: 220 extra SQLite statements on this
        corpus (92% of sub-legs falling back on the 172-document eval
        corpus, an upper bound that shrinks as a corpus densifies), wall time
        indistinguishable, ZERO measured retrieval difference.
        **``and_then_prefix`` is therefore NOT the rule's own output and must
        never be described as such.**

        **What the shipped flip buys, and the residual bound.** Keyword-leg
        census 21 -> **23 of 53** (+``kw-quillon-mast``,
        +``kw-thimble-relay``), and zero-row queries 39 -> **36 of 60**. NO
        gated cell moves in any mode (0 of 105): both new hits are queries
        the vector leg already ranks highly, so the gain is leg-level and
        shows up only where the vector leg is blind, absent or scoped away.
        The residual 36 are still blocked by absent CONTENT words.

        **The vector-blind fixture, framed correctly.** Under the shipped
        construction it holds slot **9 of 10**, exactly where the outgoing
        default put it -- the flip does not move it. The row immediately
        below is a **MATHEMATICAL TIE**: the fixture scores
        ``(1-alpha)/(rrf_k+1) = 0.3/6`` and that row ``alpha/(rrf_k+9) =
        0.7/14``, and ``0.3/6 == 0.7/14 == 1/20`` EXACTLY in rational
        arithmetic (4 ULPs apart in IEEE-754). The fixture keeps its slot on
        ``reciprocal_rank_fusion``'s documented ``(-score, fts_rank,
        vector_rank)`` tie-break -- an fts_rank-1 row ahead of a row with no
        fts_rank at all -- **not on any margin**. So the fixture sits ON the
        alpha/rrf_k boundary, tie-broken in its favour; nothing in this arc
        adds headroom and the shipped construction spends none. What would
        displace it is a MERGED (fts+vector) row, which is exactly what a
        widening PRIMARY manufactures. Fusion parameters own that margin and
        remain out of scope.

        Whatever changes here, keep each token individually quoted (the
        injection property above, pinned by
        ``Tests/RAG_Search/test_fts5_query_escaping.py``, which re-runs it
        through every construction and through the fallback expression).

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
            self._quote_fts5_token(token)
            for token in self._fts5_query_tokens(query)
        ]

        # FTS5 joins space-separated quoted terms with an implicit AND.
        return " ".join(quoted_tokens)

    @staticmethod
    def _quote_fts5_token(token: str) -> str:
        """Quote ONE query token as an FTS5 string literal.

        The single place the injection-safety property of TASK-3995 is
        implemented, so every MATCH construction inherits it rather than
        re-deriving it: a bare token FTS5 parses as column-filter or
        operator syntax (``Obsidian-3`` raises
        ``OperationalError('no such column: 3')``; a typed ``OR`` becomes a
        disjunction of the user's own words), while a quoted token is a
        literal string with no operator semantics. An embedded double quote
        is doubled, FTS5's own escape for a literal quote inside a quoted
        term.

        Args:
            token: One raw token from ``_fts5_query_tokens``.

        Returns:
            The token as a quoted FTS5 term.
        """
        return quote_fts5_token(token)

    @staticmethod
    def _is_fts5_stopword(token: str) -> bool:
        """Whether a raw query token is a function word.

        Consulted by every FORM except the full AND (``and``, the
        pre-TASK-15400 one) -- so it runs on the content-token AND
        (``and_stopword_trim``), on the content-token OR (``or`` /
        ``and_then_or``'s fallback), and on the content-token PREFIX form
        (``prefix`` / ``and_then_prefix``'s fallback, TASK-15700).

        On the SHIPPED default path (``and_then_prefix`` since 2026-08-13)
        that means it is consulted only for sub-legs whose PRIMARY returned
        zero rows: the default's primary is the full AND, which never
        consults it. Under the previous default (``and_stopword_trim``) it
        ran on every search instead.

        The prefix form is where trimming matters MOST: ``"the"*`` matches
        "the", "then", "there", "their", "these" -- nearly a whole corpus --
        and the prefix form ANDs its terms, so one untrimmed function word
        does not merely add noise, it drags the whole expression toward
        matching everything the other terms allow. That is why the prefix
        constructions answer ``""`` when trimming empties the list instead of
        falling back to the full AND (see ``_fts5_match_expressions``).

        Compared on the token's alphanumeric runs, because that is how FTS5
        reads a quoted token: ``About,`` indexes and matches exactly as
        ``about``, so the trimmer must see them as the same word. A token
        with more than one run (``read-only``) is content, never a stopword.

        Args:
            token: One raw token from ``_fts5_query_tokens``.

        Returns:
            True when the token is a single alphanumeric run listed in
            ``_FTS5_STOPWORDS``.
        """
        return is_fts5_stopword(token)

    @staticmethod
    def _fts5_term_key(token: str) -> str:
        """What FTS5 actually matches for a quoted token, as a comparable key.

        A quoted token is a phrase over its alphanumeric runs, and the
        default tokenizer folds case -- so ``Wombat``, ``wombat`` and
        ``wombat,`` are all the same query term. Comparing raw token
        STRINGS would miss that, which is how a duplicate-token query
        ("wombat wombat") burns an FTS query on a fallback that cannot
        return a row the primary did not.

        Args:
            token: One raw token from ``_fts5_query_tokens``.

        Returns:
            The token's runs, space-joined and case-folded.
        """
        return " ".join(fts5_token_runs(token)).lower()

    def _resolved_fts_match_construction(self) -> str:
        """The active MATCH construction, or ``"and"`` for anything unknown.

        Resolved at use time (like ``resolve_rrf_k`` and
        ``resolve_hybrid_alpha``), so an unrecognized value degrades with one
        warning instead of crashing a search or silently retrieving
        differently. Warned once per service instance per bad value -- this
        runs on every search, and a per-call warning would bury the log.

        NOTE the fail-safe target is ``"and"``, the PRE-TASK-15400 full AND
        -- deliberately NOT the shipped default (``and_then_prefix`` since
        2026-08-13, ``and_stopword_trim`` before that). A bad value should
        land on the most conservative construction the engine has, which is
        the one that never widens a query; and it keeps this fail-safe
        stable as the measured default moves -- which it now has twice,
        exactly the drift this note was written to survive.

        Returns:
            One of ``FTS_MATCH_CONSTRUCTIONS``.
        """
        construction = getattr(
            self.config.search, "fts_match_construction", FTS_MATCH_CONSTRUCTION_AND
        )
        if construction in FTS_MATCH_CONSTRUCTIONS:
            return construction

        # `SearchConfig` is built from an untyped dict (`SearchConfig(**search_data)`
        # in config.py, reachable from user-editable profile JSON), so `construction`
        # can be a list/dict here -- unhashable, which would raise TypeError on the
        # set membership/add below and crash hybrid AND keyword search instead of
        # degrading to "and". Route non-str values through a hashable surrogate key
        # for the warn-once set; a str value keeps using itself, unchanged.
        dedup_key = (
            construction
            if isinstance(construction, str)
            else f"{type(construction).__name__}:{construction!r}"
        )

        if dedup_key not in self._warned_fts_constructions:
            self._warned_fts_constructions.add(dedup_key)
            logger.warning(
                "Unknown fts_match_construction; using conservative fallback"
            )
        return FTS_MATCH_CONSTRUCTION_AND

    def _fts5_match_forms(self) -> Tuple[str, Optional[str]]:
        """The FORMS the active construction's two expressions run.

        `FTS_MATCH_FORMS_BY_CONSTRUCTION` read through
        `_resolved_fts_match_construction`, so an unrecognized value takes
        the forms of the conservative ``and`` the leg actually ran rather
        than of a construction nothing executed. Shaped exactly like
        `_fts5_match_expressions`' return so the two line up positionally:
        ``(primary_form, fallback_form)``, the fallback ``None`` when the
        construction has none.

        Returns:
            ``(primary_form, fallback_form)`` for the active construction.
        """
        return FTS_MATCH_FORMS_BY_CONSTRUCTION[
            self._resolved_fts_match_construction()
        ]

    def _fts5_primary_form(self) -> str:
        """The FORM the active construction's PRIMARY expression runs.

        ONE definition of "which form is not a fallback" (TASK-15700),
        derived from the same table the row stamp uses. Consumers that must
        never disagree: `_keyword_search`'s tier partition, which puts
        primary-form sub-legs ahead of fallback ones, and
        `_keyword_row_metadata`, whose default for a row that arrived
        unstamped has to be the value that keeps it in tier 1.

        Fallback-ness is f(construction, form), never the stamp alone --
        under the `or` construction the OR form IS the primary, so tiering
        on the stamp by itself would demote every row that construction
        returns.

        Returns:
            The construction's primary form from
            ``FTS_MATCH_FORMS_BY_CONSTRUCTION`` — ``FTS_MATCH_OR`` under
            ``or``, ``FTS_MATCH_PREFIX`` under ``prefix``, and
            ``FTS_MATCH_AND`` under every AND-primary construction
            (including the shipped ``and_then_prefix``). Enumerating the
            table here went stale once already; the table is the truth.
        """
        return self._fts5_match_forms()[0]

    def _fts5_match_expressions(self, query: str) -> Tuple[str, Optional[str]]:
        """Build the MATCH expression(s) one search runs (TASK-15400).

        The construction seam: ONE definition consumed by all four FTS
        sub-legs (media, notes, conversations, prompts). Every token in
        every form is individually quoted by ``_quote_fts5_token`` -- only
        the JOIN between tokens is in play here, never the quoting.

        The pre-registered candidates
        (``SearchConfig.fts_match_construction``) -- TASK-15400's four, then
        TASK-15700's two:

        * ``and`` (pre-TASK-15400; still the fail-safe for an unrecognized
          value) -- implicit AND over every token. Byte-identical to
          ``_escape_fts5_query``, which it delegates to.
        * ``and_stopword_trim`` (TASK-15400's measured winner; **the shipped
          default 2026-08-11 -> 2026-08-13**) -- implicit AND over the
          CONTENT tokens, falling back to the full AND when trimming empties
          the token list (an empty MATCH expression is an FTS5 syntax error,
          not "no results").
        * ``or`` -- the content tokens joined by FTS5's ``OR`` operator.
          Stopwords are trimmed here because a raw OR of every token matches
          every document containing "the"; when trimming empties the list
          the answer is honestly no rows (``""``, the existing skip
          contract), never a syntax error.
        * ``and_then_or`` -- both: the AND form as primary, the OR form as
          the fallback each sub-leg runs ONLY when its own primary returns
          zero rows. A non-empty AND is therefore never widened, which
          preserves every hit the full AND finds within a SUB-LEG -- but
          NOT, as the sweep measured, every hit the LEG contributes to
          fusion: the round-robin merge re-ranks the untouched rows. That
          is why this candidate was disqualified (see
          ``_escape_fts5_query``) -- and why TASK-15700 fixed that merge and
          re-ran the sweep with the two rows below added.
        * ``prefix`` (TASK-15700) -- the content tokens as PREFIX terms,
          space-joined: an implicit AND over ``"tok"*``. Byte-identical to
          the expression the 15400 sweep's report-only probe measured its
          3-rescue lead on (``prefix_probe_expression``,
          `Tests/RAG_Eval/harness/fusion_sweep.py`); reproducing THAT form
          is what makes the lead evidence for this construction rather than
          for a different query. Stopwords are trimmed for a sharper reason
          than under ``or``: a stopword prefix (``"the"*``) matches most of a
          corpus. When trimming empties the list the answer is honestly no
          rows (``""``). It widens as a PRIMARY form, so the tiered merge
          cannot protect the untouched AND rows from it -- the sweep's matrix
          has to show what it displaces.
        * ``and_then_prefix`` (TASK-15700; **the shipped default since
          2026-08-13**) -- the AND form as primary, the prefix form as the
          per-sub-leg zero-row fallback: the composition with BOTH
          protections, every AND hit preserved inside a sub-leg by
          construction and the widened rows confined behind the primary ones
          by the tiered merge. It ships by OWNER RULING, not as the
          sweep's computed winner -- the pre-registered rule's tie-break
          selected the measurement-identical ``prefix`` on statement count
          (240 vs 460) and the owner overrode it for that structural
          protection; ``_escape_fts5_query`` carries the full record. Note
          its primary is the FULL AND, so unlike ``and_stopword_trim`` a
          default search does not trim unless a sub-leg finds nothing.

        The OR fallback is suppressed when it cannot widen anything -- when
        both forms reduce to the same single FTS5 term -- since re-running
        it costs one query per zero-row sub-leg and can only return the same
        zero rows. That test belongs to the OR composition alone: a PREFIX
        fallback over identical terms is STRICTLY wider, so
        ``and_then_prefix`` suppresses only an empty prefix expression.

        Args:
            query: Raw search query.

        Returns:
            ``(primary, fallback)``. ``primary == ""`` means "no rows" and
            callers must skip the FTS5 query entirely (the pre-existing
            contract). ``fallback is None`` means "never widen".
        """
        construction = self._resolved_fts_match_construction()
        and_expression = self._escape_fts5_query(query)

        if construction == FTS_MATCH_CONSTRUCTION_AND:
            return and_expression, None

        tokens = self._fts5_query_tokens(query)
        content_tokens = [
            token for token in tokens if not self._is_fts5_stopword(token)
        ]
        quoted = [self._quote_fts5_token(token) for token in content_tokens]

        if construction == FTS_MATCH_CONSTRUCTION_AND_STOPWORD_TRIM:
            # Trimming everything away leaves the FULL AND, not "" -- so an
            # only-function-word query is byte-identical to the pre-arc
            # construction rather than a syntax error or an empty answer.
            return " ".join(quoted) or and_expression, None

        if construction in (
            FTS_MATCH_CONSTRUCTION_PREFIX,
            FTS_MATCH_CONSTRUCTION_AND_THEN_PREFIX,
        ):
            # The ONE prefix-form builder (TASK-17755 moved it to
            # `Utils/fts5_match_forms.py`; the Library's four-seam path runs
            # the same form for the same reason). It re-applies the stopword
            # trim, which is idempotent on `content_tokens` -- so this is
            # byte-identical to the inline construction it replaced, pinned
            # by `test_the_engine_and_the_library_build_the_same_prefix_form`.
            prefix_expression = build_prefix_match_expression(content_tokens)

            if construction == FTS_MATCH_CONSTRUCTION_PREFIX:
                # Trimming empties -> "" (the skip contract), never the full
                # AND: a stopword PREFIX (`"the"*`) matches most of a corpus,
                # so the honest answer is no rows, exactly as under `or`.
                return prefix_expression, None

            # and_then_prefix. The fallback is suppressed ONLY when there is
            # nothing to widen to (an empty prefix expression) -- never on
            # the term-set equality `and_then_or` uses. A prefix form over
            # the same terms is a STRICTLY WIDER query ("wombat" vs every
            # word starting with it), so the OR composition's "both forms
            # reduce to one identical term" reasoning does not transfer:
            # copying it would silence the fallback on exactly the
            # single-content-token queries the 15400 probe's rescues came
            # from.
            return and_expression, prefix_expression or None

        or_expression = " OR ".join(quoted)

        if construction == FTS_MATCH_CONSTRUCTION_OR:
            return or_expression, None

        # and_then_or. The fallback is suppressed when it cannot possibly
        # widen: that is when both forms reduce to ONE identical FTS5 term
        # (`OR` over a single term is that term, and an implicit AND over
        # repetitions of it is too). Compared on TERM keys, not on the
        # expression strings, so "wombat wombat" and "wombat, Wombat" are
        # recognized as the single-term queries FTS5 reads them as instead
        # of burning a redundant query per zero-row sub-leg.
        and_terms = {self._fts5_term_key(token) for token in tokens}
        or_terms = {self._fts5_term_key(token) for token in content_tokens}
        if not or_expression or (len(or_terms) == 1 and or_terms == and_terms):
            return and_expression, None
        return and_expression, or_expression

    def _fts_rows_with_fallback(
        self,
        run_expression: Callable[[str], List[Dict[str, Any]]],
        expressions: Tuple[str, Optional[str]],
    ) -> List[Dict[str, Any]]:
        """Run one sub-leg's FTS query, widening only when it finds nothing.

        Wraps a sub-leg's SQL-executing helper (never the tokenizer), so
        each sub-leg decides independently whether to widen: one query can
        legitimately carry AND rows from one sub-leg and OR rows from
        another, which is why every row is stamped with the form that
        matched it. That mix is TIERED, not interleaved (TASK-15700): the
        merge in `_keyword_search` puts every primary-form sub-leg ahead of
        every fallback sub-leg, because a fallback row taking leg rank 1
        from an untouched primary row is what cost the vector-blind fixture
        its hybrid rescue. Widening is still decided per sub-leg here; where
        the widened rows LAND is decided there.

        Because the fallback runs only when the primary returned zero rows,
        a sub-leg's rows are all-primary or all-fallback within one query --
        the all-or-nothing fact that lets the merge tier whole sub-legs
        instead of individual rows.

        Args:
            run_expression: Executes ONE MATCH expression and returns its
                rows (already degrading to ``[]`` on a database error --
                the fallback inherits that path unchanged, adding no new
                failure modes).
            expressions: ``_fts5_match_expressions``' ``(primary,
                fallback)`` pair.

        Returns:
            The rows, each carrying an ``fts_match`` key naming the form
            that matched them -- the active construction's primary form, or
            its fallback form when the fallback ran, both read from
            ``FTS_MATCH_FORMS_BY_CONSTRUCTION``. ``_keyword_row_metadata``
            promotes the key into the row's metadata.
        """
        primary, fallback = expressions
        rows = run_expression(primary) if primary else []
        # BOTH stamps come from the construction's own row in
        # `FTS_MATCH_FORMS_BY_CONSTRUCTION`, never from a hardcoded branch:
        # the stamp names the form, not the position (see FTS_MATCH_AND), so
        # the primary is an OR form under `or` and the fallback need not be
        # an OR form at all. The merge's tier partition reads the same
        # table, so the stamp and the tiering cannot disagree about which
        # form was the primary.
        primary_form, fallback_form = self._fts5_match_forms()
        form = primary_form

        if not rows and fallback:
            rows = run_expression(fallback)
            if fallback_form is None:
                # The construction produced a fallback EXPRESSION while
                # naming no fallback FORM -- a missed
                # `FTS_MATCH_FORMS_BY_CONSTRUCTION` entry, unreachable while
                # `test_every_construction_names_the_forms_it_can_run` is
                # green. Degrade to the OR form (the only fallback the
                # engine has ever shipped) and say so once, rather than
                # stamping the PRIMARY form and letting a widened row into
                # tier 1.
                logger.warning(
                    "Construction {!r} ran a fallback expression but names "
                    "no fallback form in FTS_MATCH_FORMS_BY_CONSTRUCTION; "
                    "stamping its rows {!r}. The sweep's form attribution "
                    "for this construction is not trustworthy until the "
                    "table names it.",
                    self._resolved_fts_match_construction(),
                    FTS_MATCH_OR,
                )
                form = FTS_MATCH_OR
            else:
                form = fallback_form

        for row in rows:
            row["fts_match"] = form
        return rows

    @staticmethod
    def _keyword_citation_spans(
        content: str, tokens: List[str]
    ) -> List[Tuple[int, int, frozenset]]:
        """Locate the citation spans for a keyword hit, from the query's tokens.

        TASK-3996 follow-up (Qodo, PR #1469). Before TASK-3995 the keyword
        leg used phrase semantics, so a hit guaranteed the raw query was one
        contiguous substring of the document and the citation builder could
        just look that raw query up. Per-token implicit AND deleted that
        guarantee: documents now match with the tokens scattered, the raw
        lookup found nothing, and the rows the fix had just made reachable
        came back with ``citations=[]``.

        Spans are located per token, case-insensitively, from the FULL token
        list ``_fts5_query_tokens`` returns for the query (not necessarily
        the subset ``_fts5_match_expressions`` required for the MATCH itself
        when a widening form ran -- under the shipped ``and_then_prefix``
        that is its prefix fallback; see that method). A token is matched as
        its alphanumeric runs separated by non-alphanumerics
        ("Obsidian-3" -> ``Obsidian`` then ``3``), which is
        how FTS5 reads a quoted token: a phrase over the runs, adjacency
        required.

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
        self,
        pool,
        query: str,
        limit: int,
        allowed_ids: Optional[Collection[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform FTS5 search using connection pool with proper SQL injection prevention.

        Args:
            pool: Connection pool instance
            query: Search query
            limit: Maximum number of results
            allowed_ids: Optional media ids this query may return
                (TASK-15020/B1). Bound as ONE json_each parameter, never a
                placeholder per id -- see ``_json_id_param``. ``None`` leaves
                the query shape byte-identical to what it was before scopes
                reached this leg.

        Returns:
            List of search results
        """
        # Properly escape the query for FTS5, in whichever MATCH
        # construction is active (TASK-15400/15700). Under `and` this is
        # `_escape_fts5_query` verbatim with no fallback; under the SHIPPED
        # `and_then_prefix` it is that same full AND as the primary WITH a
        # prefix fallback, which this sub-leg runs if the primary returns
        # zero rows. The compositions (`and_then_or`, `and_then_prefix`) are
        # the ones that return a second expression.
        expressions = self._fts5_match_expressions(query)

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
        # Only "?" placeholders are interpolated into the SQL; every value,
        # including the whole id allowlist, is bound.
        id_filter_sql = ""
        if allowed_ids is not None:
            id_filter_sql = "AND m.id IN (SELECT value FROM json_each(?))"

        sql = f"""
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
        {id_filter_sql}
        ORDER BY media_fts.rank
        LIMIT ?
        """

        def _run(escaped_query: str) -> List[Dict[str, Any]]:
            params: List[Any] = [escaped_query]
            if allowed_ids is not None:
                params.append(_json_id_param(allowed_ids))
            params.append(limit)

            results = []
            try:
                # Use transaction for consistent read
                with pool.transaction() as conn:
                    cursor = conn.cursor()
                    # Use parameterized query - the escaped_query is already safe
                    cursor.execute(sql, tuple(params))

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

        # The zero-row fallback lives INSIDE this helper so the caller's
        # retry wrapper cannot multiply it (TASK-15400).
        return self._fts_rows_with_fallback(_run, expressions)

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
