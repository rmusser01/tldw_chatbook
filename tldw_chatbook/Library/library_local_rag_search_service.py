"""Local production backend for Library Search/RAG retrieval."""

from __future__ import annotations

import asyncio
from enum import Enum
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from functools import partial
from typing import Any, Hashable, Optional

from loguru import logger

from tldw_chatbook.Chat.rag_scope import (
    EffectiveScope,
    SCOPE_REASON_CONVERSATIONS_EXCLUDED,
    SCOPE_REASON_PROMPTS_EXCLUDED,
    SCOPE_STATUS_EXCLUDED,
    SOURCE_TYPE_MEDIA,
    SOURCE_TYPE_NOTE,
    build_semantic_allowlists,
    media_id_params,
    note_id_params,
)
from tldw_chatbook.Library.library_fts_query import (
    build_fts_match_query,
    build_prefix_match_query,
)
from tldw_chatbook.Library.library_rag_service import LibraryRagSearchOutcome
from tldw_chatbook.Library.library_rag_state import (
    LIBRARY_RAG_EMPTY_STATE_SELECTOR,
    LIBRARY_RAG_QUERY_MAX_LENGTH,
    LIBRARY_RAG_ROUTE_NOTES_KEY,
    LIBRARY_RAG_SERVICE_ERROR_SELECTOR,
)

#: Diagnostics slot for per-seam keyword outcomes (TASK-18903). Mirrors
#: ``SCOPE_DIAGNOSTICS_KEY``/``SEMANTIC_DIAGNOSTICS_KEY``: a LIST of
#: ``{"status", "seam", "message"}`` entries, APPENDED never assigned, so two
#: failing seams in one call cannot overwrite each other (the task-9 review
#: finding that shaped the scope slot applies identically here).
KEYWORD_SEAM_DIAGNOSTICS_KEY = "keyword_seams"

#: Status value used inside that slot.
SEAM_STATUS_FAILED = "failed"


class SeamState(Enum):
    """Outcome of one keyword seam.

    The boolean this replaces could not tell "not configured" from "ran and
    threw": both flowed on as *some* value alongside an empty row list, and a
    thrown seam reported itself AVAILABLE. That collapse let a total backend
    outage present as a successful search with zero results (TASK-18903), and
    it is the same shape that produced TASK-17855's wrong defect filing and
    TASK-18255.

    NOTE for anyone reading the gate: every ``Enum`` member is TRUTHY,
    including ``UNAVAILABLE`` and ``FAILED``. A guard written as
    ``if not state`` is always False and silently inert -- compare with ``is``.
    """

    #: Configured, ran, and its rows are its answer (possibly none).
    AVAILABLE = "available"
    #: Not configured in this runtime -- nothing was searched.
    UNAVAILABLE = "unavailable"
    #: Configured, ran, and RAISED. Its empty rows mean nothing.
    FAILED = "failed"


# Single source of truth for the pipeline diagnostics "scope" slot key
# (task-4/Backend A); reused here so the Library service's own
# conversations-excluded notice uses the exact same vocabulary rather than a
# parallel raw literal.
from tldw_chatbook.RAG_Search.pipeline_functions_simple import SCOPE_DIAGNOSTICS_KEY

# The engine's own rank-fair round-robin, imported rather than re-implemented
# (TASK-16071): the four-seam keyword merge has exactly the problem this
# primitive exists for -- several per-source rankings whose raw scores are not
# comparable. See the rule written at the merge site in `_search_keyword`.
from tldw_chatbook.RAG_Search.fusion import interleave_rankings

# The shared factory is the single process-wide RAG service constructor
# (task-247): resolving through it guarantees Library RAG Answer queries read
# the exact vector store / collection / embedding model that ingestion-time
# indexing writes to.
from tldw_chatbook.RAG_Search.ingestion_indexing import (
    get_shared_rag_service,
    shared_rag_service_generation,
)
from tldw_chatbook.RAG_Search.simplified.active_config import (
    normalize_rag_search_mode,
)

# One staleness rule for the `app._rag_service` cache, shared with the chat/
# Search resolver (`resolve_semantic_rag_service`): a profile switch resets
# the shared singleton, and both app-level caches must notice.
from tldw_chatbook.RAG_Search.semantic_availability import (
    cache_app_rag_service,
    current_app_rag_service,
)
from tldw_chatbook.UI.destination_recovery import DestinationRecoveryState
from tldw_chatbook.Utils.input_validation import sanitize_string, validate_text_input
from tldw_chatbook.Utils.optional_deps import embeddings_rag_deps_installed

logger = logger.bind(module="LibraryLocalRagSearchService")

_SEARCH_RUNTIME_BACKEND = "local-fts"
_RAG_RUNTIME_BACKEND = "rag-semantic"
# Mode-truthful backend label for the engine's RRF-fused hybrid path (spec
# Workstream A item 3): a hybrid search must never report itself as
# "rag-semantic", since the two produce different score kinds.
_RAG_HYBRID_RUNTIME_BACKEND = "rag-hybrid"
_KNOWN_KEYWORD_SOURCE_TYPES = ("notes", "media", "conversations", "prompts")
# The active RAG profile's `default_search_mode` vocabulary
# (`RAG_Search/simplified/config.py::SearchConfig`). Anything else -- a
# hand-edited TOML, a future mode this build does not know -- resolves to
# "semantic", the historical behavior.
# Routing disclosures (spec Workstream A: "each handled by disclosure
# rather than silence"). Lowercase fragments; `library_rag_state`'s
# `_route_note_sentence` renders them as sentences on the Evidence
# region's one quiet coverage line.
ROUTE_NOTE_PLAIN_PROFILE_TEMPLATE = "{profile}: keyword search (no vectors)"
# (TASK-15020/B1) `ROUTE_NOTE_HYBRID_SCOPED` ("scope active — semantic only
# until scope-aware hybrid lands") lived here. That landing is done: scope
# allowlists now reach BOTH engine legs, so a scoped query under a hybrid
# profile runs the profile's own fused search and has no divert to disclose.
# The constant is deliberately not kept as an alias -- a disclosure nothing
# can emit is prose that outlives its fact.
# (TASK-15020/B2) NARROWED, not retired. Until B2 this fired for any
# selection that was prompts-only, since prompts had no sub-leg; the engine's
# keyword leg now serves all FOUR of the Search canvas's source types, so no
# combination of real toggles can reach it any more. What remains reachable
# is a selection carrying nothing the leg knows -- an empty selection, or one
# built from identifiers this build does not recognize. The wording still
# describes exactly that case, so it stays as the disclosure for it rather
# than becoming a sentence nothing can emit (`ROUTE_NOTE_HYBRID_SCOPED`'s
# fate, above).
ROUTE_NOTE_HYBRID_NO_KEYWORD_SOURCES = (
    "no keyword leg for the selected sources — semantic only"
)
ROUTE_NOTE_SEMANTIC_LEG_EMPTY = "semantic leg empty — keyword-only results"
# (TASK-18903) A seam that RAN AND THREW contributed nothing, and its silence
# must not read as "that source has nothing". This is the same principle the
# routing disclosures already encode -- and, as their own docstring says, zero
# rows is exactly when it matters most.
ROUTE_NOTE_SEAMS_FAILED_TEMPLATE = "{seams} failed — results exclude them"
# Mirrors `library_rag_state`'s `_OPEN_SOURCE_TYPE_MAP` canonicalization:
# raw provenance `source_type` values -> the scope-toggle identifiers used
# by `LibraryRagScopeState`/the Search canvas's per-source toggles.
_SEMANTIC_SOURCE_TYPE_MAP = {
    "note": "notes",
    "notes": "notes",
    "media": "media",
    "media_chunk": "media",
    "conversation": "conversations",
    "conversations": "conversations",
    "chat": "conversations",
    # TASK-15020/B2. The engine's keyword leg now emits `prompt` rows, and
    # this map is what the source-type POST-FILTER canonicalizes with: an
    # unrecognized provenance value passes the filter untouched
    # (`_semantic_row_matches_scope` returns True), so without this entry a
    # prompt row would survive a selection that had Prompts turned OFF.
    "prompt": "prompts",
    "prompts": "prompts",
}
# The canonical source types the SEMANTIC leg can structurally speak to
# (task-15 finding I2): `prompts` (and `workspaces`/`collections`) has no
# semantic-index seam at all -- no vector row will ever carry it.
# `selected_source_types` reaching `_search_semantic` is the Search canvas's
# full scope, which includes `prompts` under its default (all four toggles
# on, the common case whenever a workspace has >=1 prompt) -- diffing that
# raw scope against `present` in `_semantic_scope_coverage` would flag
# `prompts` "uncovered" on every single non-empty rag-mode query, forever,
# turning a per-query signal ("semantic search looked at X and found
# nothing") into a permanent false nag with the wrong implicature (prompts
# are structurally absent from the semantic leg, not "searched and empty").
#
# **Written out, not derived.** This used to read
# `frozenset(_SEMANTIC_SOURCE_TYPE_MAP.values())`, which was correct only
# while that map's domain happened to be exactly the vector-indexed types.
# TASK-15020/B2 added `prompt`/`prompts` to it for the post-filter, and the
# derivation would have silently re-admitted `prompts` to the coverage
# partition -- reinstating the exact false nag I2 removed, with every test
# green. The two sets answer different questions; only one of them is about
# the vector index.
_SEMANTICALLY_COVERABLE_SOURCE_TYPES = frozenset({"media", "notes", "conversations"})
# The Library scope identifiers the ENGINE's keyword (FTS5) leg can serve --
# `media_fts`, plus `notes_fts`/`messages_fts` since TASK-3996 and
# `prompts_fts` since TASK-15020/B2. Kept separate from
# `_SEMANTICALLY_COVERABLE_SOURCE_TYPES` because they answer different
# questions, and since B2 they genuinely DIVERGE: `prompts` is FTS-servable
# and has no vector index at all, so a prompt reaches hybrid results through
# this leg alone (rescued by the fusion weighting) and must never appear in
# a semantic-coverage claim.
_FTS_SERVABLE_SOURCE_TYPES = frozenset(
    {"media", "notes", "conversations", "prompts"}
)
# Library scope identifier -> the ENGINE's keyword-leg vocabulary
# (`rag_service.SOURCE_TYPE_*`, singular). The Library speaks plurals
# (`notes`, `conversations`); the engine's FTS sub-legs are selected -- and
# its rows stamped -- with the singular ingestion spelling, and a plural
# handed to `keyword_source_types` is simply dropped as unknown, which would
# leave the keyword leg empty rather than scoped. Domain: exactly
# `_FTS_SERVABLE_SOURCE_TYPES`.
_ENGINE_KEYWORD_SOURCE_TYPES = {
    "media": "media",
    "notes": "note",
    "conversations": "conversation",
    "prompts": "prompt",
}


def _validated_query(query: str) -> str:
    """Validate a user query before it reaches retrieval or FTS seams.

    Args:
        query: Raw Library search or RAG query.

    Returns:
        The unchanged query when it passes the shared input validators.

    Raises:
        ValueError: If the query is empty, oversized, contains stripped
            control characters, or fails shared text-safety validation.
    """
    if not isinstance(query, str):
        raise ValueError("Enter a safe Library search query.")
    sanitized = sanitize_string(query, max_length=LIBRARY_RAG_QUERY_MAX_LENGTH)
    if (
        sanitized != query
        or not sanitized.strip()
        or not validate_text_input(
            sanitized,
            max_length=LIBRARY_RAG_QUERY_MAX_LENGTH,
            allow_html=False,
        )
    ):
        raise ValueError("Enter a safe Library search query.")
    return sanitized


class LibraryLocalRagSearchService:
    """Keyword-first Library retrieval over the app's local source seams.

    `search` mode fans out over notes/media/conversations/prompts FTS seams and
    always works when at least one seam is available. `rag` mode uses the
    app's `_rag_service`, lazily initializing it from the process-wide
    shared RAG service on first use when the embeddings deps are installed
    (task-249), and degrades to a blocked outcome with setup routing when
    the runtime is unavailable. Which retrieval `rag` mode actually runs is
    the ACTIVE profile's decision, not a constant -- see `_search_rag`.
    """

    def __init__(self, app_instance: Any) -> None:
        self._app = app_instance

    async def search(
        self,
        query: str,
        source_types: tuple[str, ...],
        mode: str,
        *,
        scope: Optional[EffectiveScope] = None,
        **kwargs: Any,
    ) -> Any:
        """Run a Library-native keyword or RAG retrieval request.

        Args:
            query: User question or search query to run against Library sources.
            source_types: Selected Library source type identifiers (e.g.
                `notes`, `media`, `conversations`). Unknown types are
                ignored quietly.
            mode: Retrieval mode: `search` (keyword, local FTS seams) or
                `rag` (routed by the active RAG profile's
                `default_search_mode` -- keyword seams, semantic, or the
                engine's fused hybrid; see `_search_rag`).
            scope: Optional resolved RAG retrieval scope (rag-scope
                narrowing, task-6). Caller-passed only -- this service never
                resolves scope itself, so a Library-screen call site that
                never passes this keyword gets today's exact unrestricted
                behavior (spec decision D2). `None` or an unscoped scope
                performs unrestricted retrieval; a scoped value restricts
                keyword search to the scope's media/note id allowlists and
                excludes the conversations and prompts seams entirely
                (neither is part of the scope vocabulary, spec D5), and
                restricts semantic search via one store query per
                allowlisted source type, merged by score.
            **kwargs: Backend options. `top_k` caps the result count per
                source (default 5). `include_citations` is used in `rag`
                mode only.

        Returns:
            A mapping with `results`/`runtime_backend` keys for the caller
            to normalize into evidence rows, or a `LibraryRagSearchOutcome`
            directly for blocked/empty states (missing local seams, missing
            RAG runtime, a working search that stayed within scope but
            found nothing).

        Raises:
            ValueError: If `query` fails shared Library input validation, or
                if `scope.state == "empty"` -- callers must short-circuit an
                EMPTY effective scope before ever calling this method (it
                would otherwise search everything, the opposite of EMPTY's
                "nothing left to retrieve from" meaning).
        """
        query = _validated_query(query)
        top_k = max(1, int(kwargs.get("top_k") or 5))
        if scope is not None and scope.state == "empty":
            raise ValueError(
                "LibraryLocalRagSearchService.search() was called with an "
                "EMPTY effective scope; callers must short-circuit before "
                "calling (there is nothing left to retrieve from)."
            )
        if mode == "rag":
            return await self._search_rag(
                query, source_types, top_k, kwargs, scope=scope
            )
        return await self._search_keyword(query, source_types, top_k, scope=scope)

    async def _search_rag(
        self,
        query: str,
        source_types: tuple[str, ...],
        top_k: int,
        kwargs: Mapping[str, Any],
        *,
        scope: Optional[EffectiveScope] = None,
    ) -> Any:
        """Route a `rag`-mode request per the ACTIVE profile's search mode.

        Before this existed the live path hardcoded `search_type="semantic"`:
        a user who selected "BM25 Only" or "Hybrid Basic" in Settings > RAG
        got vector-only retrieval anyway, with nothing on screen saying so
        (spec Workstream A). Routing, with every divergence disclosed
        through the coverage-note channel rather than silently applied:

        - `plain` -> the Library's own four-seam, scope-aware keyword path
          (`_search_keyword`), NOT the engine's keyword leg: the four-seam
          path is scope-aware and also searches prompts, so a BM25 profile
          must not get a strictly worse search in `rag` mode than `search`
          mode already gives it.
        - `hybrid`, with at least one selected source the engine's FTS leg
          can serve -> the engine's fused hybrid, scoped or not. A scope
          used to divert this arm to semantic, because `RAGService.search`
          raised for a non-empty `metadata_allowlist` with any non-semantic
          search type; TASK-15020/B1 pushed allowlists into the FTS
          sub-legs, so the scope now travels with the query instead of
          costing the user their profile's retrieval mode.
        - `hybrid`, no FTS-servable source selected (in practice: prompts
          only) -> semantic. The engine's FTS leg covers media, notes and
          conversations (TASK-3996 added the latter two; before it, this arm
          fired whenever media was deselected); rows from a source outside
          the selection could only be dropped by the source-type
          post-filter, so running the leg would spend a query to produce
          nothing. Any scope still travels with the diverted search.
        - `semantic` (and any unknown mode) -> today's exact behavior.

        Args:
            query: Already-validated user query.
            source_types: Selected Library source type identifiers.
            top_k: Result cap.
            kwargs: Backend options (`include_citations`).
            scope: Optional resolved retrieval scope. It no longer decides
                the route (TASK-15020/B1) -- it is threaded into whichever
                path runs, as the allowlists both engine legs now honor.

        Returns:
            The chosen path's own return shape (mapping or
            `LibraryRagSearchOutcome`), with any routing disclosure attached
            under `diagnostics[LIBRARY_RAG_ROUTE_NOTES_KEY]`.
        """
        rag_service = await self._resolve_rag_runtime()
        if rag_service is None:
            return LibraryRagSearchOutcome(
                status="blocked",
                recovery_state=_rag_mode_unavailable_recovery_state(),
            )
        profile_mode = _resolve_profile_search_mode(rag_service)

        if profile_mode == "plain":
            result = await self._search_keyword(
                query, source_types, top_k, scope=scope
            )
            return _with_route_notes(
                result,
                (
                    ROUTE_NOTE_PLAIN_PROFILE_TEMPLATE.format(
                        profile=_profile_disclosure_label(rag_service)
                    ),
                ),
            )

        if profile_mode == "hybrid":
            if not _FTS_SERVABLE_SOURCE_TYPES.intersection(source_types):
                return await self._search_semantic(
                    query,
                    source_types,
                    top_k,
                    kwargs,
                    scope=scope,
                    rag_service=rag_service,
                    route_notes=(ROUTE_NOTE_HYBRID_NO_KEYWORD_SOURCES,),
                )
            return await self._search_hybrid(
                query,
                source_types,
                top_k,
                kwargs,
                scope=scope,
                rag_service=rag_service,
            )

        return await self._search_semantic(
            query, source_types, top_k, kwargs, scope=scope, rag_service=rag_service
        )

    async def _search_keyword(
        self,
        query: str,
        source_types: tuple[str, ...],
        top_k: int,
        *,
        scope: Optional[EffectiveScope] = None,
    ) -> Any:
        """Fan out a keyword search over the notes/media/conversations/prompts seams."""
        user_id = getattr(self._app, "notes_user_id", None) or "default_user"
        is_scoped = scope is not None and scope.state == "scoped"
        note_allowlist = note_id_params(scope) if scope is not None else None
        media_allowlist = media_id_params(scope) if scope is not None else None

        coroutines: dict[str, Any] = {}
        diagnostics: dict[str, Any] = {}
        if "notes" in source_types:
            if is_scoped and note_allowlist is None:
                # Notes absent from the allowlist under an active scope:
                # empty allowlist for this seam, not "search everything".
                coroutines["notes"] = _empty_scoped_seam()
            else:
                coroutines["notes"] = self._search_notes(
                    query, top_k, user_id, id_allowlist=note_allowlist
                )
        if "media" in source_types:
            if is_scoped and media_allowlist is None:
                coroutines["media"] = _empty_scoped_seam()
            else:
                coroutines["media"] = self._search_media(
                    query, top_k, id_allowlist=media_allowlist
                )
        if "conversations" in source_types:
            if is_scoped:
                # Conversations are not part of the scope vocabulary (spec
                # D5): any active scope excludes this seam entirely rather
                # than searching unrestricted or guessing at an allowlist.
                # Mirrors pipeline_functions_simple's
                # _record_scope_conversations_excluded shape. Appended (not
                # assigned) so a conversations-AND-prompts exclusion under
                # the same scoped call doesn't silently overwrite this one
                # (task-9 review finding).
                diagnostics.setdefault(SCOPE_DIAGNOSTICS_KEY, []).append(
                    {
                        "status": SCOPE_STATUS_EXCLUDED,
                        "reason": SCOPE_REASON_CONVERSATIONS_EXCLUDED,
                    }
                )
            else:
                coroutines["conversations"] = self._search_conversations(query, top_k)
        if "prompts" in source_types:
            if is_scoped:
                # Prompts are not part of the scope vocabulary either (spec
                # D5): mirror the conversations exclusion exactly rather
                # than searching unrestricted or guessing at an allowlist.
                # See the conversations branch above for why this appends.
                diagnostics.setdefault(SCOPE_DIAGNOSTICS_KEY, []).append(
                    {
                        "status": SCOPE_STATUS_EXCLUDED,
                        "reason": SCOPE_REASON_PROMPTS_EXCLUDED,
                    }
                )
            else:
                coroutines["prompts"] = self._search_prompts(query, top_k)

        if not coroutines:
            return LibraryRagSearchOutcome(
                status="blocked",
                recovery_state=_no_backend_recovery_state(),
                diagnostics=diagnostics,
            )

        gathered = await asyncio.gather(*coroutines.values())
        outcomes = dict(zip(coroutines.keys(), gathered))

        # Record every seam that RAN AND THREW, so a partial failure cannot
        # read as "the corpus has nothing" one layer up. Appended to a list
        # for the same reason the scope slot is (task-9 review finding): two
        # seams can fail in one call and neither may overwrite the other.
        failed_seams = sorted(
            name for name, (state, _rows) in outcomes.items()
            if state is SeamState.FAILED
        )
        for name in failed_seams:
            diagnostics.setdefault(KEYWORD_SEAM_DIAGNOSTICS_KEY, []).append(
                {
                    "status": SEAM_STATUS_FAILED,
                    "seam": name,
                    "message": f"The {name} seam failed and returned no rows.",
                }
            )
        if failed_seams:
            # The human-readable half, through the channel the panel already
            # renders. The structured slot above is for machines (and for the
            # eval harness, which until TASK-18255 could not tell an unwired
            # seam from an empty one); this sentence is what the user reads.
            diagnostics.setdefault(LIBRARY_RAG_ROUTE_NOTES_KEY, []).append(
                ROUTE_NOTE_SEAMS_FAILED_TEMPLATE.format(
                    seams=", ".join(failed_seams)
                )
            )

        if not any(state is SeamState.AVAILABLE for state, _rows in outcomes.values()):
            # NOT the same condition, and the difference is the whole task.
            # Nothing CONFIGURED is "blocked" -- a setup problem the user can
            # act on. Everything configured having THROWN is a retrieval
            # FAILURE, and it must never present as a zero-row success:
            # `_outcome_from_service_result` maps empty rows to `status=
            # "empty"`, which is in `LIBRARY_RAG_ANSWERABLE_RETRIEVAL_STATUSES`,
            # so the RAG answer path would run and answer from NO context at
            # all. `failed` is deliberately reused rather than a new status --
            # `run_library_rag_search` already returns it for a raised search,
            # and it is already absent from every answerable allowlist.
            if failed_seams:
                return LibraryRagSearchOutcome(
                    status="failed",
                    recovery_state=_seams_failed_recovery_state(failed_seams),
                    diagnostics=diagnostics,
                )
            return LibraryRagSearchOutcome(
                status="blocked",
                recovery_state=_no_backend_recovery_state(),
                diagnostics=diagnostics,
            )

        # THE MERGE RULE (TASK-16071): rank-fair round-robin, NOT
        # concatenation in seam order.
        #
        # Each seam above ran its own query under its own `limit=top_k`, and
        # every row builder sets `"score": None` on purpose -- raw FTS5
        # scores from four different tables are not comparable, so a
        # cross-seam SCORE merge is unavailable here (the engine rejected one
        # for the same reason). What IS comparable is RANK POSITION, and this
        # is the engine's own primitive for exactly that situation
        # (`interleave_rankings`, used by its keyword leg since the fusion
        # cluster): position 0 of each seam, then position 1, and so on.
        #
        # Why it matters, measured. Concatenating in `_KNOWN_KEYWORD_SOURCE_
        # TYPES` order made a row's cross-seam position a function of its
        # SOURCE TYPE and of how many rows the earlier seams happened to
        # return -- never of how well it matched -- so any pass matching
        # `top_k` notes buried every media, conversation and prompt hit
        # behind them, and every downstream cut (the evidence list, RAG
        # Answer's budget, a harness's doc-level k) cut exactly there.
        # TASK-16071's worked examples: kw-quillon-mast's media target landed
        # at merged position 14 behind 13 notes, and conversation targets at
        # 19-21 -- displacement, not non-match (all were present at a deeper
        # k). Prompts, iterated last, were the most buried of the four.
        #
        # Order WITHIN one position is still `_KNOWN_KEYWORD_SOURCE_TYPES`
        # order: at equal rank there is genuinely no signal to choose on, so
        # the seam order breaks the tie. That is a pinned convention, not a
        # relevance claim (`Tests/Library/test_library_keyword_cross_seam.py`).
        #
        # NO TIERING -- and TASK-17755 is the change that makes that a live
        # decision rather than a vacuous one, so read this before touching
        # it. TASK-15700's tiered merge exists because the ENGINE's keyword
        # leg mixes primary matches with widened FALLBACK rows, and a
        # fallback row must fill positions rather than displace a primary
        # one. Until TASK-17755 this path had no such split (every seam ran
        # `build_fts_match_query` and nothing else) and the note here said
        # the 15700 tier design would apply if it ever gained fallback
        # forms. It now has them: a sub-leg whose AND primary returns zero
        # rows re-runs a prefix form (`_rows_with_prefix_fallback`).
        #
        # This merge is nonetheless left UNTIERED, deliberately and
        # narrowly: `and_then_prefix` was MEASURED untiered on this path
        # (TASK-3997's report -- 0.396 -> 0.423 MRR, the numbers the owner
        # decided on), and tiering it would ship an ordering nothing has
        # measured. What protects the primary rows meanwhile is weaker than
        # the engine's tiering but not nothing: the widened rows only ever
        # come from sub-legs that contributed NOTHING before, so no primary
        # row loses a position it used to hold at the front of the merge --
        # position 0 of each seam still comes first, in seam order. Deeper
        # in the list a fallback row can interleave ahead of a primary one,
        # which is exactly what tiering would fix.
        #
        # MEASURED AND CLOSED (TASK-17955, 2026-08-18): tiering is
        # UNOBSERVABLE on the gated corpus. Note the reason carefully, because
        # the obvious one is wrong: it is NOT that nothing gets cut. MRR and
        # NDCG consume ORDER, so a reordering changes a score with nothing
        # cut at all (Qodo PR-1801 caught that argument). The actual reason is
        # narrower -- a reordering can only move a score for a query with >=2
        # rows AND a RELEVANT row among them, and this corpus has none: 59 of
        # 60 plain queries return 0 or 1 row, and the single 6-row query
        # (`ng-mains-supply`) retrieves no relevant document at all, so every
        # permutation of it scores identically.
        #
        # Untiered is therefore the MEASURED choice, not an unpaid debt, and
        # TASK-16071's "the 15700 tier design would apply" note is retired.
        # RE-CHECK when the corpus or construction changes -- runnable, not
        # prose: `Docs/superpowers/qa/2026-08-18-merge-tiering/
        # tier_observability_census.py`. If ANY plain query starts returning
        # >=2 rows INCLUDING a relevant one, ordering becomes observable and
        # this decision is due for review. Tiering before then would ship an
        # ordering nothing can distinguish; `Tests/Library/test_library_keyword_and_then_prefix.py`
        # pins the untiered order so the change cannot happen silently.
        #
        # Dedup across seams is structurally vacuous -- the seams are
        # disjoint by source type -- but the primitive needs a key, and
        # `(provenance.source_type, source_id)` is the row identity every
        # builder already stamps (`_note_row` and friends). Note that
        # `interleave_rankings`' `seen` set also collapses duplicates WITHIN
        # one seam (it is one set across the whole merge, not one per
        # ranking); no seam can emit the same document twice today, so that
        # arm is unreachable, but it is the reason the key must be the full
        # document identity rather than the source id alone -- two seams'
        # ids are independent numbering spaces and a bare id would let a
        # note and a media row collide.
        #
        # NO TRUNCATION is added: a four-seam query has always been able to
        # return up to `4 * top_k` rows and the cut belongs to the consumers.
        # This change is ORDER only.
        rows: list[dict[str, Any]] = interleave_rankings(
            [
                outcomes[source_type][1]
                for source_type in _KNOWN_KEYWORD_SOURCE_TYPES
                if source_type in outcomes
            ],
            key=_keyword_row_identity,
        )

        if is_scoped and not rows:
            item_count = _scope_item_count(scope, source_types)
            return LibraryRagSearchOutcome(
                status="empty",
                recovery_state=_scope_zero_results_recovery_state(item_count),
                runtime_backend=_SEARCH_RUNTIME_BACKEND,
                diagnostics=diagnostics,
            )
        return {
            "results": rows,
            "runtime_backend": _SEARCH_RUNTIME_BACKEND,
            "diagnostics": diagnostics,
        }

    async def _search_notes(
        self,
        query: str,
        top_k: int,
        user_id: str,
        *,
        id_allowlist: Optional[Sequence[str]] = None,
    ) -> tuple[SeamState, list[dict[str, Any]]]:
        """Search the notes seam. Returns (state, rows)."""
        service = getattr(self._app, "notes_scope_service", None)
        if service is None:
            return SeamState.UNAVAILABLE, []
        # Forward the allowlist only when provided so an unscoped call
        # keeps the exact legacy call shape (byte-identical, spec D2).
        allowlist_kwargs = (
            {"id_allowlist": id_allowlist} if id_allowlist is not None else {}
        )

        async def run_match(fts_match_query: str) -> list[dict[str, Any]]:
            # Pre-built MATCH string (plural/singular widened) so the notes
            # seam is not limited to its exact-phrase fallback -- FTS5
            # unicode61 has no stemming (task-185 UAT). Which expression
            # arrives here -- the AND primary or the prefix widening -- is
            # `_rows_with_prefix_fallback`'s decision, not this seam's.
            raw_results = await service.search_notes(
                scope="local_note",
                query=query,
                limit=top_k,
                user_id=user_id,
                fts_match_query=fts_match_query,
                **allowlist_kwargs,
            )
            return [
                _note_row(item)
                for item in raw_results or ()
                if isinstance(item, Mapping)
            ]

        try:
            return SeamState.AVAILABLE, await _rows_with_prefix_fallback(query, run_match)
        except Exception:
            logger.opt(exception=True).warning(
                "Library keyword search: notes seam failed."
            )
            return SeamState.FAILED, []

    async def _search_media(
        self,
        query: str,
        top_k: int,
        *,
        id_allowlist: Optional[Sequence[str]] = None,
    ) -> tuple[SeamState, list[dict[str, Any]]]:
        """Search the media seam. Returns (state, rows)."""
        service = getattr(self._app, "media_reading_scope_service", None)
        if service is None:
            return SeamState.UNAVAILABLE, []
        # Forward the allowlist only when provided so an unscoped call
        # keeps the exact legacy call shape (byte-identical, spec D2).
        allowlist_kwargs = (
            {"id_allowlist": id_allowlist} if id_allowlist is not None else {}
        )

        async def run_match(fts_match_query: str) -> list[dict[str, Any]]:
            payload = await service.search_media(
                mode="local",
                query=query,
                limit=top_k,
                offset=0,
                fts_match_query=fts_match_query,
                **allowlist_kwargs,
            )
            items = payload.get("items", []) if isinstance(payload, Mapping) else []
            return [_media_row(item) for item in items if isinstance(item, Mapping)]

        try:
            return SeamState.AVAILABLE, await _rows_with_prefix_fallback(query, run_match)
        except Exception:
            logger.opt(exception=True).warning(
                "Library keyword search: media seam failed."
            )
            return SeamState.FAILED, []

    async def _search_conversations(
        self,
        query: str,
        top_k: int,
    ) -> tuple[SeamState, list[dict[str, Any]]]:
        """Search the conversations seam. Returns (state, rows)."""
        db = getattr(self._app, "chachanotes_db", None)
        if db is None:
            return SeamState.UNAVAILABLE, []

        async def run_match(fts_query: str) -> list[dict[str, Any]]:
            # Pre-built MATCH string, same as the notes/media/prompts seams.
            # TASK-19558: this used to arrive through the plain-text
            # `search_query` parameter and work only because that argument
            # was bound to MATCH raw. It now goes through the explicit
            # `fts_match_query` seam the siblings already used, so the
            # plain-text parameter can quote what it is given.
            if getattr(db, "is_memory_db", False):
                # In-memory SQLite connections are thread-local and only the
                # thread that created the database has the migrated schema;
                # offloading to a worker thread would hit a blank connection.
                raw_results = db.search_conversations_by_content(
                    query, top_k, fts_match_query=fts_query
                )
            else:
                raw_results = await asyncio.to_thread(
                    partial(
                        db.search_conversations_by_content,
                        query,
                        top_k,
                        fts_match_query=fts_query,
                    )
                )
            return [
                _conversation_row(item)
                for item in raw_results or ()
                if isinstance(item, Mapping)
            ]

        try:
            return SeamState.AVAILABLE, await _rows_with_prefix_fallback(query, run_match)
        except Exception:
            logger.opt(exception=True).warning(
                "Library keyword search: conversations seam failed."
            )
            return SeamState.FAILED, []

    async def _search_prompts(
        self,
        query: str,
        top_k: int,
    ) -> tuple[SeamState, list[dict[str, Any]]]:
        """Search the prompts seam. Returns (state, rows)."""
        service = getattr(self._app, "prompt_scope_service", None)
        if service is None:
            return SeamState.UNAVAILABLE, []

        async def run_match(fts_match_query: str) -> list[dict[str, Any]]:
            # Pre-built MATCH string, same as the notes/media/conversations
            # seams above -- and, since TASK-17755, under the same
            # ``and_then_prefix`` rule as all three.
            raw_results = await service.search_prompts(
                mode="local",
                query=query,
                limit=top_k,
                fts_match_query=fts_match_query,
            )
            return [
                _prompt_row(item)
                for item in raw_results or ()
                if isinstance(item, Mapping)
            ]

        try:
            return SeamState.AVAILABLE, await _rows_with_prefix_fallback(query, run_match)
        except Exception:
            logger.opt(exception=True).warning(
                "Library keyword search: prompts seam failed."
            )
            return SeamState.FAILED, []

    async def _search_semantic(
        self,
        query: str,
        source_types: tuple[str, ...],
        top_k: int,
        kwargs: Mapping[str, Any],
        *,
        scope: Optional[EffectiveScope] = None,
        rag_service: Any = None,
        route_notes: Sequence[str] = (),
    ) -> Any:
        """Query the RAG runtime, initializing it lazily on first use (task-249).

        The RAG runtime's index is not itself scoped by source type, so
        results are post-filtered here: each row's provenance
        ``source_type`` is canonicalized via ``_SEMANTIC_SOURCE_TYPE_MAP``
        (mirroring ``library_rag_state``'s ``_OPEN_SOURCE_TYPE_MAP``) and
        dropped when it resolves to a *known* type that is not in
        `source_types` (e.g. `media` toggled off drops `media`/`media_chunk`
        rows). Rows whose provenance source type is missing or unrecognized
        are always kept -- there is no way to attribute them to a scope
        toggle, and silently hiding un-attributable evidence would be worse
        than occasionally over-including it. An empty `source_types`
        disables filtering entirely as a defensive guard; in practice the
        Search canvas's run gate never lets a query reach this method with
        no source selected.

        Zero raw results over a verifiably empty vector store return a
        distinct "Index empty" outcome instead of the bare zero-results
        state (AC #4): "no evidence for this query" and "nothing has been
        indexed yet" demand different user actions.

        Args:
            scope: Optional resolved RAG retrieval scope (rag-scope
                narrowing, task-6). `None` or an unscoped scope performs
                today's single unrestricted store query (no
                `metadata_allowlist` at all). A scoped value runs one store
                query per source_type present in the scope's allowlist --
                a flat `metadata_allowlist` cannot express an OR across
                source types, see `rag_scope.build_semantic_allowlists` --
                and merges the per-type results by score, descending,
                before trimming to `top_k` (mirrors
                `pipeline_functions_simple.search_semantic`'s merge).
            rag_service: Already-resolved runtime, passed by `_search_rag`
                so profile resolution and the search share one instance.
                `None` resolves it here, keeping this method self-contained.
            route_notes: Routing disclosures to attach to whatever this
                returns (e.g. "a hybrid profile ran semantic because a
                scope is active"). Empty for a search that ran exactly as
                the profile configured it.
        """
        if rag_service is None:
            rag_service = await self._resolve_rag_runtime()
        if rag_service is None:
            return LibraryRagSearchOutcome(
                status="blocked",
                recovery_state=_rag_mode_unavailable_recovery_state(),
            )
        include_citations = bool(kwargs.get("include_citations", True))

        allowlists = build_semantic_allowlists(scope) if scope is not None else None
        if allowlists is None:
            raw_results = await rag_service.search(
                query=query,
                top_k=top_k,
                search_type="semantic",
                include_citations=include_citations,
            )
        else:
            per_type_results: list[Any] = []
            for allowlist in allowlists:
                per_type_results.extend(
                    await rag_service.search(
                        query=query,
                        top_k=top_k,
                        search_type="semantic",
                        include_citations=include_citations,
                        metadata_allowlist=allowlist,
                    )
                )
            per_type_results.sort(key=_raw_semantic_score, reverse=True)
            raw_results = per_type_results[:top_k]

        rows = _filtered_semantic_rows(raw_results, source_types)
        if not raw_results and await self._semantic_index_is_empty(rag_service):
            return _with_route_notes(
                LibraryRagSearchOutcome(
                    status="empty",
                    recovery_state=_rag_index_empty_recovery_state(),
                    runtime_backend=_RAG_RUNTIME_BACKEND,
                ),
                route_notes,
            )
        if scope is not None and scope.state == "scoped" and not rows:
            item_count = _scope_item_count(scope, source_types)
            return _with_route_notes(
                LibraryRagSearchOutcome(
                    status="empty",
                    recovery_state=_scope_zero_results_recovery_state(item_count),
                    runtime_backend=_RAG_RUNTIME_BACKEND,
                ),
                route_notes,
            )
        return _with_route_notes(
            _retrieval_payload(rows, source_types, _RAG_RUNTIME_BACKEND), route_notes
        )

    async def _search_hybrid(
        self,
        query: str,
        source_types: tuple[str, ...],
        top_k: int,
        kwargs: Mapping[str, Any],
        *,
        scope: Optional[EffectiveScope] = None,
        rag_service: Any,
    ) -> Any:
        """Run the engine's RRF-fused hybrid search (FTS-servable, any scope).

        Only `_search_rag` calls this, and only once it has established the
        one condition the engine imposes: at least one selected source type
        the FTS leg can actually serve (`_FTS_SERVABLE_SOURCE_TYPES`),
        otherwise its rows would all be dropped by the source-type
        post-filter. That condition used to read "media selected", because
        the engine's FTS leg was media-only; TASK-3996 gave it notes and
        conversation sub-legs and TASK-15020/B2 gave it prompts, so every one
        of the Search canvas's four toggles now keeps the query on the hybrid
        path -- including a prompts-only selection, whose rows can ONLY come
        from this leg (prompts have no vector index, so they reach the fused
        output as FTS-only rows). There used to be a SECOND condition -- no scope --
        because allowlist pushdown was semantic-only; TASK-15020/B1 removed
        it at the engine (each FTS sub-leg takes its entry's ids as a
        parameterized filter, and a sub-leg the scope never names is skipped
        rather than run unfiltered), which is what this method's `scope`
        argument now carries.

        The selection is also pushed INTO the engine's keyword leg
        (TASK-14751). Before that, the leg split one fixed `top_k` FTS budget
        three ways regardless of what the user had selected, and the rows for
        unselected types were then discarded by the post-filter below -- a
        media-only search over an empty vector index showed roughly a third
        of the media rows it should. Pushing the (translated, singular)
        selection down spends the whole budget on types that survive; the
        post-filter stays, because it also drops semantic-leg rows the
        keyword selection has no say over.

        Zero-results honesty (spec Workstream A item 5): "Index empty" is a
        claim about the whole runtime, so it may only be made when the
        engine returned nothing at all. When the FTS leg DID return rows
        over an empty vector store, the user has evidence on screen and the
        honest statement is the narrower one -- the semantic leg is empty,
        these are keyword-only results.

        Args:
            query: Already-validated user query.
            source_types: Selected Library source type identifiers.
            top_k: Result cap (the fused cap; each leg is over-fetched by
                the engine).
            kwargs: Backend options (`include_citations`).
            scope: Optional resolved retrieval scope. A scoped value is
                translated by `build_semantic_allowlists` into the union of
                per-source-type AND-groups the engine expects (a flat dict
                cannot express "media in A OR note in B") and passed as ONE
                `metadata_allowlist` -- unlike `_search_semantic`, which
                predates the engine's multi-entry support and still issues
                one store query per entry itself. Both legs honor it.
            rag_service: The resolved runtime.

        Returns:
            A mapping with `results`/`runtime_backend` (`rag-hybrid`) plus
            diagnostics, or the "Index empty" / scoped-zero-results recovery
            outcome.
        """
        # `_search_rag` has already established this is non-empty, so the
        # keyword leg is never asked to serve nothing.
        keyword_source_types = {
            _ENGINE_KEYWORD_SOURCE_TYPES[source_type]
            for source_type in source_types
            if source_type in _ENGINE_KEYWORD_SOURCE_TYPES
        }
        allowlists = build_semantic_allowlists(scope) if scope is not None else None
        # Forwarded only when there IS a scope, so an unscoped hybrid search
        # calls the engine with exactly the arguments it did before B1 (and
        # every pre-existing runtime double keeps working unchanged).
        allowlist_kwargs = (
            {"metadata_allowlist": allowlists} if allowlists is not None else {}
        )
        raw_results = await rag_service.search(
            query=query,
            top_k=top_k,
            search_type="hybrid",
            include_citations=bool(kwargs.get("include_citations", True)),
            keyword_source_types=keyword_source_types,
            **allowlist_kwargs,
        )
        rows = _filtered_semantic_rows(raw_results, source_types)
        if not raw_results and await self._semantic_index_is_empty(rag_service):
            return LibraryRagSearchOutcome(
                status="empty",
                recovery_state=_rag_index_empty_recovery_state(),
                runtime_backend=_RAG_HYBRID_RUNTIME_BACKEND,
            )
        if scope is not None and scope.state == "scoped" and not rows:
            # The same distinction the scoped semantic path draws (task-6):
            # "nothing matched among the N items you scoped to" is a
            # scope-shaped dead end with a scope-shaped next action, and
            # routing scoped queries to hybrid must not quietly demote it to
            # the generic no-match state.
            return LibraryRagSearchOutcome(
                status="empty",
                recovery_state=_scope_zero_results_recovery_state(
                    _scope_item_count(scope, source_types)
                ),
                runtime_backend=_RAG_HYBRID_RUNTIME_BACKEND,
            )
        route_notes: list[str] = []
        if _rows_are_keyword_only(rows) and await self._semantic_index_is_empty(
            rag_service
        ):
            # Both halves matter: rows with no vector leg alone would also
            # describe a populated index whose vectors simply lost the
            # ranking, and "semantic leg empty" would then be a false claim.
            route_notes.append(ROUTE_NOTE_SEMANTIC_LEG_EMPTY)
        return _with_route_notes(
            _retrieval_payload(rows, source_types, _RAG_HYBRID_RUNTIME_BACKEND),
            route_notes,
        )

    async def _resolve_rag_runtime(self) -> Any:
        """Return a usable RAG runtime, lazily creating the shared one.

        Resolution order:

        1. An existing ``app._rag_service`` with a callable ``search`` wins
           (already initialized by any surface, or injected by tests) --
           UNLESS a profile switch superseded it since it was cached
           (``current_app_rag_service``, the one staleness rule shared with
           ``semantic_availability``'s resolver). Without that check, a
           Settings profile change would leave this path retrieving under
           the OLD profile for the rest of the session, and `_search_rag`
           would attribute its disclosure to a profile that is no longer
           active -- a false claim about the very thing being disclosed.
        2. The ``embeddings_rag`` deps gate (cheap ``find_spec`` probe, no
           imports) short-circuits BEFORE any heavy work, so missing-deps
           installs keep the existing recovery routing at zero cost (AC #3).
        3. ``get_shared_rag_service()`` constructs the process-wide runtime.
           First-time construction loads an embedding model (can take
           seconds), so it runs in ``asyncio.to_thread`` -- never on the UI
           event loop. The factory is double-checked-locked, so concurrent
           Library queries racing here serialize inside it and share one
           instance. The factory already converts construction failures to
           None; the guard here additionally maps anything it might still
           raise to None, so a failed first initialization always renders
           the RAG-unavailable recovery state (setup routing) rather than
           ``run_library_rag_search``'s generic "Retrieval failed / Retry"
           outcome -- retrying cannot fix a runtime that will not build.

        Returns:
            The RAG runtime, or None when it is unavailable (missing deps or
            failed construction) -- the caller renders the recovery state.
        """
        cached = current_app_rag_service(self._app)
        if cached is not None:
            return cached
        if not embeddings_rag_deps_installed():
            return None
        # Captured BEFORE the build -- see `cache_app_rag_service`.
        generation = shared_rag_service_generation()
        try:
            service = await asyncio.to_thread(get_shared_rag_service)
        except Exception:
            logger.opt(exception=True).error(
                "Library RAG: shared RAG service initialization raised; "
                "treating the runtime as unavailable."
            )
            return None
        if service is None or not callable(getattr(service, "search", None)):
            return None
        # Cache on the app so every RAG surface (chat sidebar readiness,
        # repeat Library queries) sees the initialized runtime, stamped so a
        # later profile switch invalidates it.
        cache_app_rag_service(self._app, service, generation)
        return service

    async def _semantic_index_is_empty(self, rag_service: Any) -> bool:
        """True only when the runtime's vector store verifiably has 0 documents.

        Anything short of a trustworthy zero -- no ``vector_store``, stats
        call failing, an ``error`` payload, a non-integer count -- returns
        False so the caller falls back to the generic zero-results outcome
        rather than claiming an empty index it cannot verify.
        """
        get_stats = getattr(
            getattr(rag_service, "vector_store", None), "get_collection_stats", None
        )
        if not callable(get_stats):
            return False
        try:
            # ChromaDB-backed stats can touch disk; keep it off the event loop.
            stats = await asyncio.to_thread(get_stats)
        except Exception:
            logger.opt(exception=True).debug(
                "Library RAG: vector store stats probe failed."
            )
            return False
        if not isinstance(stats, Mapping) or stats.get("error"):
            return False
        try:
            return int(stats.get("count")) == 0
        except (TypeError, ValueError):
            return False


def _resolve_profile_search_mode(rag_service: Any) -> str:
    """Map the active profile's default_search_mode to an execution route.

    "plain" deliberately routes to the four-seam scope-aware keyword path,
    NOT the engine's keyword leg (spec: plain-profile routing) -- the
    engine's ``search_type="keyword"`` still refuses a
    ``metadata_allowlist`` (it has no semantic leg to scope), so a scoped
    plain search needs the four-seam path regardless of which source types
    the engine's leg now spans; the four-seam path remains the product
    surface for plain mode.
    Unknown values -- and any runtime without a profile config at all, which
    includes every pre-profile test fake -- fall back to "semantic", the
    behavior this path had before profiles were honored.

    Args:
        rag_service: The resolved RAG runtime.

    Returns:
        ``plain``, ``semantic``, or ``hybrid``.
    """
    mode = getattr(
        getattr(getattr(rag_service, "config", None), "search", None),
        "default_search_mode",
        "semantic",
    )
    return normalize_rag_search_mode(mode)


def _profile_disclosure_label(rag_service: Any) -> str:
    """Name the active profile for a routing disclosure.

    `EnhancedRAGServiceV2` carries the selected `ProfileConfig` on
    `.profile`; a bare `RAGService` (or a custom config) has none, in which
    case the disclosure still has to be makeable -- it just cannot name a
    profile.

    Args:
        rag_service: The resolved RAG runtime.

    Returns:
        `"Profile '<name>'"`, or `"Active RAG profile"` when the runtime
        exposes no usable profile name.
    """
    name = getattr(getattr(rag_service, "profile", None), "name", None)
    if isinstance(name, str) and name.strip():
        return f"Profile '{name.strip()}'"
    return "Active RAG profile"


def _with_route_notes(result: Any, notes: Sequence[str]) -> Any:
    """Attach routing disclosures to a retrieval result, without clobbering.

    Works on both shapes this service returns (a raw mapping and a
    `LibraryRagSearchOutcome`) so a disclosure survives whichever path the
    query took, including the empty/blocked outcomes. Appends to any notes
    already present rather than replacing them.

    Args:
        result: A retrieval payload mapping or `LibraryRagSearchOutcome`.
        notes: Disclosure fragments to attach. Empty leaves `result`
            untouched and identical -- callers on the no-divergence path
            keep their pre-existing byte-for-byte payload (no empty
            `diagnostics` key materializes).

    Returns:
        `result` when `notes` is empty, else a copy carrying the notes under
        `diagnostics[LIBRARY_RAG_ROUTE_NOTES_KEY]`.
    """
    if not notes:
        return result
    if isinstance(result, LibraryRagSearchOutcome):
        diagnostics = dict(result.diagnostics or {})
        diagnostics[LIBRARY_RAG_ROUTE_NOTES_KEY] = [
            *(diagnostics.get(LIBRARY_RAG_ROUTE_NOTES_KEY) or ()),
            *notes,
        ]
        return replace(result, diagnostics=diagnostics)
    if isinstance(result, Mapping):
        existing = result.get("diagnostics")
        diagnostics = dict(existing) if isinstance(existing, Mapping) else {}
        diagnostics[LIBRARY_RAG_ROUTE_NOTES_KEY] = [
            *(diagnostics.get(LIBRARY_RAG_ROUTE_NOTES_KEY) or ()),
            *notes,
        ]
        return {**result, "diagnostics": diagnostics}
    return result


def _filtered_semantic_rows(
    raw_results: Any, source_types: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Normalize engine results and apply the source-type post-filter.

    Shared by the semantic and hybrid arms: hybrid's FTS-leg rows carry a
    provenance `source_type` stamped upstream (`media`, `note` or
    `conversation` -- the singular ingestion vocabulary, canonicalized here
    by `_semantic_row_matches_scope`) precisely so they survive this same
    filter instead of vanishing. A row whose type is outside the selection
    is dropped here, which is why the routing gate only runs hybrid when the
    selection contains something the FTS leg can serve.
    """
    rows = [_semantic_row(item) for item in raw_results or ()]
    if not source_types:
        return rows
    return [row for row in rows if _semantic_row_matches_scope(row, source_types)]


def _row_is_keyword_only(row: Mapping[str, Any]) -> bool:
    """True when this ONE row reached the results with no vector-leg score.

    Reads the fusion block's `vector_score` (preserved by
    `_fuse_hybrid_results`): `None` means the chunk was never returned by
    the vector leg. A row with no fusion block at all cannot be judged, so
    it is NOT called keyword-only. That default is load-bearing twice over:
    a semantic-path row never carries a fusion block, which is how the whole
    pre-hybrid world keeps its coverage answer byte-identical, and inventing
    a "keyword matches only" claim out of absent provenance would be the
    same defect TASK-14752 exists to remove, pointed the other way.
    """
    provenance = row.get("provenance")
    fusion = (
        provenance.get("hybrid_fusion") if isinstance(provenance, Mapping) else None
    )
    return isinstance(fusion, Mapping) and fusion.get("vector_score") is None


def _rows_are_keyword_only(rows: Sequence[Mapping[str, Any]]) -> bool:
    """True when every row came from hybrid's FTS leg with no vector leg.

    The per-row judgement (and its "un-judgeable rows are not keyword-only"
    default) lives in `_row_is_keyword_only`, so this whole-set claim and
    the per-source-type one `_semantic_scope_coverage` makes cannot drift
    apart into two different readings of the same fusion block.
    """
    if not rows:
        return False
    return all(_row_is_keyword_only(row) for row in rows)


def _retrieval_payload(
    rows: list[dict[str, Any]],
    source_types: tuple[str, ...],
    runtime_backend: str,
) -> dict[str, Any]:
    """Build the `rag`-mode result mapping plus its coverage diagnostics.

    Task 8: report which requested source types the semantic leg actually
    touched. Deliberately omitted (not an empty dict) when `rows` is empty
    -- the zero-rows path is the empty/no-match state (Task 11's
    territory), not a coverage claim, and omitting the key keeps the
    pre-existing bare `{"results": [], "runtime_backend": ...}` contract for
    that path byte-identical (see
    `test_rag_mode_zero_results_with_populated_index_stays_generic`).

    The hybrid arm shares this, and TASK-3996 is the change the old note
    here anticipated ("revisit when the keyword leg goes four-seam"): the
    engine's FTS leg now serves notes and conversations too, so a type CAN
    be covered by the FTS leg alone. TASK-14752 is the follow-up this docstring
    used to defer -- `_semantic_scope_coverage` now separates that case out
    as `keyword_only`, and `library_rag_state.library_rag_coverage_note`
    gives it its own sentence, so "Semantic search found nothing from X" is
    never rendered over a type whose rows are on screen.
    """
    result: dict[str, Any] = {"results": rows, "runtime_backend": runtime_backend}
    if rows and source_types:
        result["diagnostics"] = {
            "semantic_scope_coverage": _semantic_scope_coverage(source_types, rows)
        }
    return result


async def _rows_with_prefix_fallback(
    query: str,
    run_match: Callable[[str], Awaitable[list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    """Run ONE sub-leg's keyword query under the ``and_then_prefix`` rule.

    TASK-17755, and the whole of it: the AND-of-variant-groups
    (`build_fts_match_query`) stays the primary, and the PREFIX form
    (`build_prefix_match_query`) runs **only** for a sub-leg whose primary
    returned zero rows.

    Written once and used by all four seams rather than inlined four times,
    for the reason the four seams have historically drifted: each one calls
    a different service with a different signature, and a construction
    copied four ways is a construction that will eventually be three
    constructions. The seams differ in HOW to run a MATCH; that is what
    `run_match` carries. They do not differ in WHEN to widen.

    The decision is per SUB-LEG, exactly as the engine makes it
    (`RAGService._fts_rows_with_fallback`, shipped since TASK-15700), never
    per query. One search can legitimately carry AND rows from the notes
    seam and prefix rows from the media seam -- that mix is the point.
    Deciding per query would either leave every seam strict (no rescue at
    all, since any one seam matching would suppress the others' fallback) or
    widen seams that had already found the right document.

    Why this cannot regress a hit: a sub-leg that found rows never builds
    the prefix expression, let alone runs it. That is a property of the
    control flow, not of the two forms' relative recall, which is why it
    survives whatever either builder is changed to later. It is pinned by
    counting prefix CONSTRUCTIONS rather than comparing rows, in
    `Tests/Library/test_library_keyword_and_then_prefix.py`.

    An empty expression means "no rows" on both sides and the query is
    skipped: an empty MATCH is an FTS5 syntax error rather than an empty
    answer, and for the prefix form specifically the only alternative to
    skipping would be an unbounded stopword prefix over the whole corpus.

    Args:
        query: The raw user query. Both forms are built from it; the caller
            never builds either itself.
        run_match: Runs ONE MATCH expression against this seam's service and
            returns the seam's finished row list (already projected by the
            seam's row builder, so the zero-row test is made on what would
            actually reach the merge -- not on a raw payload whose items the
            projection might have dropped).

    Returns:
        The primary's rows, or -- only when those were empty -- the prefix
        form's rows. Never both: a sub-leg's rows are all-primary or
        all-fallback within one query.
    """
    primary = build_fts_match_query(query)
    rows = await run_match(primary) if primary else []
    if rows:
        return rows

    # Built here and nowhere earlier: constructing it eagerly would cost
    # nothing measurable but would destroy the ability to PROVE the fallback
    # never fires for a hitting sub-leg, which is the property the owner's
    # decision rests on.
    fallback = build_prefix_match_query(query)
    if not fallback:
        return []
    return await run_match(fallback)


def _keyword_row_identity(row: Mapping[str, Any]) -> Hashable:
    """The dedup identity `interleave_rankings` merges the four seams on.

    `(source_type, source_id)` is the identity every row builder stamps, and
    the pair -- not the bare id -- is required because the seams number
    independently: a note 7 and a media 7 are different documents.

    THE DEGENERATE ARM (final review, TASK-16071): every builder falls back
    to `""` when its id is missing (`str(item.get("id", ""))` and siblings;
    the prompts normalizer yields `local_id=None` for any non-local
    backend). `interleave_rankings` dedups on ONE `seen` set spanning the
    whole merge, so two id-less rows of the same source type would collide
    and the second would be SILENTLY DROPPED -- a truncation at the one site
    whose contract is that it truncates nothing. An id-less row is therefore
    keyed by its own object identity: it can never be deduped against
    anything, which is the honest reading of "we do not know what document
    this is". Pinned by `test_d2_rows_with_an_empty_source_id_are_not_
    collapsed`.
    """
    source_id = row.get("source_id")
    if not source_id:
        return ("", "", id(row))
    return ((row.get("provenance") or {}).get("source_type"), source_id)


def _note_row(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_id": str(item.get("id", "")),
        "chunk_id": "",
        "title": item.get("title") or "",
        "snippet": item.get("content") or "",
        "score": None,
        "provenance": {"source_type": "note"},
    }


def _media_row(item: Mapping[str, Any]) -> dict[str, Any]:
    media_type = item.get("media_type") or "media"
    return {
        "source_id": str(item.get("source_id") or item.get("id") or ""),
        "chunk_id": "",
        "title": item.get("title") or "",
        "snippet": f"Matched media · {media_type}",
        "score": None,
        "provenance": {"source_type": "media"},
    }


def _conversation_row(item: Mapping[str, Any]) -> dict[str, Any]:
    try:
        message_count = int(item.get("message_count") or 0)
    except (TypeError, ValueError):
        message_count = 0
    return {
        "source_id": str(item.get("id", "")),
        "chunk_id": "",
        "title": item.get("title") or "",
        "snippet": (
            f"Matched conversation · {message_count} "
            f"{'message' if message_count == 1 else 'messages'}"
        ),
        # C1: keyword-mode rows show no score, uniformly with notes/media --
        # `relevance_score`/`best_rank` are an FTS ranking artifact, not a
        # retrieval similarity score, so surfacing it here was misleading.
        # RAG-mode rows (see `_semantic_row`) keep their real scores.
        "score": None,
        "provenance": {"source_type": "conversation"},
    }


def _prompt_row(item: Mapping[str, Any]) -> dict[str, Any]:
    # Trap (Task 4 review): `PromptScopeService.search_prompts` normalizes
    # each result via `normalize_prompt_record`, whose "id" is a composite
    # "local:prompt:<n>" string -- the raw integer prompt id lives under
    # "local_id". Using "id" here would break
    # `_open_library_item_by_id("prompt", ...)`/`handle_library_prompt_row`,
    # which both expect the raw int.
    local_id = item.get("local_id")
    return {
        "source_id": str(local_id) if local_id is not None else "",
        "chunk_id": "",
        "title": item.get("name") or "",
        "snippet": item.get("user_prompt") or item.get("details") or "",
        "score": None,
        "provenance": {"source_type": "prompt"},
    }


def _semantic_row(item: Any) -> dict[str, Any]:
    values = (
        item
        if isinstance(item, Mapping)
        else {
            "id": getattr(item, "id", None),
            "score": getattr(item, "score", None),
            "document": getattr(item, "document", None),
            "metadata": getattr(item, "metadata", None),
            "citations": getattr(item, "citations", None),
        }
    )
    metadata_value = values.get("metadata")
    metadata = dict(metadata_value) if isinstance(metadata_value, Mapping) else {}
    provenance = dict(metadata)
    source_type = (
        provenance.pop("source_type", None)
        or provenance.pop("item_type", None)
        or provenance.pop("type", None)
    )
    if source_type:
        provenance["source_type"] = source_type
    row: dict[str, Any] = {
        "source_id": str(
            metadata.get("source_id")
            or metadata.get("document_id")
            or values.get("id")
            or ""
        ),
        "chunk_id": str(metadata.get("chunk_id") or ""),
        "title": metadata.get("title") or metadata.get("document_title") or "",
        "snippet": values.get("document") or "",
        "score": _coerce_score(values.get("score")),
        "provenance": provenance,
    }
    citations = values.get("citations")
    if citations:
        row["citations"] = [_semantic_citation(citation) for citation in citations]
    return row


def _semantic_row_matches_scope(row: Mapping[str, Any], scope: tuple[str, ...]) -> bool:
    """True when `row` survives rag-mode scope post-filtering.

    Args:
        row: A normalized `_semantic_row` output.
        scope: Selected Library source type identifiers (never empty --
            callers guard that case before calling this).

    Returns:
        `False` only when the row's provenance `source_type` canonicalizes
        to a *known* type that is not in `scope`. Rows with missing or
        unrecognized provenance always return `True` (see `_search_semantic`).
    """
    provenance = row.get("provenance")
    raw_source_type = (
        provenance.get("source_type") if isinstance(provenance, Mapping) else None
    )
    canonical = _SEMANTIC_SOURCE_TYPE_MAP.get(
        str(raw_source_type or "").strip().lower()
    )
    if canonical is None:
        return True
    return canonical in scope


def _semantic_scope_coverage(
    source_types: tuple[str, ...], rows: Sequence[Mapping[str, Any]]
) -> dict[str, list[str]]:
    """Which requested source types the semantic leg actually touched (Task 8).

    The semantic leg is one merged store query trimmed to `top_k` (or one
    per-type query merged and trimmed, when scoped) -- unlike keyword mode,
    which fans out one query per selected source (always "per source"). A
    requested type can therefore come back with zero rows even though other
    requested types matched well, and there is nothing on screen today that
    tells a user "semantic search never looked at your notes" versus "your
    notes have nothing relevant" (live UAT, RAG-29/Task 8).

    Under a HYBRID profile a third state exists, and TASK-14752 is it: a
    type can be on screen entirely from the engine's FTS leg. Before
    TASK-3996 that leg served media only, so "no semantic hits for this
    type" and "no evidence at all for this type" were the same fact and one
    `uncovered` list said everything; with notes and conversation sub-legs,
    folding keyword-sourced types into `covered` hides that the semantic leg
    never matched them, while folding them into `uncovered` would tell a
    user that a type produced nothing while its rows are in front of them.

    Args:
        source_types: The caller's requested Library source type
            identifiers (e.g. `notes`, `media`) -- never empty; the caller
            guards that case before calling this.
        rows: The final, already scope-post-filtered `_semantic_row` rows
            (i.e. what will actually be shown as evidence).

    Returns:
        `{"covered": [...], "uncovered": [...]}` plus `"keyword_only": [...]`
        when (and only when) that third list is non-empty -- so the semantic
        and plain profiles, which cannot produce it, keep a byte-identical
        payload. All lists are in `source_types` order, they PARTITION the
        requested types the semantic leg can structurally speak to
        (`_SEMANTICALLY_COVERABLE_SOURCE_TYPES`, task-15 finding I2), and a
        requested type with no semantic-index seam at all (`prompts`) never
        appears in any of them, since it was never "searched" in any sense
        this note can honestly claim. Membership:

        * `covered` -- at least one row of that type carried a vector-leg
          contribution (or came from a path that has no fusion provenance to
          judge, i.e. every non-hybrid search).
        * `keyword_only` -- rows of that type are present, but every one of
          them is FTS-only (`_row_is_keyword_only`).
        * `uncovered` -- no rows of that type at all.

        A row whose provenance is missing or unrecognized (edge case: it
        survives the scope post-filter because it cannot be attributed to
        any toggle) contributes to none of them -- it cannot prove any
        specific requested type was actually searched-and-found, so it must
        not mask a genuinely uncovered type.
    """
    semantic_present: set[str] = set()
    keyword_present: set[str] = set()
    for row in rows:
        provenance = row.get("provenance")
        raw_source_type = (
            provenance.get("source_type") if isinstance(provenance, Mapping) else None
        )
        canonical = _SEMANTIC_SOURCE_TYPE_MAP.get(
            str(raw_source_type or "").strip().lower()
        )
        if not canonical:
            continue
        if _row_is_keyword_only(row):
            keyword_present.add(canonical)
        else:
            semantic_present.add(canonical)
    coverable_source_types = [
        source_type
        for source_type in source_types
        if source_type in _SEMANTICALLY_COVERABLE_SOURCE_TYPES
    ]
    keyword_only = [
        source_type
        for source_type in coverable_source_types
        if source_type not in semantic_present and source_type in keyword_present
    ]
    coverage = {
        "covered": [
            source_type
            for source_type in coverable_source_types
            if source_type in semantic_present
        ],
        "uncovered": [
            source_type
            for source_type in coverable_source_types
            if source_type not in semantic_present
            and source_type not in keyword_present
        ],
    }
    if keyword_only:
        coverage["keyword_only"] = keyword_only
    return coverage


def _semantic_citation(citation: Any) -> dict[str, Any]:
    if isinstance(citation, Mapping):
        return dict(citation)
    return {
        "label": getattr(citation, "document_title", None)
        or getattr(citation, "text", None)
        or "Citation",
        "source_id": getattr(citation, "document_id", None) or "",
        "chunk_id": getattr(citation, "chunk_id", None) or "",
    }


def _coerce_score(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _raw_semantic_score(item: Any) -> float:
    """Sortable score for a raw (pre-``_semantic_row``) semantic result item.

    Mirrors ``_coerce_score``'s dual Mapping/attribute handling so the
    per-type merge (``_search_semantic``) can sort mixed-shape results from
    ``rag_service.search`` without first normalizing every item.
    """
    value = item.get("score") if isinstance(item, Mapping) else getattr(item, "score", None)
    return _coerce_score(value) or float("-inf")


async def _empty_scoped_seam() -> tuple[SeamState, list[dict[str, Any]]]:
    """Stand-in for a keyword seam whose source type is scoped to zero ids.

    The seam itself is available (the app has it configured) -- scope just
    leaves nothing for it to return -- so this reports ``(True, [])`` rather
    than ``(False, [])`` (unavailable), keeping ``_search_keyword``'s
    any-seam-available gate accurate when other seams still have results.
    """
    return SeamState.AVAILABLE, []


#: This service's own keyword-source-type vocabulary ("media", "notes", ...)
#: mapped to ``rag_scope``'s scope-vocabulary keys. "conversations" and
#: "prompts" are deliberately absent -- an active scope always excludes
#: those seams entirely rather than restricting them (spec D5), so they
#: never contribute to a "items searched" count.
_SCOPE_ELIGIBLE_KEYWORD_TYPES = {"media": SOURCE_TYPE_MEDIA, "notes": SOURCE_TYPE_NOTE}


def _scope_item_count(
    scope: EffectiveScope, searched_source_types: Sequence[str]
) -> int:
    """Total scope items across only the source types actually searched.

    Counting every allowlisted source type regardless of what was actually
    queried overstates "items searched" whenever a caller narrows
    ``source_types`` to a subset (e.g. media only) -- the previous
    implementation summed the WHOLE scoped allowlist unconditionally,
    including source types the caller never asked to search (task-9 review
    finding).

    Args:
        scope: The resolved (scoped) effective scope.
        searched_source_types: This service's own source-type identifiers
            (``"media"``, ``"notes"``, ...) that were actually queried
            under this scope. Types outside
            ``_SCOPE_ELIGIBLE_KEYWORD_TYPES`` (conversations, prompts) are
            ignored automatically -- callers may pass their full
            ``source_types`` tuple unfiltered.

    Returns:
        The sum of allowlisted item counts for only the eligible, searched
        source types.
    """
    total = 0
    for source_type in searched_source_types:
        scope_key = _SCOPE_ELIGIBLE_KEYWORD_TYPES.get(source_type)
        if scope_key is None:
            continue
        total += len(scope.allowlist.get(scope_key, ()))
    return total


def _seams_failed_recovery_state(
    failed_seams: Sequence[str],
) -> DestinationRecoveryState:
    """Recovery state for "every configured seam ran and threw".

    Distinct from `_no_backend_recovery_state` on purpose: nothing configured
    is a SETUP problem ("configure retrieval"), whereas everything configured
    failing is an OPERATIONAL one ("retry, check indexing"). Naming the seams
    matters because the useful next action differs per backend.

    Args:
        failed_seams: Seam names that ran and raised, already sorted.

    Returns:
        The recovery state for a total keyword-retrieval failure.
    """
    named = ", ".join(failed_seams) if failed_seams else "every"
    return DestinationRecoveryState(
        status_label="Retrieval failed",
        unavailable_what="Library Search/RAG retrieval",
        why=f"Every configured Library seam failed ({named})",
        next_action="Retry the query or check Library indexing",
        recovery_action="Retry",
        authority_owner="Library retrieval service",
        stable_selector=LIBRARY_RAG_SERVICE_ERROR_SELECTOR,
        disabled_tooltip=(
            f"Library retrieval failed in every configured seam ({named}). "
            "Retry the query or check Library indexing."
        ),
    )


def _no_backend_recovery_state() -> DestinationRecoveryState:
    return DestinationRecoveryState(
        status_label="Unavailable",
        unavailable_what="Library Search/RAG retrieval",
        why="No local Library source seam (notes, media, conversations, or prompts) is available",
        next_action="Configure Library RAG retrieval or use standalone Search/RAG",
        recovery_action="Search/RAG setup",
        authority_owner="Library retrieval service",
        stable_selector=LIBRARY_RAG_SERVICE_ERROR_SELECTOR,
        disabled_tooltip=(
            "No local Library source seam is available in this runtime. "
            "Configure retrieval or use standalone Search/RAG."
        ),
    )


def _rag_mode_unavailable_recovery_state() -> DestinationRecoveryState:
    return DestinationRecoveryState(
        status_label="RAG unavailable",
        unavailable_what="Library Search/RAG retrieval",
        why="The RAG runtime is not available in this app instance",
        # (Task-14 enabler) name the pip extra to install -- "unavailable"
        # alone leaves no next step. Voice mirrors
        # RAG_Search/semantic_availability.py's SEMANTIC_REASON_DEPS_MISSING
        # copy family (that module's own equivalent seam, deliberately
        # untouched -- see its module docstring). The durable fix (install)
        # is paired with the immediate escape ("switch mode to Search"),
        # matching both sibling RAG-blocked states in this file
        # (`_rag_index_empty_recovery_state`, `_no_backend_recovery_state`)
        # -- this is the always-rendered "Next:" line, unlike the
        # mode-toggle button's hover/focus-only tooltip, so a blocked user
        # needs the escape spelled out here too (review finding).
        next_action=(
            'Install RAG support: pip install "tldw_chatbook[embeddings_rag]", '
            "then restart, or switch mode to Search."
        ),
        recovery_action="Settings > RAG",
        authority_owner="Library retrieval service",
        stable_selector=LIBRARY_RAG_SERVICE_ERROR_SELECTOR,
        disabled_tooltip=(
            "RAG runtime is unavailable in this app instance. "
            'Install RAG support: pip install "tldw_chatbook[embeddings_rag]", '
            "then restart, or switch mode to Search."
        ),
    )


def _rag_index_empty_recovery_state() -> DestinationRecoveryState:
    """Recovery copy for a working RAG runtime over an empty semantic index.

    Distinct from the generic zero-results state (AC #4): the runtime is
    fine, there is simply nothing indexed yet, so the next action is to add
    content (ingestion indexes automatically, task-247) or backfill existing
    content -- not to rephrase the query.
    """
    return DestinationRecoveryState(
        status_label="Index empty",
        unavailable_what="Library RAG Answer evidence",
        why="The semantic index has no content yet",
        next_action=(
            "Import content to index it automatically, run a semantic index "
            "backfill, or switch mode to Search"
        ),
        recovery_action="Library import",
        authority_owner="Library retrieval service",
        stable_selector=LIBRARY_RAG_EMPTY_STATE_SELECTOR,
        disabled_tooltip=(
            "The semantic index has no content yet. "
            "Import content to index it automatically or run a semantic index backfill."
        ),
    )


def _scope_zero_results_recovery_state(item_count: int) -> DestinationRecoveryState:
    """Recovery copy for a scoped search that ran cleanly but matched nothing.

    Distinct from both the generic zero-results state and "Index empty"
    (AC #4-style distinction, task-6): the retrieval scope excluded
    everything it did not allowlist and the query still matched none of the
    ``item_count`` allowlisted items, so the next action is to broaden or
    clear the conversation's retrieval scope -- not to rephrase the query or
    treat the index itself as unpopulated.
    """
    message = f"No results within scope ({item_count} items searched)"
    return DestinationRecoveryState(
        status_label="No results",
        unavailable_what="Library Search/RAG evidence",
        why=message,
        next_action="Broaden or clear the conversation's retrieval scope",
        recovery_action="Conversation scope",
        authority_owner="Library retrieval service",
        stable_selector=LIBRARY_RAG_EMPTY_STATE_SELECTOR,
        disabled_tooltip=(
            f"{message}. Broaden or clear the conversation's retrieval scope."
        ),
    )
