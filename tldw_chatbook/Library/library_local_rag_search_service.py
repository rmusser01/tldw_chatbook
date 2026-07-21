"""Local production backend for Library Search/RAG retrieval."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any, Optional

from loguru import logger

from tldw_chatbook.Chat.rag_scope import (
    EffectiveScope,
    SCOPE_REASON_CONVERSATIONS_EXCLUDED,
    SCOPE_STATUS_EXCLUDED,
    build_semantic_allowlists,
    media_id_params,
    note_id_params,
)
from tldw_chatbook.Library.library_fts_query import build_fts_match_query
from tldw_chatbook.Library.library_notes_sync_state import count_noun
from tldw_chatbook.Library.library_rag_service import LibraryRagSearchOutcome
from tldw_chatbook.Library.library_rag_state import (
    LIBRARY_RAG_EMPTY_STATE_SELECTOR,
    LIBRARY_RAG_QUERY_MAX_LENGTH,
    LIBRARY_RAG_SERVICE_ERROR_SELECTOR,
)

# Single source of truth for the pipeline diagnostics "scope" slot key
# (task-4/Backend A); reused here so the Library service's own
# conversations-excluded notice uses the exact same vocabulary rather than a
# parallel raw literal.
from tldw_chatbook.RAG_Search.pipeline_functions_simple import SCOPE_DIAGNOSTICS_KEY

# The shared factory is the single process-wide RAG service constructor
# (task-247): resolving through it guarantees Library RAG Answer queries read
# the exact vector store / collection / embedding model that ingestion-time
# indexing writes to.
from tldw_chatbook.RAG_Search.ingestion_indexing import get_shared_rag_service
from tldw_chatbook.UI.destination_recovery import DestinationRecoveryState
from tldw_chatbook.Utils.input_validation import sanitize_string, validate_text_input
from tldw_chatbook.Utils.optional_deps import embeddings_rag_deps_installed

logger = logger.bind(module="LibraryLocalRagSearchService")

_SEARCH_RUNTIME_BACKEND = "local-fts"
_RAG_RUNTIME_BACKEND = "rag-semantic"
_KNOWN_KEYWORD_SOURCE_TYPES = ("notes", "media", "conversations", "prompts")
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
    the runtime is unavailable.
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
                `rag` (delegates to the app's optional `_rag_service`).
            scope: Optional resolved RAG retrieval scope (rag-scope
                narrowing, task-6). Caller-passed only -- this service never
                resolves scope itself, so a Library-screen call site that
                never passes this keyword gets today's exact unrestricted
                behavior (spec decision D2). `None` or an unscoped scope
                performs unrestricted retrieval; a scoped value restricts
                keyword search to the scope's media/note id allowlists and
                excludes the conversations seam entirely (conversations are
                not part of the scope vocabulary, spec D5), and restricts
                semantic search via one store query per allowlisted source
                type, merged by score.
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
            return await self._search_semantic(
                query, source_types, top_k, kwargs, scope=scope
            )
        return await self._search_keyword(query, source_types, top_k, scope=scope)

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
                # _record_scope_conversations_excluded shape.
                diagnostics[SCOPE_DIAGNOSTICS_KEY] = {
                    "status": SCOPE_STATUS_EXCLUDED,
                    "reason": SCOPE_REASON_CONVERSATIONS_EXCLUDED,
                }
            else:
                coroutines["conversations"] = self._search_conversations(query, top_k)
        if "prompts" in source_types:
            coroutines["prompts"] = self._search_prompts(query, top_k)

        if not coroutines:
            return LibraryRagSearchOutcome(
                status="blocked",
                recovery_state=_no_backend_recovery_state(),
                diagnostics=diagnostics,
            )

        gathered = await asyncio.gather(*coroutines.values())
        outcomes = dict(zip(coroutines.keys(), gathered))

        if not any(available for available, _rows in outcomes.values()):
            return LibraryRagSearchOutcome(
                status="blocked",
                recovery_state=_no_backend_recovery_state(),
                diagnostics=diagnostics,
            )

        rows: list[dict[str, Any]] = []
        for source_type in _KNOWN_KEYWORD_SOURCE_TYPES:
            if source_type in outcomes:
                rows.extend(outcomes[source_type][1])

        if is_scoped and not rows:
            item_count = _scope_item_count(scope)
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
    ) -> tuple[bool, list[dict[str, Any]]]:
        """Search the notes seam. Returns (seam_available, rows)."""
        service = getattr(self._app, "notes_scope_service", None)
        if service is None:
            return False, []
        try:
            # Forward the allowlist only when provided so an unscoped call
            # keeps the exact legacy call shape (byte-identical, spec D2).
            allowlist_kwargs = (
                {"id_allowlist": id_allowlist} if id_allowlist is not None else {}
            )
            raw_results = await service.search_notes(
                scope="local_note",
                query=query,
                limit=top_k,
                user_id=user_id,
                # Pre-built MATCH string (plural/singular widened) so the
                # notes seam is not limited to its exact-phrase fallback --
                # FTS5 unicode61 has no stemming (task-185 UAT).
                fts_match_query=build_fts_match_query(query),
                **allowlist_kwargs,
            )
        except Exception:
            logger.opt(exception=True).warning(
                "Library keyword search: notes seam failed."
            )
            return True, []
        return True, [
            _note_row(item) for item in raw_results or () if isinstance(item, Mapping)
        ]

    async def _search_media(
        self,
        query: str,
        top_k: int,
        *,
        id_allowlist: Optional[Sequence[str]] = None,
    ) -> tuple[bool, list[dict[str, Any]]]:
        """Search the media seam. Returns (seam_available, rows)."""
        service = getattr(self._app, "media_reading_scope_service", None)
        if service is None:
            return False, []
        try:
            # Forward the allowlist only when provided so an unscoped call
            # keeps the exact legacy call shape (byte-identical, spec D2).
            allowlist_kwargs = (
                {"id_allowlist": id_allowlist} if id_allowlist is not None else {}
            )
            payload = await service.search_media(
                mode="local",
                query=query,
                limit=top_k,
                offset=0,
                fts_match_query=build_fts_match_query(query),
                **allowlist_kwargs,
            )
        except Exception:
            logger.opt(exception=True).warning(
                "Library keyword search: media seam failed."
            )
            return True, []
        items = payload.get("items", []) if isinstance(payload, Mapping) else []
        return True, [_media_row(item) for item in items if isinstance(item, Mapping)]

    async def _search_conversations(
        self,
        query: str,
        top_k: int,
    ) -> tuple[bool, list[dict[str, Any]]]:
        """Search the conversations seam. Returns (seam_available, rows)."""
        db = getattr(self._app, "chachanotes_db", None)
        if db is None:
            return False, []
        fts_query = build_fts_match_query(query)
        try:
            if getattr(db, "is_memory_db", False):
                # In-memory SQLite connections are thread-local and only the
                # thread that created the database has the migrated schema;
                # offloading to a worker thread would hit a blank connection.
                raw_results = db.search_conversations_by_content(fts_query, top_k)
            else:
                raw_results = await asyncio.to_thread(
                    db.search_conversations_by_content, fts_query, top_k
                )
        except Exception:
            logger.opt(exception=True).warning(
                "Library keyword search: conversations seam failed."
            )
            return True, []
        return True, [
            _conversation_row(item)
            for item in raw_results or ()
            if isinstance(item, Mapping)
        ]

    async def _search_prompts(
        self,
        query: str,
        top_k: int,
    ) -> tuple[bool, list[dict[str, Any]]]:
        """Search the prompts seam. Returns (seam_available, rows)."""
        service = getattr(self._app, "prompt_scope_service", None)
        if service is None:
            return False, []
        try:
            raw_results = await service.search_prompts(
                mode="local",
                query=query,
                limit=top_k,
                # Pre-built MATCH string (plural/singular widened), same as
                # the notes/media/conversations seams above.
                fts_match_query=build_fts_match_query(query),
            )
        except Exception:
            logger.opt(exception=True).warning(
                "Library keyword search: prompts seam failed."
            )
            return True, []
        return True, [
            _prompt_row(item) for item in raw_results or () if isinstance(item, Mapping)
        ]

    async def _search_semantic(
        self,
        query: str,
        source_types: tuple[str, ...],
        top_k: int,
        kwargs: Mapping[str, Any],
        *,
        scope: Optional[EffectiveScope] = None,
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
        """
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

        rows = [_semantic_row(item) for item in raw_results or ()]
        if source_types:
            rows = [
                row for row in rows if _semantic_row_matches_scope(row, source_types)
            ]
        if not raw_results and await self._semantic_index_is_empty(rag_service):
            return LibraryRagSearchOutcome(
                status="empty",
                recovery_state=_rag_index_empty_recovery_state(),
                runtime_backend=_RAG_RUNTIME_BACKEND,
            )
        if scope is not None and scope.state == "scoped" and not rows:
            item_count = _scope_item_count(scope)
            return LibraryRagSearchOutcome(
                status="empty",
                recovery_state=_scope_zero_results_recovery_state(item_count),
                runtime_backend=_RAG_RUNTIME_BACKEND,
            )
        return {"results": rows, "runtime_backend": _RAG_RUNTIME_BACKEND}

    async def _resolve_rag_runtime(self) -> Any:
        """Return a usable RAG runtime, lazily creating the shared one.

        Resolution order:

        1. An existing ``app._rag_service`` with a callable ``search`` always
           wins (already initialized by any surface, or injected by tests).
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
        rag_service = getattr(self._app, "_rag_service", None)
        if rag_service is not None and callable(getattr(rag_service, "search", None)):
            return rag_service
        if not embeddings_rag_deps_installed():
            return None
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
        try:
            # Cache on the app so every RAG surface (chat sidebar readiness,
            # repeat Library queries) sees the initialized runtime.
            self._app._rag_service = service
        except Exception:
            logger.opt(exception=True).debug(
                "Could not cache the shared RAG service on the app instance."
            )
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
        "snippet": f"Matched conversation · {count_noun(message_count, 'message')}",
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


async def _empty_scoped_seam() -> tuple[bool, list[dict[str, Any]]]:
    """Stand-in for a keyword seam whose source type is scoped to zero ids.

    The seam itself is available (the app has it configured) -- scope just
    leaves nothing for it to return -- so this reports ``(True, [])`` rather
    than ``(False, [])`` (unavailable), keeping ``_search_keyword``'s
    any-seam-available gate accurate when other seams still have results.
    """
    return True, []


def _scope_item_count(scope: EffectiveScope) -> int:
    """Total items across every source type in a scoped allowlist."""
    return sum(len(ids) for ids in scope.allowlist.values())


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
        next_action="Install embeddings support or switch mode to Search",
        recovery_action="Settings > RAG",
        authority_owner="Library retrieval service",
        stable_selector=LIBRARY_RAG_SERVICE_ERROR_SELECTOR,
        disabled_tooltip=(
            "RAG runtime is unavailable in this app instance. "
            "Install embeddings support or switch mode to Search."
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
            "Ingest content to index it automatically, run a semantic index "
            "backfill, or switch mode to Search"
        ),
        recovery_action="Library ingest",
        authority_owner="Library retrieval service",
        stable_selector=LIBRARY_RAG_EMPTY_STATE_SELECTOR,
        disabled_tooltip=(
            "The semantic index has no content yet. "
            "Ingest content to index it automatically or run a semantic index backfill."
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
