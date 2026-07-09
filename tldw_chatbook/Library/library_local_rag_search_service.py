"""Local production backend for Library Search/RAG retrieval."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from loguru import logger

from tldw_chatbook.Library.library_rag_service import LibraryRagSearchOutcome
from tldw_chatbook.Library.library_rag_state import LIBRARY_RAG_SERVICE_ERROR_SELECTOR
from tldw_chatbook.UI.destination_recovery import DestinationRecoveryState

logger = logger.bind(module="LibraryLocalRagSearchService")

_SEARCH_RUNTIME_BACKEND = "local-fts"
_RAG_RUNTIME_BACKEND = "rag-semantic"
_KNOWN_KEYWORD_SOURCE_TYPES = ("notes", "media", "conversations")


class LibraryLocalRagSearchService:
    """Keyword-first Library retrieval over the app's local source seams.

    `search` mode fans out over notes/media/conversations FTS seams and
    always works when at least one seam is available. `rag` mode delegates
    to the app's `_rag_service` and degrades to a blocked outcome with
    setup routing when that runtime is absent.
    """

    def __init__(self, app_instance: Any) -> None:
        self._app = app_instance

    async def search(self, query: str, scope: tuple[str, ...], mode: str, **kwargs: Any) -> Any:
        """Run a Library-native keyword or RAG retrieval request.

        Args:
            query: User question or search query to run against Library sources.
            scope: Selected Library source type identifiers (e.g. `notes`,
                `media`, `conversations`). Unknown types are ignored quietly.
            mode: Retrieval mode: `search` (keyword, local FTS seams) or
                `rag` (delegates to the app's optional `_rag_service`).
            **kwargs: Backend options. `top_k` caps the result count per
                source (default 5). `include_citations` is used in `rag`
                mode only.

        Returns:
            A mapping with `results`/`runtime_backend` keys for the caller
            to normalize into evidence rows, or a `LibraryRagSearchOutcome`
            directly for blocked states (missing local seams, missing RAG
            runtime).
        """
        top_k = max(1, int(kwargs.get("top_k") or 5))
        if mode == "rag":
            return await self._search_semantic(query, scope, top_k, kwargs)
        return await self._search_keyword(query, scope, top_k)

    async def _search_keyword(
        self,
        query: str,
        scope: tuple[str, ...],
        top_k: int,
    ) -> Any:
        """Fan out a keyword search over the notes/media/conversations seams."""
        user_id = getattr(self._app, "notes_user_id", None) or "default_user"
        coroutines: dict[str, Any] = {}
        if "notes" in scope:
            coroutines["notes"] = self._search_notes(query, top_k, user_id)
        if "media" in scope:
            coroutines["media"] = self._search_media(query, top_k)
        if "conversations" in scope:
            coroutines["conversations"] = self._search_conversations(query, top_k)

        if not coroutines:
            return LibraryRagSearchOutcome(
                status="blocked",
                recovery_state=_no_backend_recovery_state(),
            )

        gathered = await asyncio.gather(*coroutines.values())
        outcomes = dict(zip(coroutines.keys(), gathered))

        if not any(available for available, _rows in outcomes.values()):
            return LibraryRagSearchOutcome(
                status="blocked",
                recovery_state=_no_backend_recovery_state(),
            )

        rows: list[dict[str, Any]] = []
        for source_type in _KNOWN_KEYWORD_SOURCE_TYPES:
            if source_type in outcomes:
                rows.extend(outcomes[source_type][1])
        return {"results": rows, "runtime_backend": _SEARCH_RUNTIME_BACKEND}

    async def _search_notes(
        self,
        query: str,
        top_k: int,
        user_id: str,
    ) -> tuple[bool, list[dict[str, Any]]]:
        """Search the notes seam. Returns (seam_available, rows)."""
        service = getattr(self._app, "notes_scope_service", None)
        if service is None:
            return False, []
        try:
            raw_results = await service.search_notes(
                scope="local_note",
                query=query,
                limit=top_k,
                user_id=user_id,
            )
        except Exception:
            logger.warning("Library keyword search: notes seam failed.", exc_info=True)
            return True, []
        return True, [_note_row(item) for item in raw_results or () if isinstance(item, Mapping)]

    async def _search_media(
        self,
        query: str,
        top_k: int,
    ) -> tuple[bool, list[dict[str, Any]]]:
        """Search the media seam. Returns (seam_available, rows)."""
        service = getattr(self._app, "media_reading_scope_service", None)
        if service is None:
            return False, []
        try:
            payload = await service.search_media(mode="local", query=query, limit=top_k, offset=0)
        except Exception:
            logger.warning("Library keyword search: media seam failed.", exc_info=True)
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
        try:
            if getattr(db, "is_memory_db", False):
                # In-memory SQLite connections are thread-local and only the
                # thread that created the database has the migrated schema;
                # offloading to a worker thread would hit a blank connection.
                raw_results = db.search_conversations_by_content(query, top_k)
            else:
                raw_results = await asyncio.to_thread(
                    db.search_conversations_by_content, query, top_k
                )
        except Exception:
            logger.warning("Library keyword search: conversations seam failed.", exc_info=True)
            return True, []
        return True, [_conversation_row(item) for item in raw_results or () if isinstance(item, Mapping)]

    async def _search_semantic(
        self,
        query: str,
        scope: tuple[str, ...],  # noqa: ARG002 - accepted for parity with keyword search
        top_k: int,
        kwargs: Mapping[str, Any],
    ) -> Any:
        """Delegate to the app's optional RAG runtime, or report it as absent.

        `scope` is accepted for interface symmetry with keyword search but is
        not used to filter the RAG runtime's index today.
        """
        rag_service = getattr(self._app, "_rag_service", None)
        search = getattr(rag_service, "search", None)
        if rag_service is None or not callable(search):
            return LibraryRagSearchOutcome(
                status="blocked",
                recovery_state=_rag_mode_unavailable_recovery_state(),
            )
        include_citations = bool(kwargs.get("include_citations", True))
        raw_results = await search(
            query=query,
            top_k=top_k,
            search_type="semantic",
            include_citations=include_citations,
        )
        rows = [_semantic_row(item) for item in raw_results or ()]
        return {"results": rows, "runtime_backend": _RAG_RUNTIME_BACKEND}


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
    message_count = item.get("message_count") or 0
    return {
        "source_id": str(item.get("id", "")),
        "chunk_id": "",
        "title": item.get("title") or "",
        "snippet": f"Matched conversation · {message_count} messages",
        "score": _coerce_score(item.get("relevance_score")),
        "provenance": {"source_type": "conversation"},
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
            metadata.get("source_id") or metadata.get("document_id") or values.get("id") or ""
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


def _semantic_citation(citation: Any) -> dict[str, Any]:
    if isinstance(citation, Mapping):
        return dict(citation)
    return {
        "label": getattr(citation, "document_title", None) or getattr(citation, "text", None) or "Citation",
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


def _no_backend_recovery_state() -> DestinationRecoveryState:
    return DestinationRecoveryState(
        status_label="Unavailable",
        unavailable_what="Library Search/RAG retrieval",
        why="No local Library source seam (notes, media, or conversations) is available",
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
