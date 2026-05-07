"""Pure display-state contracts for Library-native Search/RAG."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from tldw_chatbook.Utils.input_validation import sanitize_string, validate_text_input


LIBRARY_RAG_SOURCE_TYPES = ("notes", "media", "conversations", "workspaces", "collections")
LIBRARY_RAG_SOURCE_LABELS = {
    "notes": "Notes",
    "media": "Media",
    "conversations": "Conversations",
    "workspaces": "Workspaces",
    "collections": "Collections",
}
LIBRARY_RAG_EMPTY_NEXT_ACTION = (
    "Add notes, media, conversations, Workspaces, or Collections before running retrieval."
)
LIBRARY_RAG_EMPTY_RECOVERY = (
    "No Library sources are available for Search/RAG. "
    f"Next: {LIBRARY_RAG_EMPTY_NEXT_ACTION} Owner: Library sources."
)
LIBRARY_RAG_EMPTY_QUERY_REASON = "Enter a Search/RAG query before running retrieval."
LIBRARY_RAG_EMPTY_QUERY_RECOVERY = (
    "Enter a question or search terms, then run Search/RAG against Library sources. "
    "Next: Provide a query. Owner: Library Search/RAG."
)
LIBRARY_RAG_NO_EVIDENCE_REASON = "Run Search/RAG and select evidence before using Console."
LIBRARY_RAG_RUNNING_REASON = "Search/RAG retrieval is already running."
LIBRARY_RAG_WAIT_FOR_RESULTS_REASON = "Wait for retrieval to finish before using Console."


def _clean_text(value: Any, fallback: str = "", *, max_length: int = 1000) -> str:
    text = sanitize_string(str(value or ""), max_length=max_length).strip()
    if not text:
        return fallback
    if not validate_text_input(text, max_length=max_length, allow_html=False):
        return fallback
    return text


def _coerce_count(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _coerce_score(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_citation_label(value: Any) -> str:
    if isinstance(value, Mapping):
        return _clean_text(
            value.get("label")
            or value.get("title")
            or value.get("citation")
            or value.get("source")
        )
    return _clean_text(value)


def _coerce_citation_labels(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raw_values: Iterable[Any] = (value,)
    elif isinstance(value, Iterable):
        raw_values = value
    else:
        raw_values = (value,)
    labels = tuple(label for item in raw_values if (label := _coerce_citation_label(item)))
    return labels


@dataclass(frozen=True)
class LibraryRagActionState:
    """One Library Search/RAG action and its disabled-state copy."""

    widget_id: str
    label: str
    enabled: bool
    disabled_reason: str = ""
    recovery_copy: str = ""
    authority_owner: str = "Library Search/RAG"
    classes: str = "destination-action-button library-rag-action"

    @property
    def tooltip(self) -> str:
        return "" if self.enabled else self.disabled_reason


@dataclass(frozen=True)
class LibraryRagScopeState:
    """Display state for Library Search/RAG source scope controls."""

    summary_label: str
    counts: Mapping[str, int] = field(default_factory=dict)
    source_count_labels: tuple[str, ...] = ()
    selectable_source_types: tuple[str, ...] = ()
    total_count: int = 0
    is_empty: bool = True
    recovery_copy: str = LIBRARY_RAG_EMPTY_RECOVERY
    next_action: str = LIBRARY_RAG_EMPTY_NEXT_ACTION
    authority_owner: str = "Library sources"

    @classmethod
    def from_source_counts(
        cls,
        *,
        notes: Any = 0,
        media: Any = 0,
        conversations: Any = 0,
        workspaces: Any = 0,
        collections: Any = 0,
    ) -> "LibraryRagScopeState":
        """Build source-scope display state from loose source counts.

        Args:
            notes: Count-like value for note sources.
            media: Count-like value for media sources.
            conversations: Count-like value for conversation sources.
            workspaces: Count-like value for workspace scopes.
            collections: Count-like value for collection scopes.

        Returns:
            Source-scope display state with normalized non-negative counts,
            selectable source types, and empty-state recovery copy.
        """
        counts = {
            "notes": _coerce_count(notes),
            "media": _coerce_count(media),
            "conversations": _coerce_count(conversations),
            "workspaces": _coerce_count(workspaces),
            "collections": _coerce_count(collections),
        }
        labels = tuple(
            f"{LIBRARY_RAG_SOURCE_LABELS[source_type]}: {counts[source_type]}"
            for source_type in LIBRARY_RAG_SOURCE_TYPES
        )
        selectable = tuple(
            source_type
            for source_type in LIBRARY_RAG_SOURCE_TYPES
            if counts[source_type] > 0
        )
        total_count = sum(counts.values())
        is_empty = total_count == 0
        return cls(
            summary_label="Source Scope: All local sources",
            counts=counts,
            source_count_labels=labels,
            selectable_source_types=selectable,
            total_count=total_count,
            is_empty=is_empty,
            recovery_copy=LIBRARY_RAG_EMPTY_RECOVERY if is_empty else "",
            next_action=LIBRARY_RAG_EMPTY_NEXT_ACTION if is_empty else "",
        )


@dataclass(frozen=True)
class LibraryRagQueryState:
    """Display state for the Library Search/RAG query controls."""

    query: str = ""
    mode: str = "rag"
    can_run: bool = False
    disabled_reason: str = LIBRARY_RAG_EMPTY_QUERY_REASON
    recovery_copy: str = LIBRARY_RAG_EMPTY_QUERY_RECOVERY
    authority_owner: str = "Library Search/RAG"

    @classmethod
    def from_values(cls, *, query: Any = "", mode: Any = "rag") -> "LibraryRagQueryState":
        """Build query display state from user-provided query controls.

        Args:
            query: User-entered query or loose seam value.
            mode: Requested retrieval mode. Supported values are ``rag`` and
                ``search``; invalid values fall back to ``rag``.

        Returns:
            Query display state with sanitized text, normalized mode, runnable
            state, and recovery copy when execution is blocked.
        """
        normalized_query = _clean_text(query, max_length=5000)
        normalized_mode = _clean_text(mode, "rag", max_length=32).lower()
        if normalized_mode not in {"rag", "search"}:
            normalized_mode = "rag"
        can_run = bool(normalized_query)
        return cls(
            query=normalized_query,
            mode=normalized_mode,
            can_run=can_run,
            disabled_reason="" if can_run else LIBRARY_RAG_EMPTY_QUERY_REASON,
            recovery_copy="" if can_run else LIBRARY_RAG_EMPTY_QUERY_RECOVERY,
        )


@dataclass(frozen=True)
class LibraryRagResultRow:
    """One normalized Library Search/RAG evidence row."""

    title: str
    snippet: str = ""
    score: float | None = None
    source_id: str = ""
    chunk_id: str = ""
    citation_labels: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def source_authority_label(self) -> str:
        return f"Source: {self.source_id or 'unknown'}"

    @classmethod
    def from_result(cls, result: Mapping[str, Any] | Any) -> "LibraryRagResultRow":
        """Normalize a retrieval result into a Library evidence row.

        Args:
            result: Mapping-like retrieval result. Missing, malformed, or
                unsafe values are tolerated and replaced with safe fallbacks.

        Returns:
            Evidence row preserving safe title, snippet, score, source ID,
            chunk ID, citation labels, and provenance metadata.
        """
        data: Mapping[str, Any] = result if isinstance(result, Mapping) else {}
        title = _clean_text(
            data.get("document_title")
            or data.get("title")
            or data.get("source_title")
            or data.get("name")
            or data.get("label"),
            "Untitled result",
            max_length=240,
        )
        snippet = _clean_text(
            data.get("snippet")
            or data.get("text")
            or data.get("content")
            or data.get("excerpt"),
            max_length=1000,
        )
        source_id = _clean_text(
            data.get("source_id")
            or data.get("document_id")
            or data.get("media_id")
            or data.get("note_id"),
            max_length=200,
        )
        chunk_id = _clean_text(data.get("chunk_id") or data.get("segment_id"), max_length=200)
        metadata = data.get("metadata")
        provenance = dict(metadata) if isinstance(metadata, Mapping) else {}
        return cls(
            title=title,
            snippet=snippet,
            score=_coerce_score(data.get("score")),
            source_id=source_id,
            chunk_id=chunk_id,
            citation_labels=_coerce_citation_labels(data.get("citations")),
            provenance=provenance,
        )


@dataclass(frozen=True)
class LibraryRagPanelState:
    """Display state for the destination-native Library Search/RAG panel."""

    scope: LibraryRagScopeState
    query: LibraryRagQueryState
    status: str
    status_label: str
    results: tuple[LibraryRagResultRow, ...] = ()
    run_action: LibraryRagActionState = field(default_factory=lambda: LibraryRagActionState(
        widget_id="library-rag-run-query",
        label="Run Search/RAG",
        enabled=False,
        disabled_reason=LIBRARY_RAG_EMPTY_QUERY_REASON,
        recovery_copy=LIBRARY_RAG_EMPTY_QUERY_RECOVERY,
    ))
    console_action: LibraryRagActionState = field(default_factory=lambda: LibraryRagActionState(
        widget_id="library-rag-use-in-console",
        label="Use in Console",
        enabled=False,
        disabled_reason=LIBRARY_RAG_NO_EVIDENCE_REASON,
    ))
    recovery_copy: str = ""
    next_action: str = ""
    authority_owner: str = "Library Search/RAG"

    @classmethod
    def from_values(
        cls,
        *,
        scope: LibraryRagScopeState | None = None,
        query: LibraryRagQueryState | None = None,
        results: Sequence[LibraryRagResultRow] | None = None,
        is_searching: bool = False,
        recovery_copy: Any = "",
    ) -> "LibraryRagPanelState":
        """Build aggregate panel state for the Library Search/RAG surface.

        Args:
            scope: Source-scope display state. Defaults to an empty Library
                scope when omitted.
            query: Query display state. Defaults to an empty blocked query when
                omitted.
            results: Normalized retrieval evidence rows to render.
            is_searching: Whether retrieval is currently in flight.
            recovery_copy: Optional blocker copy from an index, dependency,
                policy, or service failure.

        Returns:
            Panel display state with status, actions, result rows, recovery
            copy, and next action derived from scope/query/runtime state.
        """
        scope_state = scope or LibraryRagScopeState.from_source_counts()
        query_state = query or LibraryRagQueryState.from_values()
        result_rows = tuple(results or ())
        blocked_copy = _clean_text(recovery_copy, max_length=1000)

        if scope_state.is_empty:
            return cls(
                scope=scope_state,
                query=query_state,
                status="empty",
                status_label="No sources",
                results=result_rows,
                run_action=LibraryRagActionState(
                    widget_id="library-rag-run-query",
                    label="Run Search/RAG",
                    enabled=False,
                    disabled_reason=scope_state.recovery_copy,
                    recovery_copy=scope_state.recovery_copy,
                ),
                console_action=LibraryRagActionState(
                    widget_id="library-rag-use-in-console",
                    label="Use in Console",
                    enabled=False,
                    disabled_reason=LIBRARY_RAG_NO_EVIDENCE_REASON,
                ),
                recovery_copy=scope_state.recovery_copy,
                next_action=scope_state.next_action,
            )

        if not query_state.can_run or blocked_copy:
            reason = blocked_copy or query_state.disabled_reason
            return cls(
                scope=scope_state,
                query=query_state,
                status="blocked",
                status_label="Blocked",
                results=result_rows,
                run_action=LibraryRagActionState(
                    widget_id="library-rag-run-query",
                    label="Run Search/RAG",
                    enabled=False,
                    disabled_reason=reason,
                    recovery_copy=blocked_copy or query_state.recovery_copy,
                ),
                console_action=LibraryRagActionState(
                    widget_id="library-rag-use-in-console",
                    label="Use in Console",
                    enabled=False,
                    disabled_reason=LIBRARY_RAG_NO_EVIDENCE_REASON,
                ),
                recovery_copy=blocked_copy or query_state.recovery_copy,
                next_action=reason,
            )

        if is_searching:
            return cls(
                scope=scope_state,
                query=query_state,
                status="searching",
                status_label="Searching",
                results=result_rows,
                run_action=LibraryRagActionState(
                    widget_id="library-rag-run-query",
                    label="Run Search/RAG",
                    enabled=False,
                    disabled_reason=LIBRARY_RAG_RUNNING_REASON,
                ),
                console_action=LibraryRagActionState(
                    widget_id="library-rag-use-in-console",
                    label="Use in Console",
                    enabled=False,
                    disabled_reason=LIBRARY_RAG_WAIT_FOR_RESULTS_REASON,
                ),
                next_action=LIBRARY_RAG_WAIT_FOR_RESULTS_REASON,
            )

        return cls(
            scope=scope_state,
            query=query_state,
            status="ready",
            status_label="Ready",
            results=result_rows,
            run_action=LibraryRagActionState(
                widget_id="library-rag-run-query",
                label="Run Search/RAG",
                enabled=True,
            ),
            console_action=LibraryRagActionState(
                widget_id="library-rag-use-in-console",
                label="Use in Console",
                enabled=bool(result_rows),
                disabled_reason="" if result_rows else LIBRARY_RAG_NO_EVIDENCE_REASON,
            ),
            next_action=(
                "Review retrieved evidence and use it in Console."
                if result_rows
                else "Run Search/RAG to retrieve Library evidence."
            ),
        )
