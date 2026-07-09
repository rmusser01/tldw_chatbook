"""Pure display-state contracts for Library-native Search/RAG."""

from __future__ import annotations

from dataclasses import dataclass, replace
import html
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from rich.markup import escape as escape_markup

from tldw_chatbook.Utils.input_validation import (
    sanitize_string,
    validate_number_range,
    validate_text_input,
    validate_url,
)


LIBRARY_RAG_SOURCE_TYPES: tuple[tuple[str, str], ...] = (
    ("notes", "Notes"),
    ("media", "Media"),
    ("conversations", "Conversations"),
    ("workspaces", "Workspaces"),
    ("collections", "Collections"),
)
# The subset of LIBRARY_RAG_SOURCE_TYPES with a real per-source toggle in the
# Search canvas scope region (B2): workspaces/collections have no retrieval
# seam of their own yet, so they get no toggle row.
LIBRARY_RAG_SCOPE_TOGGLE_SOURCE_TYPES: tuple[str, ...] = ("notes", "media", "conversations")
LIBRARY_RAG_DEFAULT_TOP_K = 5
LIBRARY_RAG_RUN_ACTION_ID = "library-rag-run-query"
LIBRARY_RAG_USE_IN_CONSOLE_ACTION_ID = "library-rag-use-in-console"
LIBRARY_RAG_SERVICE_ERROR_SELECTOR = "library-rag-service-error"
LIBRARY_RAG_EMPTY_STATE_SELECTOR = "library-rag-empty-state"
LIBRARY_RAG_USE_IN_CONSOLE_DISABLED_REASON = (
    "Run a query and select usable evidence before sending to Console."
)
LIBRARY_RAG_QUERY_MAX_LENGTH = 2_000
LIBRARY_RAG_DISPLAY_MAX_LENGTH = 1_000
LIBRARY_RAG_SNIPPET_MAX_LENGTH = 4_000
LIBRARY_RAG_TOP_K_MAX = 50
LIBRARY_RAG_PROVENANCE_KEYS = frozenset(
    {
        "active_context_eligible",
        "active_workspace_id",
        "authority_label",
        "evidence_status",
        "eligibility_reason",
        "item_type",
        "source_type",
        "type",
        "workspace_id",
        "workspace_ids",
    }
)
_SCRIPT_BLOCK_PATTERN = re.compile(
    r"<script\b[^>]*>.*?</script\s*>",
    re.IGNORECASE | re.DOTALL,
)
LIBRARY_SEARCH_HISTORY_LIMIT = 10
LIBRARY_SEARCH_HISTORY_ENTRY_MAX_CHARS = 200
# Disabled-reason text for the two run-gate blockers that render as a single
# quiet line (A1) instead of the full callout + recovery-copy presentation.
# Both strings are unique across the gate ladder in
# `LibraryRagQueryState.from_values`, so `blocked_is_empty_query` /
# `blocked_is_no_scope` below can key off them directly.
_EMPTY_QUERY_DISABLED_REASON = "Enter a question or search query."
_NO_SCOPE_DISABLED_REASON = "Select at least one Library source."
LIBRARY_RAG_SEARCHING_LABEL = "Searching…"
_OPEN_SOURCE_TYPE_MAP = {
    "note": "notes", "notes": "notes",
    "media": "media", "media_chunk": "media",
    "conversation": "conversations", "conversations": "conversations",
    "chat": "conversations",
}


def update_search_history(history: Sequence[str], query: str) -> tuple[str, ...]:
    """Return search history with `query` prepended, deduped, capped at 10.

    Args:
        history: Existing history entries, most recent first.
        query: Newly submitted query; blank input leaves history unchanged.

    Returns:
        New history tuple, entries truncated to 200 chars, length <= 10.
    """
    entry = (query or "").strip()[:LIBRARY_SEARCH_HISTORY_ENTRY_MAX_CHARS]
    if not entry:
        return tuple(str(item) for item in history)
    deduped = [entry] + [str(item) for item in history if str(item) != entry]
    return tuple(deduped[:LIBRARY_SEARCH_HISTORY_LIMIT])


def searching_status_line(source_types: Sequence[str]) -> str:
    """Build the visible in-flight status line for a running search.

    Args:
        source_types: Selected source type IDs for the in-flight query.

    Returns:
        User-facing status line, e.g. `searching · notes, media…`.
    """
    labels = ", ".join(str(s) for s in source_types if str(s).strip())
    return f"searching · {labels}…" if labels else "searching…"


def _clean_text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    text = " ".join(str(value).strip().split())
    return text or fallback


def _remove_dangerous_display_patterns(value: str) -> tuple[str, bool]:
    scrubbed = _SCRIPT_BLOCK_PATTERN.sub("", value)
    changed = scrubbed != value
    for pattern in ("javascript:", "onclick=", "onerror="):
        if pattern in scrubbed.lower():
            scrubbed = re.sub(re.escape(pattern), "", scrubbed, flags=re.IGNORECASE)
            changed = True
    return scrubbed, changed


def _collapse_text(value: str, *, preserve_newlines: bool) -> str:
    if not preserve_newlines:
        return " ".join(value.strip().split())
    lines = (" ".join(line.strip().split()) for line in value.strip().splitlines())
    return "\n".join(line for line in lines if line)


def _sanitize_display_text(
    value: Any,
    fallback: str,
    *,
    max_length: int = LIBRARY_RAG_DISPLAY_MAX_LENGTH,
    preserve_newlines: bool = False,
    escape: bool = True,
) -> str:
    if value is None:
        return fallback
    sanitized = sanitize_string(str(value), max_length=max_length)
    scrubbed, _ = _remove_dangerous_display_patterns(sanitized)
    if not validate_text_input(scrubbed, max_length=max_length, allow_html=False):
        return fallback
    text = _collapse_text(scrubbed, preserve_newlines=preserve_newlines)
    if not text:
        return fallback
    return escape_markup(html.escape(text, quote=False)) if escape else text


def _sanitize_query(value: Any) -> tuple[str, bool]:
    if value is None:
        return "", False
    sanitized = sanitize_string(str(value), max_length=LIBRARY_RAG_QUERY_MAX_LENGTH)
    scrubbed, changed = _remove_dangerous_display_patterns(sanitized)
    valid = validate_text_input(
        scrubbed,
        max_length=LIBRARY_RAG_QUERY_MAX_LENGTH,
        allow_html=False,
    )
    if changed or not valid:
        return "", True
    return _collapse_text(scrubbed, preserve_newlines=False), False


def _sanitize_url(value: Any) -> str:
    text = _sanitize_display_text(value, "", max_length=2_000, escape=False)
    if not text:
        return ""
    if text.startswith("file://"):
        return text
    return text if validate_url(text) else ""


def _sentence(value: str) -> str:
    text = value.strip()
    if not text or text.endswith((".", "!", "?")):
        return text
    return f"{text}."


def _coerce_non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _coerce_positive_int(value: Any, fallback: int) -> int:
    if not validate_number_range(value, min_val=1, max_val=LIBRARY_RAG_TOP_K_MAX):
        return fallback
    coerced = int(value)
    return coerced if coerced > 0 else fallback


def _coerce_score(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_mode(value: Any) -> str:
    mode = _sanitize_display_text(value, "rag", max_length=32, escape=False).lower()
    return mode if mode in {"rag", "search"} else "rag"


def _recovery_copy(
    *,
    status_label: str,
    unavailable_what: str,
    why: str,
    next_action: str,
    recovery_action: str,
    owner: str,
) -> str:
    return "\n".join(
        (
            _sentence(status_label),
            f"Unavailable: {_sentence(unavailable_what)}",
            f"Why: {_sentence(why)}",
            f"Next: {_sentence(next_action)}",
            f"Recovery: {_sentence(recovery_action)}",
            f"Owner: {_sentence(owner)}",
        )
    )


@dataclass(frozen=True)
class LibraryRagActionState:
    """Display state for one Library Search/RAG action."""

    label: str
    enabled: bool
    widget_id: str
    disabled_reason: str = ""

    @property
    def tooltip(self) -> str:
        return "" if self.enabled else self.disabled_reason


@dataclass(frozen=True)
class LibraryRagSourceOption:
    """One source-scope option in Library Search/RAG."""

    source_type: str
    label: str
    count: int
    selected: bool
    status: str
    recovery: str = ""

    @property
    def available(self) -> bool:
        return self.count > 0

    @property
    def count_label(self) -> str:
        suffix = "source" if self.count == 1 else "sources"
        return f"{self.count} {suffix}"


@dataclass(frozen=True)
class LibraryRagScopeState:
    """Display state for Library Search/RAG source scope."""

    heading: str
    options: tuple[LibraryRagSourceOption, ...]
    selected_source_types: tuple[str, ...]
    total_count: int
    status: str = "ready"
    recovery_copy: str = ""

    @classmethod
    def from_source_counts(
        cls,
        *,
        notes: Any = 0,
        media: Any = 0,
        conversations: Any = 0,
        workspaces: Any = 0,
        collections: Any = 0,
        selected: Sequence[str] | None = None,
        heading: str = "Source Scope: All local sources",
    ) -> "LibraryRagScopeState":
        """Build source-scope display state from loose source counts.

        Args:
            notes: Available note source count.
            media: Available media source count.
            conversations: Available conversation source count.
            workspaces: Available workspace source count.
            collections: Available collection source count.
            selected: Selected source type IDs. `None` selects all available sources;
                an empty sequence represents an explicit empty selection.
            heading: User-facing source-scope heading.

        Returns:
            Display state for the Library Search/RAG source scope.
        """

        counts = {
            "notes": _coerce_non_negative_int(notes),
            "media": _coerce_non_negative_int(media),
            "conversations": _coerce_non_negative_int(conversations),
            "workspaces": _coerce_non_negative_int(workspaces),
            "collections": _coerce_non_negative_int(collections),
        }
        available_source_types = {
            source_type for source_type, count in counts.items() if count > 0
        }
        selected_source_types = available_source_types if selected is None else selected
        selected_values = {
            _clean_text(source_type).lower() for source_type in selected_source_types
        }
        options = tuple(
            LibraryRagSourceOption(
                source_type=source_type,
                label=label,
                count=counts[source_type],
                selected=source_type in selected_values and counts[source_type] > 0,
                status="ready" if counts[source_type] > 0 else "empty",
                recovery=(
                    ""
                    if counts[source_type] > 0
                    else f"No {source_type} available. Add or import {source_type} before querying."
                ),
            )
            for source_type, label in LIBRARY_RAG_SOURCE_TYPES
        )
        total_count = sum(counts.values())
        recovery_copy = ""
        status = "ready"
        if total_count == 0:
            status = "blocked"
            recovery_copy = "\n".join(
                (
                    _recovery_copy(
                        status_label="No sources",
                        unavailable_what="Library Search/RAG",
                        why="No Library sources are available for retrieval",
                        next_action="Add or import Library sources before querying",
                        recovery_action="Library Import/Export",
                        owner="Library source index",
                    ),
                    "Recovery checklist",
                    "1. Import Library sources.",
                    "2. Run Search/RAG.",
                    "3. Select evidence, then Use in Console.",
                )
            )
        elif not any(option.selected for option in options):
            status = "blocked"
            recovery_copy = _recovery_copy(
                status_label="No source selected",
                unavailable_what="Library Search/RAG",
                why="No Library source scope is selected",
                next_action="Select at least one Library source before querying",
                recovery_action="Library source scope",
                owner="Library source scope",
            )
        return cls(
            heading=heading,
            options=options,
            selected_source_types=tuple(
                option.source_type for option in options if option.selected
            ),
            total_count=total_count,
            status=status,
            recovery_copy=recovery_copy,
        )

    @property
    def has_available_sources(self) -> bool:
        return self.total_count > 0

    @property
    def has_selected_sources(self) -> bool:
        return bool(self.selected_source_types)

    def option_by_type(self, source_type: str) -> LibraryRagSourceOption:
        normalized_source_type = _clean_text(source_type).lower()
        for option in self.options:
            if option.source_type == normalized_source_type:
                return option
        raise KeyError(source_type)


@dataclass(frozen=True)
class LibraryRagQueryState:
    """Display state for Library Search/RAG query controls."""

    query: str
    mode: str
    mode_label: str
    top_k: int
    include_citations: bool
    status: str
    run_action: LibraryRagActionState
    recovery_copy: str = ""

    @property
    def blocked_is_empty_query(self) -> bool:
        """True when the run gate's blocker is a missing query (A1).

        The Search canvas renders a single muted line for this case instead
        of the full callout + recovery-copy presentation.
        """
        return self.run_action.disabled_reason == _EMPTY_QUERY_DISABLED_REASON

    @property
    def blocked_is_no_scope(self) -> bool:
        """True when the run gate's blocker is an empty source scope (A1/B2).

        Reached when no Library source is selected -- either no sources are
        available at all, or the user deselected every scope toggle. Like
        `blocked_is_empty_query`, this renders as a single muted line.
        """
        return self.run_action.disabled_reason == _NO_SCOPE_DISABLED_REASON

    @classmethod
    def from_values(
        cls,
        *,
        query: Any = "",
        mode: Any = "rag",
        top_k: Any = LIBRARY_RAG_DEFAULT_TOP_K,
        include_citations: bool = True,
        has_source_scope: bool = True,
        dependencies_ready: bool = True,
        index_ready: bool = True,
        provider_ready: bool = True,
    ) -> "LibraryRagQueryState":
        """Build query-control display state from UI or service values.

        Args:
            query: User query text.
            mode: Search mode, either `rag` or `search`; invalid values default to `rag`.
            top_k: Requested result count. Values outside the allowed range use the default.
            include_citations: Whether citation metadata should be requested/displayed.
            has_source_scope: Whether at least one source is selected.
            dependencies_ready: Whether Search/RAG optional dependencies are available.
            index_ready: Whether the selected source scope has an index.
            provider_ready: Whether a provider/model is ready for RAG-answer mode.

        Returns:
            Display state for query controls and the run action.
        """

        normalized_query, unsafe_query = _sanitize_query(query)
        normalized_mode = _normalize_mode(mode)
        mode_label = "Search" if normalized_mode == "search" else "RAG Answer"
        normalized_top_k = _coerce_positive_int(top_k, LIBRARY_RAG_DEFAULT_TOP_K)
        disabled_reason = ""
        owner = ""
        next_action = ""
        recovery_action = ""
        if unsafe_query:
            disabled_reason = "Enter a safe question or search query."
            owner = "user"
            next_action = "Remove markup or script content before running Search/RAG"
            recovery_action = "Query input"
        elif not has_source_scope:
            disabled_reason = _NO_SCOPE_DISABLED_REASON
            owner = "Library source scope"
            next_action = "Select or import a source before querying"
            recovery_action = "Library source scope"
        elif not normalized_query:
            disabled_reason = _EMPTY_QUERY_DISABLED_REASON
            owner = "user"
            next_action = "Type a query before running Search/RAG"
            recovery_action = "Query input"
        elif not dependencies_ready:
            disabled_reason = "Install or enable Search/RAG dependencies."
            owner = "optional dependency"
            next_action = "Install Search/RAG dependencies and restart"
            recovery_action = "Settings or package extras"
        elif not index_ready:
            disabled_reason = "Index selected Library sources before querying."
            owner = "Library source index"
            next_action = "Build or refresh the Library index"
            recovery_action = "Library indexing"
        elif normalized_mode == "rag" and not provider_ready:
            disabled_reason = "Select a provider/model before asking for a RAG answer."
            owner = "LLM provider"
            next_action = "Select a provider and model before running a RAG answer"
            recovery_action = "Console controls"

        enabled = not disabled_reason
        recovery_copy = ""
        if disabled_reason:
            recovery_copy = _recovery_copy(
                status_label="Blocked",
                unavailable_what="Run Library Search/RAG",
                why=disabled_reason,
                next_action=next_action,
                recovery_action=recovery_action,
                owner=owner,
            )
        return cls(
            query=normalized_query,
            mode=normalized_mode,
            mode_label=mode_label,
            top_k=normalized_top_k,
            include_citations=include_citations,
            status="ready" if enabled else "blocked",
            run_action=LibraryRagActionState(
                label="Run Search/RAG",
                enabled=enabled,
                widget_id=LIBRARY_RAG_RUN_ACTION_ID,
                disabled_reason=disabled_reason,
            ),
            recovery_copy=recovery_copy,
        )


@dataclass(frozen=True)
class LibraryRagCitation:
    """Normalized citation metadata for a Library Search/RAG result."""

    label: str
    url: str = ""
    source_id: str = ""
    chunk_id: str = ""


@dataclass(frozen=True)
class LibraryRagResultRow:
    """Normalized evidence row for Library Search/RAG results."""

    result_id: str
    title: str
    snippet: str
    score: float | None
    source_id: str
    chunk_id: str
    citations: tuple[LibraryRagCitation, ...]
    provenance: Mapping[str, Any]
    runtime_backend: str = ""

    @classmethod
    def from_result(cls, result: Mapping[str, Any] | Any) -> "LibraryRagResultRow":
        """Normalize a retrieval result into immutable evidence display state.

        Args:
            result: Retrieval result mapping from a local or remote Search/RAG adapter.

        Returns:
            Normalized evidence row with sanitized display text, citations, score, IDs,
            backend metadata, and immutable provenance.
        """

        values = result if isinstance(result, Mapping) else {}
        source_id = _sanitize_display_text(values.get("source_id"), "", escape=False)
        chunk_id = _sanitize_display_text(values.get("chunk_id"), "", escape=False)
        title = _sanitize_display_text(
            values.get("document_title")
            or values.get("title")
            or values.get("source_title"),
            "Untitled source",
        )
        snippet = _sanitize_display_text(
            values.get("snippet") or values.get("text") or values.get("content"),
            "No snippet available.",
            max_length=LIBRARY_RAG_SNIPPET_MAX_LENGTH,
            preserve_newlines=True,
        )
        citations = tuple(
            _normalize_citation(citation)
            for citation in _as_sequence(values.get("citations"))
        )
        provenance_value = values.get("provenance")
        provenance = (
            dict(provenance_value) if isinstance(provenance_value, Mapping) else {}
        )
        for key in LIBRARY_RAG_PROVENANCE_KEYS:
            if key in values and key not in provenance:
                provenance[key] = values[key]
        result_id = _result_id(source_id, chunk_id, title)
        return cls(
            result_id=result_id,
            title=title,
            snippet=snippet,
            score=_coerce_score(values.get("score")),
            source_id=source_id,
            chunk_id=chunk_id,
            citations=citations,
            provenance=MappingProxyType(provenance),
            runtime_backend=_sanitize_display_text(
                values.get("runtime_backend"),
                "",
                escape=False,
            ),
        )

    @property
    def citation_labels(self) -> tuple[str, ...]:
        return tuple(citation.label for citation in self.citations)

    @property
    def source_type_badge_label(self) -> str:
        """Compact source type label for evidence rows."""
        return (
            _provenance_text(self.provenance, "source_type")
            or _provenance_text(self.provenance, "item_type")
            or _provenance_text(self.provenance, "type")
            or "source"
        )

    @property
    def workspace_badge_label(self) -> str:
        """Compact workspace authority label for evidence rows."""
        workspace_ids = _provenance_text_tuple(self.provenance, "workspace_ids")
        workspace_id = _provenance_text(self.provenance, "workspace_id")
        if workspace_id and workspace_id not in workspace_ids:
            workspace_ids = (*workspace_ids, workspace_id)
        if not workspace_ids:
            return "all workspaces"
        if len(workspace_ids) == 1:
            return workspace_ids[0]
        return f"{len(workspace_ids)} workspaces"

    @property
    def citation_count_badge_label(self) -> str:
        """Compact citation count label for evidence rows."""
        count = len(self.citations)
        suffix = "citation" if count == 1 else "citations"
        return f"{count} {suffix}"

    @property
    def eligibility_badge_label(self) -> str:
        """Compact active-context eligibility label for evidence rows."""
        explicit_eligible = _coerce_optional_bool(
            self.provenance.get("active_context_eligible")
        )
        if explicit_eligible is True:
            return "eligible"
        if explicit_eligible is False:
            return "blocked"
        workspace_ids = _provenance_text_tuple(self.provenance, "workspace_ids")
        if workspace_ids and not _provenance_text(self.provenance, "active_workspace_id"):
            return "blocked"
        return "eligible"

    @property
    def row_badge_label(self) -> str:
        """One-line source authority summary for result list scanning."""
        return " | ".join(
            (
                self.source_type_badge_label,
                self.workspace_badge_label,
                self.citation_count_badge_label,
                self.eligibility_badge_label,
            )
        )

    @property
    def source_identity_label(self) -> str:
        """User-facing source/chunk identity for selected evidence inspection."""
        if self.source_id and self.chunk_id:
            return f"Source: {self.source_id} / {self.chunk_id}"
        if self.source_id:
            return f"Source: {self.source_id}"
        if self.chunk_id:
            return f"Chunk: {self.chunk_id}"
        return "Source: unavailable"

    @property
    def score_label(self) -> str:
        """User-facing retrieval score when the adapter provides one."""
        return "" if self.score is None else f"Score: {self.score:.3f}"

    @property
    def runtime_label(self) -> str:
        """User-facing runtime/backend identity for selected evidence inspection."""
        return f"Runtime: {self.runtime_backend or 'local'}"

    @property
    def authority_display_label(self) -> str:
        """User-facing authority label aligned with evidence handoff metadata."""
        explicit_label = _provenance_text(self.provenance, "authority_label")
        if explicit_label:
            return f"Authority: {explicit_label}"

        workspace_ids = _provenance_text_tuple(self.provenance, "workspace_ids")
        workspace_id = _provenance_text(self.provenance, "workspace_id")
        if workspace_id and workspace_id not in workspace_ids:
            workspace_ids = (*workspace_ids, workspace_id)
        if workspace_ids:
            return f"Authority: Workspace: {', '.join(workspace_ids)}"

        runtime_backend = self.runtime_backend.lower()
        source_authority = (
            "server"
            if runtime_backend.startswith("server") or "server" in runtime_backend
            else "local"
        )
        return f"Authority: Source authority: {source_authority}"

    @property
    def eligibility_label(self) -> str:
        """User-facing active-context eligibility for selected evidence inspection."""
        explicit_eligible = _coerce_optional_bool(
            self.provenance.get("active_context_eligible")
        )
        explicit_reason = _provenance_text(self.provenance, "eligibility_reason")
        if explicit_eligible is True:
            return "Eligibility: available for active workspace"
        if explicit_eligible is False:
            reason = explicit_reason.replace("_", " ") if explicit_reason else "blocked"
            return f"Eligibility: blocked for active workspace ({reason})"

        workspace_ids = _provenance_text_tuple(self.provenance, "workspace_ids")
        if workspace_ids and not _provenance_text(self.provenance, "active_workspace_id"):
            return "Eligibility: blocked until an active workspace is available"
        return "Eligibility: available for active context"

    @property
    def handoff_label(self) -> str:
        """User-facing statement of what the Console handoff preserves."""
        return "Handoff: snippet + citations + source/chunk IDs"

    @property
    def open_source_type(self) -> str:
        """Library canvas target this result can open, or empty string."""
        raw = str(
            self.provenance.get("source_type")
            or self.provenance.get("item_type")
            or self.provenance.get("type")
            or ""
        ).strip().lower()
        return _OPEN_SOURCE_TYPE_MAP.get(raw, "")

    @property
    def can_open(self) -> bool:
        """True when the row carries a resolvable parent id and known type."""
        return bool(self.open_source_type and self.source_id)


@dataclass(frozen=True)
class LibraryRagPanelState:
    """Display state for the destination-native Library Search/RAG panel."""

    scope: LibraryRagScopeState
    query_state: LibraryRagQueryState
    results: tuple[LibraryRagResultRow, ...]
    retrieval_status: str
    next_action: str
    use_in_console_action: LibraryRagActionState
    selected_result_id: str = ""
    selected_result: LibraryRagResultRow | None = None
    recovery_copy: str = ""
    recovery_selector: str = ""
    history: tuple[str, ...] = ()
    history_collapsed: bool = False

    @classmethod
    def from_values(
        cls,
        *,
        source_counts: Mapping[str, Any] | None = None,
        query: Any = "",
        mode: Any = "rag",
        results: Sequence[LibraryRagResultRow | Mapping[str, Any]] = (),
        selected_result_id: Any = "",
        retrieval_status: Any = "",
        recovery_copy: Any = "",
        recovery_selector: Any = "",
        dependencies_ready: bool = True,
        index_ready: bool = True,
        provider_ready: bool = True,
        selected_source_types: Sequence[str] | None = None,
        history: Sequence[str] = (),
        history_collapsed: bool = False,
    ) -> "LibraryRagPanelState":
        """Build full Library Search/RAG panel display state.

        Args:
            source_counts: Available source counts keyed by source type.
            query: User query text.
            mode: Search mode, either `rag` or `search`.
            results: Retrieval result rows or mappings.
            selected_result_id: Result ID selected for inspector/Console handoff.
            retrieval_status: Explicit retrieval status override.
            recovery_copy: Explicit retrieval recovery copy from a service outcome.
            recovery_selector: Stable selector used for explicit retrieval recovery.
            dependencies_ready: Whether Search/RAG optional dependencies are available.
            index_ready: Whether the selected source scope has an index.
            provider_ready: Whether a provider/model is ready for RAG-answer mode.
            selected_source_types: Selected source type IDs. `None` selects all available
                source types; an empty sequence represents no selected sources.
            history: Prior submitted queries, most recent first.
            history_collapsed: Whether the `Recent searches` collapsible should
                render collapsed (D1). The caller owns this decision -- it is
                only forced on the results-arrival transition, not on every
                render -- so this is a plain passthrough, not derived here.

        Returns:
            Display state for the destination-native Library Search/RAG panel.
        """

        counts = dict(source_counts or {})
        scope = LibraryRagScopeState.from_source_counts(
            notes=counts.get("notes", 0),
            media=counts.get("media", 0),
            conversations=counts.get("conversations", 0),
            workspaces=counts.get("workspaces", 0),
            collections=counts.get("collections", 0),
            selected=selected_source_types,
        )
        query_state = LibraryRagQueryState.from_values(
            query=query,
            mode=mode,
            has_source_scope=scope.has_selected_sources,
            dependencies_ready=dependencies_ready,
            index_ready=index_ready,
            provider_ready=provider_ready,
        )
        result_rows = tuple(
            result
            if isinstance(result, LibraryRagResultRow)
            else LibraryRagResultRow.from_result(result)
            for result in results
        )
        normalized_selected_result_id = _clean_text(selected_result_id)
        selected_result = next(
            (
                result
                for result in result_rows
                if result.result_id == normalized_selected_result_id
            ),
            None,
        )
        explicit_status = _clean_text(retrieval_status).lower()
        explicit_recovery_copy = _sanitize_display_text(
            recovery_copy,
            "",
            preserve_newlines=True,
        )
        explicit_recovery_selector = _sanitize_display_text(
            recovery_selector,
            "",
            max_length=128,
            escape=False,
        )
        active_recovery_selector = ""
        if query_state.status == "blocked":
            normalized_status = "blocked"
            recovery_copy = scope.recovery_copy or query_state.recovery_copy
            next_action = _blocked_next_action(recovery_copy)
        elif explicit_status == "searching":
            normalized_status = "searching"
            recovery_copy = ""
            next_action = "Wait for retrieval results."
        elif explicit_status in {"blocked", "failed"}:
            normalized_status = explicit_status
            recovery_copy = explicit_recovery_copy or _recovery_copy(
                status_label="Retrieval unavailable",
                unavailable_what="Library Search/RAG retrieval",
                why="Library retrieval could not complete",
                next_action="Retry the query or check Library indexing",
                recovery_action="Retry",
                owner="Library retrieval",
            )
            next_action = _blocked_next_action(recovery_copy)
            active_recovery_selector = (
                explicit_recovery_selector or LIBRARY_RAG_SERVICE_ERROR_SELECTOR
            )
        elif explicit_status == "empty" or (
            explicit_status == "ready" and not result_rows
        ):
            normalized_status = "empty"
            recovery_copy = explicit_recovery_copy or _recovery_copy(
                status_label="No results",
                unavailable_what="Library Search/RAG evidence",
                why="No evidence matched the current query",
                next_action="Revise the query or broaden the source scope",
                recovery_action="Query input or source scope",
                owner="Library retrieval",
            )
            next_action = "Revise the query or broaden the source scope."
            active_recovery_selector = (
                explicit_recovery_selector or LIBRARY_RAG_EMPTY_STATE_SELECTOR
            )
        elif result_rows:
            normalized_status = "ready"
            recovery_copy = ""
            next_action = "Review cited evidence or send the selected result to Console."
        else:
            normalized_status = "ready"
            recovery_copy = ""
            next_action = "Run Search/RAG over the selected Library sources."

        if normalized_status == "searching":
            # C2: the run action itself carries the in-flight state -- label
            # "Searching…" (an ellipsis character, one unit), disabled, so
            # the canvas never shows an enabled Run button while a query is
            # already running. Only reachable when the run gate was open
            # (query_state.status != "blocked"), so there is always a
            # well-formed prior run_action to replace.
            query_state = replace(
                query_state,
                run_action=LibraryRagActionState(
                    label=LIBRARY_RAG_SEARCHING_LABEL,
                    enabled=False,
                    widget_id=LIBRARY_RAG_RUN_ACTION_ID,
                    disabled_reason="Search in progress.",
                ),
            )

        can_use_console = normalized_status == "ready" and selected_result is not None
        return cls(
            scope=scope,
            query_state=query_state,
            results=result_rows,
            retrieval_status=normalized_status,
            next_action=next_action,
            use_in_console_action=LibraryRagActionState(
                label="Use in Console",
                enabled=can_use_console,
                widget_id=LIBRARY_RAG_USE_IN_CONSOLE_ACTION_ID,
                disabled_reason=(
                    "" if can_use_console else LIBRARY_RAG_USE_IN_CONSOLE_DISABLED_REASON
                ),
            ),
            selected_result_id=normalized_selected_result_id,
            selected_result=selected_result,
            recovery_copy=recovery_copy,
            recovery_selector=active_recovery_selector,
            history=tuple(str(h) for h in history),
            history_collapsed=bool(history_collapsed),
        )


def _as_sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    if value:
        return (value,)
    return ()


def _normalize_citation(value: Any) -> LibraryRagCitation:
    if isinstance(value, Mapping):
        label = _sanitize_display_text(
            value.get("label")
            or value.get("title")
            or value.get("url")
            or value.get("source_id"),
            "Citation",
        )
        return LibraryRagCitation(
            label=label,
            url=_sanitize_url(value.get("url")),
            source_id=_sanitize_display_text(value.get("source_id"), "", escape=False),
            chunk_id=_sanitize_display_text(value.get("chunk_id"), "", escape=False),
        )
    return LibraryRagCitation(label=_sanitize_display_text(value, "Citation"))


def _provenance_text(provenance: Mapping[str, Any], key: str) -> str:
    return _sanitize_display_text(
        provenance.get(key),
        "",
        max_length=LIBRARY_RAG_DISPLAY_MAX_LENGTH,
    )


def _provenance_text_tuple(provenance: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = provenance.get(key)
    values = _as_sequence(value)
    normalized = tuple(
        _sanitize_display_text(
            item,
            "",
            max_length=LIBRARY_RAG_DISPLAY_MAX_LENGTH,
        )
        for item in values
    )
    return tuple(text for text in normalized if text)


def _coerce_optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "enabled", "eligible", "available"}:
        return True
    if text in {"0", "false", "no", "n", "disabled", "blocked", "ineligible"}:
        return False
    return None


def _result_id(source_id: str, chunk_id: str, title: str) -> str:
    if source_id and chunk_id:
        return f"{source_id}:{chunk_id}"
    if source_id:
        return source_id
    if chunk_id:
        return chunk_id
    return f"result:{title.lower().replace(' ', '-')}"


def _blocked_next_action(recovery_copy: str) -> str:
    for line in recovery_copy.splitlines():
        if line.startswith("Next: "):
            return line.removeprefix("Next: ")
    return "Resolve the blocker before running Search/RAG."
