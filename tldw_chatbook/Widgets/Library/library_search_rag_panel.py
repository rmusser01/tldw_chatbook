"""Textual widget for Library-native Search/RAG."""

from __future__ import annotations

from rich.markup import escape as escape_markup

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Collapsible, Input, Static
from textual.widget import Widget

from ...Library.library_rag_state import (
    LIBRARY_RAG_SCOPE_TOGGLE_SOURCE_TYPES,
    LibraryRagPanelState,
    LibraryRagQueryState,
    LibraryRagResultRow,
    LibraryRagScopeState,
    LibraryRagSourceOption,
    searching_status_line,
)


_SELECTED_EVIDENCE_DETAIL_IDS = (
    "library-rag-selected-status-heading",
    "library-rag-selected-status",
    "library-rag-selected-evidence-heading",
    "library-rag-selected-snippet",
    "library-rag-selected-citations",
    "library-rag-selected-authority-heading",
    "library-rag-selected-source",
    "library-rag-selected-authority",
    "library-rag-selected-eligibility",
    "library-rag-selected-allowed-heading",
    "library-rag-selected-allowed",
    "library-rag-selected-blocked-heading",
    "library-rag-selected-blocked",
    "library-rag-selected-recovery-heading",
    "library-rag-selected-recovery",
    "library-rag-selected-handoff-heading",
    "library-rag-selected-handoff",
    "library-rag-selected-future-heading",
    "library-rag-selected-future",
)


class LibrarySearchRagPanel(VerticalScroll):
    """Display the source scope, query controls, and evidence results."""

    def __init__(self, state: LibraryRagPanelState, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state = state

    def compose(self) -> ComposeResult:
        yield Static(
            "Library Search/RAG",
            id="library-rag-panel-title",
            classes="destination-section",
        )
        with Vertical(
            id="library-rag-query-controls",
            classes=_query_region_classes(self.state),
        ):
            yield Button(
                _mode_toggle_label(self.state),
                id="library-rag-mode-toggle",
                tooltip="Cycle Search/RAG mode.",
            )
            yield Input(
                value=self.state.query_state.query,
                placeholder="Ask or search Library sources",
                id="library-rag-query-input",
            )
            for child in library_rag_query_status_children(self.state):
                yield child
            yield Button(
                self.state.query_state.run_action.label,
                id=self.state.query_state.run_action.widget_id,
                disabled=not self.state.query_state.run_action.enabled,
                tooltip=self.state.query_state.run_action.tooltip,
            )

        with Vertical(
            id="library-rag-source-scope",
            classes=_scope_region_classes(self.state),
        ):
            yield Static(
                "Sources",
                id="library-rag-scope-heading",
                classes="destination-section",
            )
            yield Static(
                _scope_summary(self.state),
                id="library-rag-scope-summary",
            )
            for toggle in library_rag_scope_toggle_children(self.state):
                yield toggle
            for child in library_rag_scope_recovery_children(self.state):
                yield child

        with Vertical(id="library-rag-results", classes="library-rag-region"):
            yield Static(
                _results_heading_text(self.state),
                id="library-rag-results-heading",
                classes="destination-section",
            )
            for child in library_rag_results_body_children(self.state):
                yield child

        with Collapsible(
            title="Recent searches",
            collapsed=self.state.history_collapsed,
            id="library-rag-history",
        ):
            for child in library_rag_history_children(self.state):
                yield child


def _scope_summary(state: LibraryRagPanelState) -> str:
    """Return a compact source scope line for the main Search/RAG work lane.

    UX wave L6: drops the per-source counts -- the toggle buttons directly
    below (``library_rag_scope_toggle_children``) already carry each
    source's count, so restating them here was redundant chrome.
    """
    return "Scope: all local sources"


def _scope_toggle_label(option: LibraryRagSourceOption) -> str:
    """Return a toggle Button's visible label for one scope source option."""
    marker = "✓" if option.selected else "○"
    return f"{marker} {option.label} ({option.count})"


def library_rag_scope_toggle_children(state: LibraryRagPanelState) -> list[Widget]:
    """Return one full-width toggle `Button` per real source type (B2).

    Shared by the panel's own `compose()` and the screen's incremental
    refresh so both build identical toggles from the same state. Only
    `LIBRARY_RAG_SCOPE_TOGGLE_SOURCE_TYPES` (notes/media/conversations) get
    a toggle -- workspaces/collections have no retrieval seam of their own.

    Args:
        state: Current Library Search/RAG panel display state.

    Returns:
        One toggle `Button` per real source type, disabled when that
        source's count is 0.
    """
    return [
        Button(
            _scope_toggle_label(option),
            id=f"library-rag-scope-toggle-{option.source_type}",
            classes="library-rag-scope-toggle",
            disabled=not option.available,
            tooltip=f"Toggle {option.label} in the retrieval scope.",
        )
        for option in state.scope.options
        if option.source_type in LIBRARY_RAG_SCOPE_TOGGLE_SOURCE_TYPES
    ]


def library_rag_scope_shows_recovery(scope: LibraryRagScopeState) -> bool:
    """True when the scope region should render its recovery dump + Import media.

    Only the genuinely-empty-library case (no sources available at all)
    gets the full recovery presentation; a user deselecting every scope
    toggle with sources still available is covered by the query region's
    quiet line instead (A1/B2) -- an Import media button would not fix
    that case.
    """
    return bool(scope.recovery_copy) and not scope.has_available_sources


def library_rag_scope_recovery_children(state: LibraryRagPanelState) -> list[Widget]:
    """Return the scope region's recovery Static + Import media button, or none.

    Shared by `compose()` and the screen's incremental refresh.
    """
    if not library_rag_scope_shows_recovery(state.scope):
        return []
    return [
        Static(state.scope.recovery_copy, id="library-rag-scope-recovery"),
        Button(
            "Open Import media",
            id="library-rag-open-import-export",
            classes="library-rag-recovery-action",
            tooltip="Open Library Import media to add sources.",
        ),
    ]


def _query_blocked_is_quiet(query_state: LibraryRagQueryState) -> bool:
    """True when the run gate's blocker renders as a single quiet line (A1)."""
    return query_state.blocked_is_empty_query or query_state.blocked_is_no_scope


def library_rag_query_shows_full_recovery(query_state: LibraryRagQueryState) -> bool:
    """True when the query region should render the callout + recovery dump.

    Reserved for real failures (unsafe query, missing dependencies/index, no
    provider for RAG mode) -- the empty-query and no-scope gates render a
    single quiet line instead (A1), and the ready/searching states render
    neither.
    """
    return bool(query_state.recovery_copy) and not _query_blocked_is_quiet(query_state)


def library_rag_query_status_children(state: LibraryRagPanelState) -> list[Widget]:
    """Return the query region's conditional status widgets (A1/A2).

    Shared by `compose()` and the screen's incremental refresh. At most one
    of a single muted quiet line (empty-query / no-scope gates) or the full
    callout + recovery-copy block (real failures) renders; the ready and
    searching states render neither -- an enabled (or "Searching…") Run
    button plus the input IS the rest of the state.

    Args:
        state: Current Library Search/RAG panel display state.

    Returns:
        Zero, one, or two widgets to render between the query Input and the
        Run button.
    """
    query_state = state.query_state
    if query_state.blocked_is_empty_query:
        return [
            Static(
                "Enter a question or search query.",
                id="library-rag-query-quiet-line",
                classes="library-rag-quiet-line",
            )
        ]
    if query_state.blocked_is_no_scope:
        return [
            Static(
                "Select at least one source.",
                id="library-rag-query-quiet-line",
                classes="library-rag-quiet-line",
            )
        ]
    if library_rag_query_shows_full_recovery(query_state):
        reason = query_state.run_action.disabled_reason
        return [
            Static(
                f"Blocked | {reason}",
                id="library-rag-query-blocked-callout",
                classes="library-rag-callout is-blocked",
            ),
            Static(query_state.recovery_copy, id="library-rag-query-recovery"),
        ]
    return []


def _mode_toggle_label(state: LibraryRagPanelState) -> str:
    """Return the visible mode-cycle button label."""
    return f"mode: {state.query_state.mode_label} ▸"


def _results_heading_text(state: LibraryRagPanelState) -> str:
    """Return the Evidence region heading, surfacing top-k (A3)."""
    return f"Evidence · top {state.query_state.top_k} per source"


def library_rag_result_row_children(
    row: LibraryRagResultRow,
    index: int,
    selected_result_id: str,
) -> list[Widget]:
    """Return one evidence row's child widgets (C1): content first, Open primary.

    Shared by the panel's own `compose()` and the screen's incremental DOM
    refresh (`_refresh_library_rag_results_widgets`) so both build identical
    rows from the same state.

    Args:
        row: The evidence row to render.
        index: The row's position among the currently rendered results,
            used to build stable per-row widget ids.
        selected_result_id: The panel's currently selected result id, if any.

    Returns:
        Title -> badges -> snippet -> citations (when present) -> an action
        row with Open first (primary emphasis, when the row is openable)
        then Select evidence.
    """
    selected = row.result_id == selected_result_id
    score = "" if row.score is None else f" | score {row.score:.3f}"
    children: list[Widget] = [
        Static(
            f"{index + 1}. {row.title}{score}",
            id=f"library-rag-result-{index}",
            classes=(
                "library-rag-result-row is-selected"
                if selected
                else "library-rag-result-row"
            ),
        ),
        Static(
            row.row_badge_label,
            id=f"library-rag-result-badges-{index}",
            classes="library-rag-result-badges",
        ),
        Static(row.snippet, id=f"library-rag-result-snippet-{index}"),
    ]
    if row.citation_labels:
        children.append(
            Static(
                f"Citations: {', '.join(row.citation_labels)}",
                id=f"library-rag-result-citations-{index}",
            )
        )
    actions: list[Widget] = []
    if row.can_open:
        actions.append(
            Button(
                "Open",
                id=f"library-rag-open-result-{index}",
                classes="library-rag-result-open console-action-primary",
                tooltip="Open this result's source in its Library editor/viewer.",
            )
        )
    actions.append(
        Button(
            "Selected evidence" if selected else "Select evidence",
            id=f"library-rag-select-result-{index}",
            classes="library-rag-result-action",
            tooltip="Select this evidence result for Console handoff.",
        )
    )
    children.append(Horizontal(*actions, classes="library-rag-result-actions"))
    return children


def library_rag_results_body_children(state: LibraryRagPanelState) -> list[Widget]:
    """Return the Evidence region's body widgets below the heading.

    Shared by `compose()` and the screen's incremental refresh
    (`_refresh_library_rag_results_widgets`) so both render identically:
    exactly one of evidence rows (plus a per-row Console handoff button on
    the selected row), the in-flight searching line, explicit retrieval
    recovery copy, or empty-state guidance, depending on retrieval status
    and result count.

    Args:
        state: Current Library Search/RAG panel display state.

    Returns:
        The widgets to mount directly below the Evidence heading.
    """
    if state.results:
        children: list[Widget] = []
        for index, result in enumerate(state.results):
            children.extend(
                library_rag_result_row_children(result, index, state.selected_result_id)
            )
            if result.result_id == state.selected_result_id:
                children.append(
                    Button(
                        state.use_in_console_action.label,
                        id="library-rag-use-selected-in-console",
                        classes=(
                            "library-rag-console-action "
                            "library-rag-center-console-action"
                        ),
                        disabled=not state.use_in_console_action.enabled,
                        tooltip=state.use_in_console_action.tooltip,
                    )
                )
        return children
    if state.retrieval_status == "searching":
        return [
            Static(
                searching_status_line(state.scope.selected_source_types),
                id="library-rag-searching-line",
            )
        ]
    if state.recovery_copy and state.recovery_selector:
        return [Static(state.recovery_copy, id=state.recovery_selector)]
    return [
        Static(
            "No evidence yet. Run Search/RAG to populate results.",
            id="library-rag-results-empty",
        ),
        Static(
            _evidence_empty_guidance(),
            id="library-rag-evidence-empty-guidance",
            classes="library-rag-empty-guidance",
        ),
    ]


def library_rag_history_children(state: LibraryRagPanelState) -> list[Widget]:
    """Return the `Recent searches` collapsible's child widgets (D1).

    Shared by the widget's own `compose` and the screen's incremental
    DOM refresh so both build identical rows from the same state.

    Args:
        state: Current Library Search/RAG panel display state.

    Returns:
        When history is empty, a single muted placeholder `Static`.
        Otherwise: a muted hint `Static` first, then one full-width
        `Button` per history entry (most recent first), then a
        `Clear history` `Button` last.
    """
    if not state.history:
        return [
            Static(
                "No recent searches.",
                id="library-rag-history-empty",
                classes="library-rag-history-empty",
            )
        ]
    children: list[Widget] = [
        Static(
            "Select an entry to run it again.",
            id="library-rag-history-hint",
            classes="library-rag-history-hint",
        )
    ]
    children.extend(
        Button(
            # Textual parses a plain string Button label as markup: an
            # unescaped stored entry like "docs [/archive] cleanup" raises
            # MarkupError at construction time -- and because history is
            # persisted before this rebuild, the crash would recur on every
            # Search-canvas entry after restart. Escaping mirrors the
            # `_sanitize_display_text(escape=True)` path result titles and
            # snippets already use.
            escape_markup(entry),
            id=f"library-rag-history-{index}",
            classes="library-rag-history-row",
        )
        for index, entry in enumerate(state.history)
    )
    children.append(
        Button(
            "Clear history",
            id="library-rag-history-clear",
            classes="library-rag-history-clear",
        )
    )
    return children


def _evidence_empty_guidance() -> str:
    """Return empty evidence workflow guidance."""
    return "Add or import sources, run a query, then select evidence for Console."


def _console_handoff_summary(state: LibraryRagPanelState) -> str:
    """Return the inspector's Console handoff decision."""
    if state.use_in_console_action.enabled:
        return "\n".join(
            (
                "Use in Console: ready",
                "Why: selected evidence is workspace-eligible.",
                "Next: send selected evidence to Console.",
            )
        )
    return "\n".join(
        (
            "Use in Console: blocked",
            "Blocked: select usable evidence before Console handoff.",
        )
    )


def _inspector_recovery_summary(state: LibraryRagPanelState) -> str:
    """Return recovery guidance for the inspector decision panel."""
    if state.use_in_console_action.enabled:
        return "Recovery: choose a different evidence row, revise the query, or send to Console."
    return "Recovery: add or import sources, enter a query, run retrieval, then select evidence."


def _future_attribution_summary(state: LibraryRagPanelState) -> str:
    """Return future attribution scope without implying persistence is implemented."""
    if state.selected_result is not None:
        return (
            "Citation/snippet carry-through placeholder: preserve source, chunk, "
            "snippet, and citations."
        )
    return "Citation/snippet carry-through placeholder: reserved for selected evidence."


def _query_region_classes(state: LibraryRagPanelState) -> str:
    """Return query-region classes that reserve recovery height only when needed."""
    return (
        "library-rag-region has-recovery"
        if library_rag_query_shows_full_recovery(state.query_state)
        else "library-rag-region"
    )


def _scope_region_classes(state: LibraryRagPanelState) -> str:
    """Return source-scope classes that keep the ready state compact."""
    return (
        "library-rag-region has-recovery"
        if library_rag_scope_shows_recovery(state.scope)
        else "library-rag-region"
    )


class LibrarySearchRagInspectorPanel(Vertical):
    """Display retrieval status and Console handoff readiness."""

    def __init__(self, state: LibraryRagPanelState, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state = state

    def compose(self) -> ComposeResult:
        yield Static(
            "Retrieval Inspector",
            id="library-rag-inspector-title",
            classes="destination-section",
        )
        yield Static(
            "Retrieval Status",
            id="library-rag-retrieval-heading",
            classes="destination-section",
        )
        yield Static(
            f"Status: {self.state.retrieval_status.title()}",
            id="library-rag-retrieval-status",
        )
        yield Static(self.state.next_action, id="library-rag-next-action")
        yield Static(
            "Console Handoff",
            id="library-rag-console-handoff-heading",
            classes="destination-section",
        )
        yield Static(
            _console_handoff_summary(self.state),
            id="library-rag-console-handoff-status",
        )
        yield Button(
            self.state.use_in_console_action.label,
            id=self.state.use_in_console_action.widget_id,
            disabled=not self.state.use_in_console_action.enabled,
            tooltip=self.state.use_in_console_action.tooltip,
            classes="library-rag-console-action",
        )
        yield Static(
            "Selected Evidence",
            id="library-rag-selected-heading",
            classes="destination-section",
        )
        selected_result = self.state.selected_result
        selected_state = Static(
            (
                f"Selected: {selected_result.title}"
                if selected_result is not None
                else "Selected Evidence: none"
            ),
            id="library-rag-selected-result",
        )
        yield selected_state
        for widget_id, text in _selected_evidence_details(selected_result).items():
            detail = Static(
                text,
                id=widget_id,
                classes=_selected_evidence_detail_classes(widget_id),
            )
            detail.display = selected_result is not None
            yield detail
        empty_state = Static(
            "Blocked: select usable evidence before Console handoff.",
            id="library-rag-inspector-empty",
        )
        empty_state.display = selected_result is None
        yield empty_state
        yield Static(
            "Recovery",
            id="library-rag-inspector-recovery-heading",
            classes="destination-section",
        )
        yield Static(
            _inspector_recovery_summary(self.state),
            id="library-rag-inspector-recovery",
        )
        yield Static(
            "Future Attribution",
            id="library-rag-inspector-future-heading",
            classes="destination-section",
        )
        yield Static(
            _future_attribution_summary(self.state),
            id="library-rag-inspector-future",
        )

    def refresh_from_state(self, state: LibraryRagPanelState) -> None:
        """Update inspector copy and toggle stable selection widgets."""
        self.state = state
        self.query_one("#library-rag-retrieval-status", Static).update(
            f"Status: {state.retrieval_status.title()}"
        )
        self.query_one("#library-rag-next-action", Static).update(state.next_action)
        self.query_one("#library-rag-console-handoff-status", Static).update(
            _console_handoff_summary(state)
        )
        self.query_one("#library-rag-inspector-recovery", Static).update(
            _inspector_recovery_summary(state)
        )
        self.query_one("#library-rag-inspector-future", Static).update(
            _future_attribution_summary(state)
        )

        selected_state = self.query_one("#library-rag-selected-result", Static)
        empty_state = self.query_one("#library-rag-inspector-empty", Static)
        if state.selected_result is not None:
            selected_state.update(f"Selected: {state.selected_result.title}")
            selected_state.display = True
            empty_state.display = False
            for widget_id, text in _selected_evidence_details(
                state.selected_result
            ).items():
                detail = self.query_one(f"#{widget_id}", Static)
                detail.update(text)
                detail.display = True
        else:
            selected_state.update("Selected Evidence: none")
            selected_state.display = True
            for widget_id in _selected_evidence_detail_ids():
                detail = self.query_one(f"#{widget_id}", Static)
                detail.update("")
                detail.display = False
            empty_state.update("Blocked: select usable evidence before Console handoff.")
            empty_state.display = True

        console_action = state.use_in_console_action
        console_button = self.query_one("#library-rag-use-in-console", Button)
        console_button.disabled = not console_action.enabled
        console_button.tooltip = console_action.tooltip


def _selected_evidence_detail_ids() -> tuple[str, ...]:
    return _SELECTED_EVIDENCE_DETAIL_IDS


def _selected_evidence_detail_classes(widget_id: str) -> str:
    return "destination-section" if widget_id.endswith("-heading") else ""


def _selected_evidence_details(
    result: LibraryRagResultRow | None,
) -> dict[str, str]:
    if result is None:
        return {widget_id: "" for widget_id in _selected_evidence_detail_ids()}

    source_lines = [
        result.source_identity_label,
        result.runtime_label,
    ]
    if result.score_label:
        source_lines.append(result.score_label)
    citations = (
        ", ".join(result.citation_labels)
        if result.citation_labels
        else "No citation labels provided."
    )
    return {
        "library-rag-selected-status-heading": "Selected Status",
        "library-rag-selected-status": "Status: selected evidence ready for review",
        "library-rag-selected-evidence-heading": "Selected Evidence",
        "library-rag-selected-snippet": f"Snippet: {result.snippet}",
        "library-rag-selected-citations": f"Citations: {citations}",
        "library-rag-selected-authority-heading": "Authority & eligibility",
        "library-rag-selected-source": "\n".join(source_lines),
        "library-rag-selected-authority": result.authority_display_label,
        "library-rag-selected-eligibility": result.eligibility_label,
        "library-rag-selected-allowed-heading": "Allowed actions",
        "library-rag-selected-allowed": "Allowed: inspect snippet, review citations, use eligible evidence in Console",
        "library-rag-selected-blocked-heading": "Blocked actions",
        "library-rag-selected-blocked": "Blocked: answer generation and artifact citation persistence remain downstream work",
        "library-rag-selected-recovery-heading": "Recovery",
        "library-rag-selected-recovery": "Recovery: choose a different evidence row or revise the query",
        "library-rag-selected-handoff-heading": "Console Handoff",
        "library-rag-selected-handoff": result.handoff_label,
        "library-rag-selected-future-heading": "Future Attribution",
        "library-rag-selected-future": "Citation/snippet carry-through placeholder: preserve source authority through Console and Chatbooks",
    }
