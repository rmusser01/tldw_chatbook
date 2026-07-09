"""Textual widget for Library-native Search/RAG."""

from __future__ import annotations

from rich.markup import escape as escape_markup

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Collapsible, Input, Static
from textual.widget import Widget

from ...Library.library_rag_state import (
    LibraryRagPanelState,
    LibraryRagResultRow,
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


class LibrarySearchRagPanel(Vertical):
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
            yield Static(
                _section_rule("Query"),
                id="library-rag-query-section-rule",
                classes="library-rag-section-rule",
            )
            yield Static(
                "Retrieval Query",
                id="library-rag-query-heading",
                classes="destination-section",
            )
            yield Static(
                f"Mode: {self.state.query_state.mode_label} | Top {self.state.query_state.top_k}",
                id="library-rag-query-status",
            )
            yield Button(
                _mode_toggle_label(self.state),
                id="library-rag-mode-toggle",
                tooltip="Cycle Search/RAG mode.",
            )
            yield Static(_query_blocked_summary(self.state), id="library-rag-query-blocked")
            yield Static(
                _query_blocked_callout(self.state),
                id="library-rag-query-blocked-callout",
                classes=_query_callout_classes(self.state),
            )
            yield Input(
                value=self.state.query_state.query,
                placeholder="Ask or search Library sources",
                id="library-rag-query-input",
            )
            yield Button(
                self.state.query_state.run_action.label,
                id=self.state.query_state.run_action.widget_id,
                disabled=not self.state.query_state.run_action.enabled,
                tooltip=self.state.query_state.run_action.tooltip,
            )
            yield Static(
                _run_disabled_reason(self.state),
                id="library-rag-run-disabled-reason",
                classes="library-rag-disabled-reason",
            )
            yield Static(
                "Enter: run query | Tab: move panes | Enter on result: select | u: Use in Console",
                id="library-rag-query-shortcuts",
            )
            if self._show_query_recovery:
                yield Static(self.state.query_state.recovery_copy, id="library-rag-query-recovery")

        with Vertical(
            id="library-rag-source-scope",
            classes=_scope_region_classes(self.state),
        ):
            yield Static(
                _section_rule("Scope"),
                id="library-rag-scope-section-rule",
                classes="library-rag-section-rule",
            )
            yield Static(
                "Scope Controls",
                id="library-rag-scope-heading",
                classes="destination-section",
            )
            yield Static(
                _scope_summary(self.state),
                id="library-rag-scope-summary",
            )
            yield Static(
                _scope_table_header(),
                id="library-rag-scope-table-header",
                classes="library-rag-table-header",
            )
            for widget_id, copy in _scope_rows(self.state).items():
                yield Static(copy, id=widget_id, classes="library-rag-scope-row")
            if self.state.scope.recovery_copy:
                yield Static(self.state.scope.recovery_copy, id="library-rag-scope-recovery")
                yield Button(
                    "Open Import/Export",
                    id="library-rag-open-import-export",
                    classes="library-rag-recovery-action",
                    tooltip="Open Library Import/Export to add sources.",
                )

        with Vertical(id="library-rag-results", classes="library-rag-region"):
            yield Static(
                _section_rule("Evidence"),
                id="library-rag-results-section-rule",
                classes="library-rag-section-rule",
            )
            yield Static("Evidence Results", id="library-rag-results-heading", classes="destination-section")
            yield Static(
                _attribution_placeholder(self.state),
                id="library-rag-attribution-placeholder",
            )
            if self.state.results:
                for index, result in enumerate(self.state.results):
                    score = "" if result.score is None else f" | score {result.score:.3f}"
                    selected = result.result_id == self.state.selected_result_id
                    yield Static(
                        f"{index + 1}. {result.title}{score}",
                        id=f"library-rag-result-{index}",
                        classes=(
                            "library-rag-result-row is-selected"
                            if selected
                            else "library-rag-result-row"
                        ),
                    )
                    yield Button(
                        "Selected evidence" if selected else "Select evidence",
                        id=f"library-rag-select-result-{index}",
                        classes="library-rag-result-action",
                        tooltip="Select this evidence result for Console handoff.",
                    )
                    if result.can_open:
                        yield Button(
                            "Open",
                            id=f"library-rag-open-result-{index}",
                            classes="library-rag-result-open",
                            tooltip="Open this result's source in its Library editor/viewer.",
                        )
                    yield Static(
                        result.row_badge_label,
                        id=f"library-rag-result-badges-{index}",
                        classes="library-rag-result-badges",
                    )
                    yield Static(
                        result.snippet,
                        id=f"library-rag-result-snippet-{index}",
                    )
                    if result.citation_labels:
                        yield Static(
                            f"Citations: {', '.join(result.citation_labels)}",
                            id=f"library-rag-result-citations-{index}",
                        )
                    if selected:
                        yield Button(
                            self.state.use_in_console_action.label,
                            id="library-rag-use-selected-in-console",
                            classes=(
                                "library-rag-console-action "
                                "library-rag-center-console-action"
                            ),
                            disabled=not self.state.use_in_console_action.enabled,
                            tooltip=self.state.use_in_console_action.tooltip,
                        )
            elif self.state.retrieval_status == "searching":
                yield Static(
                    searching_status_line(self.state.scope.selected_source_types),
                    id="library-rag-searching-line",
                )
            elif self.state.recovery_copy and self.state.recovery_selector:
                yield Static(
                    self.state.recovery_copy,
                    id=self.state.recovery_selector,
                )
            else:
                yield Static(
                    "No evidence yet. Run Search/RAG to populate results.",
                    id="library-rag-results-empty",
                )
                yield Static(
                    _evidence_empty_guidance(),
                    id="library-rag-evidence-empty-guidance",
                    classes="library-rag-empty-guidance",
                )

        with Collapsible(
            title="Recent searches",
            collapsed=bool(self.state.results),
            id="library-rag-history",
        ):
            for child in library_rag_history_children(self.state):
                yield child

    @property
    def _show_query_recovery(self) -> bool:
        return bool(
            self.state.query_state.recovery_copy
            and self.state.scope.status != "blocked"
        )


def _scope_summary(state: LibraryRagPanelState) -> str:
    """Return a compact source scope line for the main Search/RAG work lane."""
    counts = {option.source_type: option.count for option in state.scope.options}
    return (
        "Scope: all local"
        f" | Notes {counts.get('notes', 0)}"
        f" | Media {counts.get('media', 0)}"
        f" | Conversations {counts.get('conversations', 0)}"
    )


def _scope_table_header() -> str:
    """Return the compact source-scope table header."""
    return "Scope                | Count        | Eligibility        | Next action"


def _scope_rows(state: LibraryRagPanelState) -> dict[str, str]:
    """Return Stage C source-scope rows with stable selectors."""
    counts = {option.source_type: option.count for option in state.scope.options}
    total = state.scope.total_count
    selected = len(state.scope.selected_source_types)
    notes = counts.get("notes", 0)
    media = counts.get("media", 0)
    conversations = counts.get("conversations", 0)
    collections = counts.get("collections", 0)
    return {
        "library-rag-scope-row-all": (
            f"All Library          | {total} sources    | Browse/search     | Add source"
        ),
        "library-rag-scope-row-workspace": (
            f"Workspace eligible   | {selected} scopes     | Stage after pick  | Select evidence"
        ),
        "library-rag-scope-row-notes": (
            f"Notes                | {notes} sources    | Retrieval-ready   | Run query"
        ),
        "library-rag-scope-row-media": (
            f"Media                | {media} sources    | Retrieval-ready   | Run query"
        ),
        "library-rag-scope-row-conversations": (
            f"Conversations        | {conversations} sources    | Retrieval-ready   | Run query"
        ),
        "library-rag-scope-row-collections": (
            f"Collections          | {collections} records    | Read/review WIP   | Open collection"
        ),
        "library-rag-scope-row-import-export": (
            "Import/Export recovery | add sources | Source intake      | Import source"
        ),
    }


def _query_blocked_summary(state: LibraryRagPanelState) -> str:
    """Return a one-line visible blocker for terminal screenshots."""
    reason = state.query_state.run_action.disabled_reason
    if not reason:
        return "Ready: run Search/RAG over selected Library sources."
    return f"Blocked: {reason[:1].lower()}{reason[1:]}"


def _query_blocked_callout(state: LibraryRagPanelState) -> str:
    """Return a visible query readiness callout."""
    reason = state.query_state.run_action.disabled_reason
    if not reason:
        return "Ready | Run retrieval over selected Library sources."
    return f"Blocked | {_query_recovery_sentence(reason)}"


def _query_recovery_sentence(reason: str) -> str:
    """Normalize query blocker copy for compact terminal callouts."""
    if reason == "Enter a question or search query.":
        return "Enter a question before running retrieval."
    if reason == "Select at least one Library source.":
        return "Select at least one Library source before running retrieval."
    return reason


def _run_disabled_reason(state: LibraryRagPanelState) -> str:
    """Return visible run-button readiness copy."""
    reason = state.query_state.run_action.disabled_reason
    if not reason:
        return "Run ready: selected Library sources are queryable."
    return f"Run disabled: {reason[:1].lower()}{reason[1:]}"


def _query_callout_classes(state: LibraryRagPanelState) -> str:
    """Return state classes for the query readiness callout."""
    if state.query_state.run_action.enabled:
        return "library-rag-callout is-ready"
    return "library-rag-callout is-blocked"


def _mode_toggle_label(state: LibraryRagPanelState) -> str:
    """Return the visible mode-cycle button label."""
    return f"mode: {state.query_state.mode_label} ▸"


def library_rag_history_children(state: LibraryRagPanelState) -> list[Widget]:
    """Return the `Recent searches` collapsible's child widgets.

    Shared by the widget's own `compose` and the screen's incremental
    DOM refresh so both build identical rows from the same state.

    Args:
        state: Current Library Search/RAG panel display state.

    Returns:
        One full-width `Button` per history entry (most recent first), or a
        single muted `Static` placeholder when there is no history yet.
    """
    if not state.history:
        return [
            Static(
                "No recent searches.",
                id="library-rag-history-empty",
                classes="library-rag-history-empty",
            )
        ]
    return [
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
    ]


def _evidence_empty_guidance() -> str:
    """Return empty evidence workflow guidance."""
    return "Add or import sources, run a query, then select evidence for Console."


def _attribution_placeholder(state: LibraryRagPanelState) -> str:
    """Return citation/snippet carry-through placeholder copy."""
    if state.selected_result is None:
        return "Citation/snippet carry-through: reserved for selected evidence."
    return "Citation/snippet carry-through placeholder: selected evidence preserves source, chunk, snippet, and citations."


def _section_rule(label: str) -> str:
    """Return a full-width terminal section rule label."""
    return f"-- {label} " + "-" * 48


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
        if state.query_state.recovery_copy and state.scope.status != "blocked"
        else "library-rag-region"
    )


def _scope_region_classes(state: LibraryRagPanelState) -> str:
    """Return source-scope classes that keep the ready state compact."""
    return (
        "library-rag-region has-recovery"
        if state.scope.recovery_copy
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
