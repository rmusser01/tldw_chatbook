"""Textual widget for Library-native Search/RAG."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Static

from ...Library.library_rag_state import LibraryRagPanelState, LibraryRagResultRow


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
                f"Mode: {self.state.query_state.mode_label} | Top {self.state.query_state.top_k}",
                id="library-rag-query-status",
            )
            yield Static(
                "Query",
                id="library-rag-query-label",
                classes="destination-section",
            )
            with Horizontal(id="library-rag-query-row"):
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
                _scope_summary(self.state),
                id="library-rag-scope-summary",
            )
            if self.state.scope.recovery_copy:
                yield Static(self.state.scope.recovery_copy, id="library-rag-scope-recovery")
                yield Button(
                    "Open Import/Export",
                    id="library-rag-open-import-export",
                    classes="library-rag-recovery-action",
                    tooltip="Open Library Import/Export to add sources.",
                )

        with Vertical(id="library-rag-results", classes="library-rag-region"):
            yield Static("Evidence / Results", classes="destination-section")
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
                yield Static("Searching Library sources...", id="library-rag-searching")
            elif self.state.recovery_copy and self.state.recovery_selector:
                yield Static(
                    self.state.recovery_copy,
                    id=self.state.recovery_selector,
                )
            else:
                yield Static(self.state.next_action, id="library-rag-results-empty")

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
            "Retrieval",
            id="library-rag-retrieval-heading",
            classes="destination-section",
        )
        yield Static(
            f"Status: {self.state.retrieval_status.title()}",
            id="library-rag-retrieval-status",
        )
        yield Static(self.state.next_action, id="library-rag-next-action")
        selected_result = self.state.selected_result
        selected_state = Static(
            f"Selected: {selected_result.title}" if selected_result is not None else "",
            id="library-rag-selected-result",
        )
        selected_state.display = selected_result is not None
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
            self.state.use_in_console_action.disabled_reason,
            id="library-rag-inspector-empty",
        )
        empty_state.display = selected_result is None
        yield empty_state
        yield Button(
            self.state.use_in_console_action.label,
            id=self.state.use_in_console_action.widget_id,
            disabled=not self.state.use_in_console_action.enabled,
            tooltip=self.state.use_in_console_action.tooltip,
            classes="library-rag-console-action",
        )

    def refresh_from_state(self, state: LibraryRagPanelState) -> None:
        """Update inspector copy and toggle stable selection widgets."""
        self.state = state
        self.query_one("#library-rag-retrieval-status", Static).update(
            f"Status: {state.retrieval_status.title()}"
        )
        self.query_one("#library-rag-next-action", Static).update(state.next_action)

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
            selected_state.update("")
            selected_state.display = False
            for widget_id in _selected_evidence_detail_ids():
                detail = self.query_one(f"#{widget_id}", Static)
                detail.update("")
                detail.display = False
            empty_state.update(state.use_in_console_action.disabled_reason)
            empty_state.display = True

        console_action = state.use_in_console_action
        console_button = self.query_one("#library-rag-use-in-console", Button)
        console_button.disabled = not console_action.enabled
        console_button.tooltip = console_action.tooltip


def _selected_evidence_detail_ids() -> tuple[str, ...]:
    return (
        "library-rag-selected-evidence-heading",
        "library-rag-selected-snippet",
        "library-rag-selected-citations",
        "library-rag-selected-authority-heading",
        "library-rag-selected-source",
        "library-rag-selected-authority",
        "library-rag-selected-eligibility",
        "library-rag-selected-handoff-heading",
        "library-rag-selected-handoff",
    )


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
        "library-rag-selected-evidence-heading": "Selected Evidence",
        "library-rag-selected-snippet": f"Snippet: {result.snippet}",
        "library-rag-selected-citations": f"Citations: {citations}",
        "library-rag-selected-authority-heading": "Authority & eligibility",
        "library-rag-selected-source": "\n".join(source_lines),
        "library-rag-selected-authority": result.authority_display_label,
        "library-rag-selected-eligibility": result.eligibility_label,
        "library-rag-selected-handoff-heading": "Console handoff",
        "library-rag-selected-handoff": result.handoff_label,
    }
