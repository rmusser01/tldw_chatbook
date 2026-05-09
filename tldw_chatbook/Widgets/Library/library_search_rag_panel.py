"""Textual widget for Library-native Search/RAG."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Static

from ...Library.library_rag_state import LibraryRagPanelState


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
        with Vertical(id="library-rag-source-scope", classes="library-rag-region"):
            yield Static(self.state.scope.heading, id="library-rag-source-scope-heading")
            for option in self.state.scope.options:
                selected = "selected" if option.selected else option.status
                yield Static(
                    f"{option.label}: {option.count_label} ({selected})",
                    id=f"library-rag-scope-{option.source_type}",
                )
            if self.state.scope.recovery_copy:
                yield Static(self.state.scope.recovery_copy, id="library-rag-scope-recovery")

        with Vertical(id="library-rag-query-controls", classes="library-rag-region"):
            yield Static(
                f"Mode: {self.state.query_state.mode_label} | Top {self.state.query_state.top_k}",
                id="library-rag-query-status",
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
            if self.state.query_state.recovery_copy:
                yield Static(self.state.query_state.recovery_copy, id="library-rag-query-recovery")

        with Vertical(id="library-rag-results", classes="library-rag-region"):
            yield Static("Evidence / Results", classes="destination-section")
            if self.state.results:
                for index, result in enumerate(self.state.results):
                    score = "" if result.score is None else f" | score {result.score:.3f}"
                    yield Static(
                        f"{index + 1}. {result.title}{score}",
                        id=f"library-rag-result-{index}",
                    )
                    yield Button(
                        "Select evidence",
                        id=f"library-rag-select-result-{index}",
                        classes="library-rag-result-action",
                        tooltip="Select this evidence result for Console handoff.",
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
            elif self.state.retrieval_status == "searching":
                yield Static("Searching Library sources...", id="library-rag-searching")
            elif self.state.recovery_copy and self.state.recovery_selector:
                yield Static(
                    self.state.recovery_copy,
                    id=self.state.recovery_selector,
                )
            else:
                yield Static(self.state.next_action, id="library-rag-results-empty")


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
        else:
            selected_state.update("")
            selected_state.display = False
            empty_state.update(state.use_in_console_action.disabled_reason)
            empty_state.display = True

        console_action = state.use_in_console_action
        console_button = self.query_one("#library-rag-use-in-console", Button)
        console_button.disabled = not console_action.enabled
        console_button.tooltip = console_action.tooltip
