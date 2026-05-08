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
                    yield Static(
                        result.snippet,
                        id=f"library-rag-result-snippet-{index}",
                    )
            elif self.state.retrieval_status == "searching":
                yield Static("Searching Library sources...", id="library-rag-searching")
            else:
                yield Static(self.state.next_action, id="library-rag-results-empty")


class LibrarySearchRagInspectorPanel(Vertical):
    """Display retrieval status and Console handoff readiness."""

    def __init__(self, state: LibraryRagPanelState, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state = state

    def compose(self) -> ComposeResult:
        yield Static("Retrieval Inspector", classes="destination-section")
        yield Static(
            f"Status: {self.state.retrieval_status.title()}",
            id="library-rag-retrieval-status",
        )
        yield Static(self.state.next_action, id="library-rag-next-action")
        if self.state.selected_result is not None:
            yield Static(
                f"Selected: {self.state.selected_result.title}",
                id="library-rag-selected-result",
            )
        else:
            yield Static(
                self.state.use_in_console_action.disabled_reason,
                id="library-rag-inspector-empty",
            )
        yield Button(
            self.state.use_in_console_action.label,
            id=self.state.use_in_console_action.widget_id,
            disabled=not self.state.use_in_console_action.enabled,
            tooltip=self.state.use_in_console_action.tooltip,
        )
