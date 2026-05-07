"""Console-native provider/model/source readiness controls."""

from __future__ import annotations

from typing import Any, Callable

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static

from tldw_chatbook.Chat.console_display_state import ConsoleControlState
from tldw_chatbook.Widgets.compact_model_bar import CompactModelBar


class ConsoleControlBar(Horizontal):
    """Visible Console control strip outside the transcript region."""

    def __init__(
        self,
        state: ConsoleControlState,
        app_instance: Any,
        *,
        on_sidebar_toggle_requested: Callable[[], Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.state = state
        self.app_instance = app_instance
        self.on_sidebar_toggle_requested = on_sidebar_toggle_requested

    def compose(self) -> ComposeResult:
        yield Static(
            self.state.provider_label,
            id="console-provider-label",
            classes="console-control-label",
        )
        yield Static(
            self.state.model_label,
            id="console-model-label",
            classes="console-control-label",
        )
        yield Static(
            self.state.persona_label,
            id="console-persona-label",
            classes="console-control-label",
        )
        yield Static(
            self.state.rag_label,
            id="console-rag-label",
            classes="console-control-label",
        )
        yield Static(
            self.state.sources_label,
            id="console-sources-label",
            classes="console-control-label",
        )
        yield Static(
            self.state.tools_label,
            id="console-tools-label",
            classes="console-control-label",
        )
        yield Static(
            self.state.approvals_label,
            id="console-approvals-label",
            classes="console-control-label",
        )
        yield CompactModelBar(
            self.app_instance,
            on_sidebar_toggle_requested=self.on_sidebar_toggle_requested,
            id="console-compact-model-bar",
            classes="console-compact-model-bar",
        )
