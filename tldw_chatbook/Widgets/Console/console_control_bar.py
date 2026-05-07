"""Console-native provider/model/source readiness controls."""

from __future__ import annotations

from typing import Any, Callable

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.css.query import NoMatches
from textual.widgets import Static

from tldw_chatbook.Chat.console_display_state import ConsoleControlState
from tldw_chatbook.Widgets.compact_model_bar import CompactModelBar


class ConsoleControlBar(Horizontal):
    """Visible Console control strip outside the transcript region.

    The widget renders Console-owned provider, model, persona, RAG, source,
    tools, and approval labels plus the compact provider/model controls. It
    exposes `sync_state()` so `ChatScreen` can refresh labels after the user
    changes provider/model state through existing sidebar or compact controls.
    """

    def __init__(
        self,
        state: ConsoleControlState,
        app_instance: Any,
        *,
        on_sidebar_toggle_requested: Callable[[], Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Console control bar.

        Args:
            state: Display-state snapshot for Console readiness labels.
            app_instance: Main application instance used by `CompactModelBar`.
            on_sidebar_toggle_requested: Optional callback for routing compact
                sidebar-toggle requests to the embedded chat settings sidebar.
            **kwargs: Additional Textual widget arguments.
        """
        super().__init__(**kwargs)
        self.state = state
        self.app_instance = app_instance
        self.on_sidebar_toggle_requested = on_sidebar_toggle_requested

    def sync_state(self, state: ConsoleControlState) -> None:
        """Refresh visible label widgets from a new display-state snapshot.

        Args:
            state: Updated display-state snapshot to render.
        """
        self.state = state
        label_values = {
            "#console-provider-label": state.provider_label,
            "#console-model-label": state.model_label,
            "#console-persona-label": state.persona_label,
            "#console-rag-label": state.rag_label,
            "#console-sources-label": state.sources_label,
            "#console-tools-label": state.tools_label,
            "#console-approvals-label": state.approvals_label,
        }
        for selector, label in label_values.items():
            try:
                self.query_one(selector, Static).update(label)
            except NoMatches:
                continue

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
