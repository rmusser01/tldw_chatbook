"""Console-native provider/model/source readiness controls."""

from __future__ import annotations

from typing import Any, Callable

from textual.app import ComposeResult
from textual import on
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_display_state import ConsoleControlState
from tldw_chatbook.UI.Workbench.workbench_widgets import WorkbenchActionRequested
from tldw_chatbook.Widgets.compact_model_bar import CompactModelBar


CONSOLE_CONTROL_BAR_HEIGHT = 2


def _summary_line(state: ConsoleControlState) -> str:
    return " | ".join(
        (
            state.provider_label,
            state.model_label,
            state.persona_label,
            state.rag_label,
            state.sources_label,
            state.tools_label,
            state.approvals_label,
        )
    )


class ConsoleControlBar(Vertical):
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
        self.styles.height = CONSOLE_CONTROL_BAR_HEIGHT
        self.styles.min_height = CONSOLE_CONTROL_BAR_HEIGHT
        self.styles.max_height = CONSOLE_CONTROL_BAR_HEIGHT

    @staticmethod
    def _compatibility_layout_widget(widget: Any) -> Any:
        """Keep pre-Workbench selectors mounted until parity tests migrate.

        Guarded by Console decomposition regressions that still query the
        legacy control-bar child IDs while visible state moves to ModeStrip.
        """
        widget.styles.display = "none"
        widget.styles.width = 0
        widget.styles.min_width = 0
        widget.styles.height = 0
        widget.styles.min_height = 0
        return widget

    def sync_state(self, state: ConsoleControlState) -> None:
        """Refresh visible label widgets from a new display-state snapshot.

        Args:
            state: Updated display-state snapshot to render.
        """
        self.state = state
        label_values = {
            "#console-control-status-line": _summary_line(state),
            "#console-provider-label": state.provider_label,
            "#console-model-label": state.model_label,
            "#console-persona-label": state.persona_label,
            "#console-rag-label": state.rag_label,
            "#console-sources-label": state.sources_label,
            "#console-tools-label": state.tools_label,
            "#console-approvals-label": state.approvals_label,
            "#console-provider-chip": state.provider_label,
            "#console-model-chip": state.model_label,
            "#console-persona-chip": state.persona_label,
            "#console-rag-chip": state.rag_label,
            "#console-sources-chip": state.sources_label,
            "#console-tools-chip": state.tools_label,
            "#console-approvals-chip": state.approvals_label,
        }
        for selector, label in label_values.items():
            try:
                self.query_one(selector, Static).update(label)
            except NoMatches:
                continue

    @staticmethod
    def _chip(label: str, *, id: str) -> Static:
        return Static(label, id=id, classes="console-control-chip")

    @staticmethod
    def _action(label: str, *, id: str, action_id: str, tooltip: str) -> Button:
        button = Button(
            label,
            id=id,
            classes="console-control-action workbench-action",
            compact=True,
            tooltip=tooltip,
        )
        setattr(button, "_workbench_action_id", action_id)
        return button

    def compose(self) -> ComposeResult:
        with Horizontal(id="console-control-chip-row", classes="console-control-chip-row"):
            yield self._chip(self.state.provider_label, id="console-provider-chip")
            yield self._chip(self.state.model_label, id="console-model-chip")
            yield self._chip(self.state.persona_label, id="console-persona-chip")
            yield self._chip(self.state.rag_label, id="console-rag-chip")
            yield self._chip(self.state.sources_label, id="console-sources-chip")
            yield self._chip(self.state.tools_label, id="console-tools-chip")
            yield self._chip(self.state.approvals_label, id="console-approvals-chip")
        with Horizontal(id="console-control-action-row", classes="console-control-action-row"):
            yield self._action(
                "Settings",
                id="console-control-settings",
                action_id="settings",
                tooltip="Configure provider, model, tools, and generation",
            )
            yield self._action(
                "Attach",
                id="console-control-attach-context",
                action_id="attach-context",
                tooltip="Stage Library or workspace context",
            )
            yield self._action(
                "Library RAG",
                id="console-control-run-library-rag",
                action_id="run-library-rag",
                tooltip="Search Library evidence before sending",
            )
            yield self._action(
                "Help",
                id="console-control-help",
                action_id="help",
                tooltip="Show visible Console actions and shortcuts",
            )
        yield Static(
            _summary_line(self.state),
            id="console-control-status-line",
            classes="console-control-summary-line",
        )
        yield self._compatibility_layout_widget(Static(
            self.state.provider_label,
            id="console-provider-label",
            classes="console-control-label console-hidden-control",
        ))
        yield self._compatibility_layout_widget(Static(
            self.state.model_label,
            id="console-model-label",
            classes="console-control-label console-hidden-control",
        ))
        yield self._compatibility_layout_widget(Static(
            self.state.persona_label,
            id="console-persona-label",
            classes="console-control-label console-hidden-control",
        ))
        yield self._compatibility_layout_widget(Static(
            self.state.rag_label,
            id="console-rag-label",
            classes="console-control-label console-hidden-control",
        ))
        yield self._compatibility_layout_widget(Static(
            self.state.sources_label,
            id="console-sources-label",
            classes="console-control-label console-hidden-control",
        ))
        yield self._compatibility_layout_widget(Static(
            self.state.tools_label,
            id="console-tools-label",
            classes="console-control-label console-hidden-control",
        ))
        yield self._compatibility_layout_widget(Static(
            self.state.approvals_label,
            id="console-approvals-label",
            classes="console-control-label console-hidden-control",
        ))
        yield self._compatibility_layout_widget(CompactModelBar(
            self.app_instance,
            on_sidebar_toggle_requested=self.on_sidebar_toggle_requested,
            id="console-compact-model-bar",
            classes="console-compact-model-bar console-hidden-control",
        ))

    @on(Button.Pressed, ".console-control-action")
    def on_console_control_action_pressed(self, event: Button.Pressed) -> None:
        """Route compact visible control actions through the Workbench seam."""
        action_id = getattr(event.button, "_workbench_action_id", "")
        if not action_id:
            return
        event.stop()
        self.post_message(WorkbenchActionRequested(action_id))
