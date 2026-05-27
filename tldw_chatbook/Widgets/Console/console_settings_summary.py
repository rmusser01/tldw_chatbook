"""Console-native settings summary widget."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_session_settings import ConsoleSettingsSummaryState


class ConsoleSettingsSummary(Vertical):
    """Render compact Console session settings rows."""

    def __init__(self, state: ConsoleSettingsSummaryState, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.state = state
        self.add_class("console-settings-summary")
        self.styles.height = "auto"
        self.styles.min_height = 0
        self.styles.max_height = 6

    def sync_state(self, state: ConsoleSettingsSummaryState) -> None:
        """Refresh the summary from a new state snapshot."""
        self.state = state
        try:
            self.query_one("#console-settings-provider-row", Static).update(state.provider_row)
            self.query_one("#console-settings-model-row", Static).update(state.model_row)
            self.query_one("#console-settings-context-row", Static).update(state.context_row)
            self.query_one("#console-settings-sampling-row", Static).update(state.sampling_row)
            self.query_one("#console-settings-identity-row", Static).update(state.identity_row)
            button = self.query_one("#console-settings-open", Button)
        except NoMatches:
            self.refresh(recompose=True)
            return

        button.label = state.action_label
        button.tooltip = state.action_tooltip
        self._apply_button_sizing(button)

    def _apply_button_sizing(self, button: Button) -> None:
        button_width = min(max(len(self.state.action_label) + 2, 9), 14)
        button.styles.width = button_width
        button.styles.min_width = button_width
        button.styles.max_width = button_width
        button.styles.height = 1
        button.styles.min_height = 1
        button.styles.max_height = 1
        button.styles.margin = 0

    def compose(self) -> ComposeResult:
        header = Horizontal(id="console-settings-header", classes="console-settings-header")
        header.styles.height = 1
        header.styles.min_height = 1
        header.styles.max_height = 1
        with header:
            title = Static(
                "Session Settings",
                id="console-settings-title",
                classes="destination-section console-settings-title",
            )
            title.styles.width = "1fr"
            title.styles.min_width = 0
            title.styles.height = 1
            title.styles.min_height = 1
            title.styles.max_height = 1
            yield title

            button = Button(
                self.state.action_label,
                id="console-settings-open",
                tooltip=self.state.action_tooltip,
                compact=True,
            )
            self._apply_button_sizing(button)
            yield button
        yield Static(
            self.state.provider_row,
            id="console-settings-provider-row",
            classes="console-settings-row",
            markup=False,
        )
        yield Static(
            self.state.model_row,
            id="console-settings-model-row",
            classes="console-settings-row",
            markup=False,
        )
        yield Static(
            self.state.context_row,
            id="console-settings-context-row",
            classes="console-settings-row",
            markup=False,
        )
        yield Static(
            self.state.sampling_row,
            id="console-settings-sampling-row",
            classes="console-settings-row",
            markup=False,
        )
        yield Static(
            self.state.identity_row,
            id="console-settings-identity-row",
            classes="console-settings-row",
            markup=False,
        )
