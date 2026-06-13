"""Console-native settings summary widget."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_session_settings import ConsoleSettingsSummaryState


CONSOLE_SETTINGS_SUMMARY_MAX_HEIGHT = 9
CONSOLE_SETTINGS_BUTTON_HORIZONTAL_PADDING = 2
CONSOLE_SETTINGS_BUTTON_MIN_WIDTH = 9
CONSOLE_SETTINGS_BUTTON_MAX_WIDTH = 14
CONSOLE_SETTINGS_ROW_HEIGHT = 1


class ConsoleSettingsSummary(Vertical):
    """Render compact Console session settings rows."""

    def __init__(self, state: ConsoleSettingsSummaryState, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.state = state
        self.add_class("console-settings-summary")
        self.styles.height = "auto"
        self.styles.min_height = 0
        self.styles.max_height = CONSOLE_SETTINGS_SUMMARY_MAX_HEIGHT

    def sync_state(self, state: ConsoleSettingsSummaryState) -> None:
        """Refresh the summary from a new state snapshot."""
        self.state = state
        try:
            self.query_one("#console-settings-provider-row", Static).update(
                self._row_text(state.provider_row)
            )
            self.query_one("#console-settings-model-row", Static).update(state.model_row)
            self.query_one("#console-settings-context-row", Static).update(state.context_row)
            self.query_one("#console-settings-endpoint-row", Static).update(state.endpoint_row)
            self.query_one("#console-settings-credential-row", Static).update(state.credential_row)
            self.query_one("#console-settings-transport-row", Static).update(state.transport_row)
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
        button_width = min(
            max(
                len(self.state.action_label)
                + CONSOLE_SETTINGS_BUTTON_HORIZONTAL_PADDING,
                CONSOLE_SETTINGS_BUTTON_MIN_WIDTH,
            ),
            CONSOLE_SETTINGS_BUTTON_MAX_WIDTH,
        )
        button.styles.width = button_width
        button.styles.min_width = button_width
        button.styles.max_width = button_width
        button.styles.height = CONSOLE_SETTINGS_ROW_HEIGHT
        button.styles.min_height = CONSOLE_SETTINGS_ROW_HEIGHT
        button.styles.max_height = CONSOLE_SETTINGS_ROW_HEIGHT
        button.styles.margin = 0

    @staticmethod
    def _row_text(value: str | None) -> str:
        """Return a Textual-safe settings row label."""
        return value or ""

    def compose(self) -> ComposeResult:
        header = Horizontal(id="console-settings-header", classes="console-settings-header")
        header.styles.height = CONSOLE_SETTINGS_ROW_HEIGHT
        header.styles.min_height = CONSOLE_SETTINGS_ROW_HEIGHT
        header.styles.max_height = CONSOLE_SETTINGS_ROW_HEIGHT
        with header:
            title = Static(
                "Session Settings",
                id="console-settings-title",
                classes="destination-section console-settings-title",
            )
            title.styles.width = "1fr"
            title.styles.min_width = 0
            title.styles.height = CONSOLE_SETTINGS_ROW_HEIGHT
            title.styles.min_height = CONSOLE_SETTINGS_ROW_HEIGHT
            title.styles.max_height = CONSOLE_SETTINGS_ROW_HEIGHT
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
            self._row_text(self.state.provider_row),
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
            self.state.endpoint_row,
            id="console-settings-endpoint-row",
            classes="console-settings-row",
            markup=False,
        )
        yield Static(
            self.state.credential_row,
            id="console-settings-credential-row",
            classes="console-settings-row",
            markup=False,
        )
        yield Static(
            self.state.transport_row,
            id="console-settings-transport-row",
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
