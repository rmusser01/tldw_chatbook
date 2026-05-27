"""Console-native settings summary widget."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_session_settings import ConsoleSettingsSummaryState


class ConsoleSettingsSummary(Vertical):
    """Render compact Console session settings rows."""

    def __init__(self, state: ConsoleSettingsSummaryState, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.state = state
        self.add_class("console-settings-summary")

    def sync_state(self, state: ConsoleSettingsSummaryState) -> None:
        """Refresh the summary from a new state snapshot."""
        self.state = state
        self.refresh(recompose=True)

    def compose(self) -> ComposeResult:
        yield Static(
            "Console Settings",
            id="console-settings-title",
            classes="destination-section",
        )
        if self.state.provider_row:
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
        yield Button(
            self.state.action_label,
            id="console-settings-open",
            tooltip=self.state.action_tooltip,
            compact=True,
        )
