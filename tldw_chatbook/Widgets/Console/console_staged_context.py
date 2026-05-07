"""Console-native staged context tray."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from tldw_chatbook.Chat.console_display_state import ConsoleStagedContextState


class ConsoleStagedContextTray(Vertical):
    """Render staged handoff/live-work provenance in the Console shell."""

    def __init__(self, state: ConsoleStagedContextState, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.state = state

    def compose(self) -> ComposeResult:
        yield Static(
            self.state.heading,
            id="console-staged-context-title",
            classes="destination-section",
        )
        yield Static(
            self.state.summary,
            id="console-staged-context-summary",
            classes="console-staged-context-summary",
        )
        for index, row in enumerate(self.state.rows):
            yield Static(
                row.text,
                id=f"console-staged-context-row-{index}",
                classes="console-staged-context-row",
            )
        if self.state.recovery:
            yield Static(
                self.state.recovery,
                id="console-staged-context-recovery",
                classes="console-staged-context-recovery",
            )
