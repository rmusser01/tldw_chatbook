"""Console-native staged context tray."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_display_state import ConsoleStagedContextState


class ConsoleStagedContextTray(Vertical):
    """Render staged handoff/live-work provenance in the Console shell.

    The tray shows the current staged-context heading, summary, structured
    provenance rows, and recovery guidance supplied by the pure Console
    display-state contract.
    """

    def __init__(self, state: ConsoleStagedContextState, **kwargs: Any) -> None:
        """Initialize the staged-context tray.

        Args:
            state: Staged-context display-state snapshot to render.
            **kwargs: Additional Textual widget arguments.
        """
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
                markup=False,
            )
        if self.state.is_empty:
            yield Button(
                "Attach",
                id="console-staged-context-attach",
                classes="console-staged-context-attach",
                compact=True,
            )
        if self.state.recovery:
            yield Static(
                self.state.recovery,
                id="console-staged-context-recovery",
                classes="console-staged-context-recovery",
            )

    def sync_state(self, state: ConsoleStagedContextState) -> None:
        """Refresh the mounted tray from a new staged-context snapshot.

        Equality-guarded like the other Console tray widgets; a real change
        recomposes only this widget (row count, Attach button, and recovery
        line presence all vary with the state), never the owning screen.

        Args:
            state: Staged-context display-state snapshot to render.
        """
        if state == self.state:
            return
        self.state = state
        self.refresh(recompose=True)
