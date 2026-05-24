"""Console Save as destination modal."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_message_actions import ConsoleSaveDestination


class ConsoleSaveAsModal(ModalScreen[None]):
    """List available and WIP Save as destinations for a selected message."""

    DEFAULT_CSS = """
    ConsoleSaveAsModal {
        align: center middle;
    }

    #console-save-as-modal {
        width: 72;
        height: auto;
        border: tall gray;
        background: black;
        padding: 1 2;
    }

    .console-save-as-destination {
        height: auto;
        margin: 0 0 1 0;
    }
    """

    BINDINGS = [("escape", "dismiss", "Close")]

    def __init__(self, *, destinations: list[ConsoleSaveDestination]) -> None:
        super().__init__()
        self.destinations = destinations

    def compose(self) -> ComposeResult:
        with Vertical(id="console-save-as-modal"):
            yield Static("Save as...", classes="console-transcript-action-row")
            for destination in self.destinations:
                state = "available" if destination.available else "WIP"
                reason = f"\n{destination.reason}" if destination.reason else ""
                yield Static(
                    f"{destination.label} [{state}]{reason}",
                    classes="console-save-as-destination",
                )
            yield Button("Close", id="console-save-as-close")

    def action_dismiss(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "console-save-as-close":
            event.stop()
            self.dismiss(None)
