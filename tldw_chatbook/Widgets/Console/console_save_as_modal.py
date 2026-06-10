"""Console Save as destination modal."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_message_actions import ConsoleSaveDestination


class ConsoleSaveAsModal(ModalScreen[str | None]):
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

    .console-save-as-destination,
    .console-save-as-wip,
    .console-save-as-empty-state {
        height: auto;
        margin: 0 0 1 0;
    }

    Button.console-save-as-destination {
        width: 100%;
        height: 3;
    }
    """

    BINDINGS = [("escape", "dismiss", "Close")]

    def __init__(self, *, destinations: list[ConsoleSaveDestination]) -> None:
        super().__init__()
        self.destinations = destinations

    def compose(self) -> ComposeResult:
        with Vertical(id="console-save-as-modal"):
            yield Static("Save as...", classes="console-transcript-action-row")
            if not any(destination.available for destination in self.destinations):
                yield Static(
                    "No Save as destinations are wired for selected messages yet.",
                    classes="console-save-as-empty-state",
                )
            for destination in self.destinations:
                destination_id = _destination_id(destination.label)
                if destination.available:
                    yield Button(
                        destination.label,
                        id=destination_id,
                        classes="console-save-as-destination",
                    )
                    continue
                reason = f"\n{destination.reason}" if destination.reason else ""
                yield Static(
                    f"{destination.label} [WIP]{reason}",
                    id=destination_id.replace("destination", "wip", 1),
                    classes="console-save-as-wip",
                )
            yield Button("Close", id="console-save-as-close")

    def action_dismiss(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "console-save-as-close":
            event.stop()
            self.dismiss(None)
            return
        for destination in self.destinations:
            if destination.available and event.button.id == _destination_id(destination.label):
                event.stop()
                self.dismiss(destination.label)
                return


def _destination_id(label: str) -> str:
    safe_label = "".join(character.lower() if character.isalnum() else "-" for character in label)
    safe_label = "-".join(part for part in safe_label.split("-") if part)
    return f"console-save-as-destination-{safe_label}"
