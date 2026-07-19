"""Console Save as destination modal."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_message_actions import ConsoleSaveDestination


class ConsoleSaveAsModal(ModalScreen[str | None]):
    """List available and unavailable Save as destinations for a selected message."""

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
    .console-save-as-unavailable,
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

    def __init__(
        self,
        *,
        destinations: list[ConsoleSaveDestination],
        message_role: str = "Message",
        message_excerpt: str = "",
    ) -> None:
        super().__init__()
        self.destinations = destinations
        self.message_role = message_role.strip() or "Message"
        self.message_excerpt = message_excerpt.strip()

    def compose(self) -> ComposeResult:
        with Vertical(id="console-save-as-modal"):
            yield Static("Save as...", classes="console-modal-header")
            yield Static(
                f"Saving selected {self.message_role} message",
                id="console-save-as-context",
                classes="console-save-as-context",
                markup=False,
            )
            if self.message_excerpt:
                yield Static(
                    f"Excerpt: {self.message_excerpt}",
                    id="console-save-as-excerpt",
                    classes="console-save-as-context",
                    markup=False,
                )
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
                    f"{destination.label} (unavailable){reason}",
                    id=_destination_id(destination.label, prefix="unavailable"),
                    classes="console-save-as-unavailable",
                    markup=False,
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


def _destination_id(label: str, prefix: str = "destination") -> str:
    safe_label = "".join(character.lower() if character.isalnum() else "-" for character in label)
    safe_label = "-".join(part for part in safe_label.split("-") if part)
    return f"console-save-as-{prefix}-{safe_label}"
