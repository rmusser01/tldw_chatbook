"""Console transcript message edit modal."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static, TextArea


class ConsoleEditMessageModal(ModalScreen[str | None]):
    """Edit an existing Console transcript message without using the composer."""

    DEFAULT_CSS = """
    ConsoleEditMessageModal {
        align: center middle;
    }

    #console-edit-message-modal {
        width: 92;
        height: 28;
        border: tall gray;
        background: black;
        padding: 1 2;
    }

    #console-edit-message-context {
        height: auto;
        margin: 1 0 1 0;
    }

    #console-edit-message-body {
        width: 100%;
        height: 16;
    }

    #console-edit-message-error {
        height: auto;
        min-height: 1;
        color: red;
    }

    #console-edit-message-actions {
        height: 3;
        min-height: 3;
        margin: 1 0 0 0;
        align-horizontal: right;
    }

    #console-edit-message-cancel,
    #console-edit-message-save {
        width: 10;
        min-width: 10;
        height: 3;
        min-height: 3;
    }
    """

    BINDINGS = [("escape", "dismiss", "Cancel")]

    def __init__(self, *, content: str) -> None:
        super().__init__()
        self._content = content

    def compose(self) -> ComposeResult:
        with Vertical(id="console-edit-message-modal"):
            yield Static("Edit Message", classes="console-modal-header")
            yield Static(
                "Editing existing transcript message. This will not create a new prompt.",
                id="console-edit-message-context",
                markup=False,
            )
            yield TextArea(self._content, id="console-edit-message-body")
            yield Static("", id="console-edit-message-error", markup=False)
            with Horizontal(id="console-edit-message-actions"):
                yield Button("Cancel", id="console-edit-message-cancel")
                yield Button("Save", id="console-edit-message-save", variant="primary")

    def on_mount(self) -> None:
        self.query_one("#console-edit-message-body", TextArea).focus()

    def action_dismiss(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#console-edit-message-cancel")
    def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)

    @on(Button.Pressed, "#console-edit-message-save")
    def _save(self, event: Button.Pressed) -> None:
        event.stop()
        edited_content = self.query_one("#console-edit-message-body", TextArea).text
        if not edited_content.strip():
            self.query_one("#console-edit-message-error", Static).update(
                "Message content cannot be blank."
            )
            return
        self.dismiss(edited_content)
