"""Console chat tab rename modal."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static


class ConsoleRenameSessionModal(ModalScreen[str | None]):
    """Edit the display title for the active Console chat tab."""

    DEFAULT_CSS = """
    ConsoleRenameSessionModal {
        align: center middle;
    }

    #console-rename-session-modal {
        width: 56;
        height: auto;
        border: tall gray;
        background: black;
        padding: 1 2;
    }

    #console-rename-session-title {
        width: 100%;
        margin: 1 0 0 0;
    }

    #console-rename-session-error {
        height: auto;
        min-height: 1;
        color: red;
    }

    #console-rename-session-actions {
        height: 3;
        min-height: 3;
        margin: 1 0 0 0;
        align-horizontal: right;
    }

    #console-rename-session-cancel,
    #console-rename-session-save {
        width: 10;
        min-width: 10;
        height: 3;
        min-height: 3;
    }
    """

    BINDINGS = [("escape", "dismiss", "Cancel")]

    def __init__(self, *, title: str) -> None:
        super().__init__()
        self._title = title

    def compose(self) -> ComposeResult:
        with Vertical(id="console-rename-session-modal"):
            yield Static("Rename Chat Tab", classes="console-transcript-action-row")
            yield Input(
                value=self._title,
                id="console-rename-session-title",
                placeholder="Chat tab name",
            )
            yield Static("", id="console-rename-session-error", markup=False)
            with Horizontal(id="console-rename-session-actions"):
                yield Button("Cancel", id="console-rename-session-cancel")
                yield Button(
                    "Save",
                    id="console-rename-session-save",
                    variant="primary",
                )

    def on_mount(self) -> None:
        rename_input = self.query_one("#console-rename-session-title", Input)
        rename_input.focus()
        rename_input.select_all()

    def action_dismiss(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#console-rename-session-cancel")
    def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)

    @on(Button.Pressed, "#console-rename-session-save")
    def _save_button(self, event: Button.Pressed) -> None:
        event.stop()
        self._save()

    @on(Input.Submitted, "#console-rename-session-title")
    def _save_input(self, event: Input.Submitted) -> None:
        event.stop()
        self._save()

    def _save(self) -> None:
        title = self.query_one("#console-rename-session-title", Input).value.strip()
        if not title:
            self.query_one("#console-rename-session-error", Static).update(
                "Name cannot be blank."
            )
            return
        self.dismiss(title)
