"""Console transcript message edit modal."""

from __future__ import annotations

from dataclasses import dataclass

from textual import events, on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static, TextArea


@dataclass(frozen=True)
class ConsoleEditResult:
    """Outcome of the edit modal: the (possibly unchanged) text, and whether
    the caller asked to fork a new branch (Console branching Phase B) rather
    than edit the message in place."""

    text: str
    resend: bool


class _EditMessageTextArea(TextArea):
    """TextArea that ignores keys typed before the modal appeared.

    TASK-360: the edit action dispatches through a Button.Pressed hop and an
    async modal push; keys pressed in that gap (e.g. a retry `e` because
    nothing visibly happened) used to land here as text and silently corrupt
    the draft. A key whose event time predates the modal's mount was aimed
    at whatever the user was looking at then — never at this textarea.
    """

    opened_at: float | None = None

    async def _on_key(self, event: events.Key) -> None:
        if self.opened_at is not None and event.time < self.opened_at:
            event.stop()
            event.prevent_default()
            return
        await super()._on_key(event)


class ConsoleEditMessageModal(ModalScreen[ConsoleEditResult | None]):
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
    #console-edit-message-save,
    #console-edit-message-resend {
        width: 10;
        min-width: 10;
        height: 3;
        min-height: 3;
    }

    #console-edit-message-resend {
        width: 18;
        min-width: 18;
    }
    """

    BINDINGS = [("escape", "dismiss", "Cancel")]

    def __init__(self, *, content: str, can_resend: bool = False) -> None:
        super().__init__()
        self._content = content
        self._can_resend = can_resend

    def compose(self) -> ComposeResult:
        with Vertical(id="console-edit-message-modal"):
            yield Static("Edit Message", classes="console-modal-header")
            if self._can_resend:
                context_copy = (
                    "Editing existing transcript message. Save keeps the edit in "
                    "place; Edit & resend forks a new branch and gets a fresh reply."
                )
            else:
                context_copy = (
                    "Editing existing transcript message. This will not create a new prompt."
                )
            yield Static(
                context_copy,
                id="console-edit-message-context",
                markup=False,
            )
            yield _EditMessageTextArea(self._content, id="console-edit-message-body")
            yield Static("", id="console-edit-message-error", markup=False)
            with Horizontal(id="console-edit-message-actions"):
                yield Button("Cancel", id="console-edit-message-cancel")
                yield Button(
                    "Save",
                    id="console-edit-message-save",
                    variant="default" if self._can_resend else "primary",
                )
                if self._can_resend:
                    yield Button(
                        "Edit & resend",
                        id="console-edit-message-resend",
                        variant="primary",
                    )

    def on_mount(self, event: events.Mount) -> None:
        # Event time shares the clock domain of Key.time — the stale-key
        # guard compares against it (TASK-360).
        self._opened_at = event.time
        area = self.query_one("#console-edit-message-body", _EditMessageTextArea)
        area.opened_at = event.time
        area.focus()

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
        self.dismiss(ConsoleEditResult(text=edited_content, resend=False))

    @on(Button.Pressed, "#console-edit-message-resend")
    def _resend(self, event: Button.Pressed) -> None:
        event.stop()
        edited_content = self.query_one("#console-edit-message-body", TextArea).text
        if not edited_content.strip():
            self.query_one("#console-edit-message-error", Static).update(
                "Message content cannot be blank."
            )
            return
        self.dismiss(ConsoleEditResult(text=edited_content, resend=True))
