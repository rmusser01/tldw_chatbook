"""Tiny choice between a reminder and a recurring-question automation.

Smallest existing house pattern (task-5, ADR-099 idiom parity): a
two-button confirm modal using `SafeModalDismissMixin`, same shape as
`ManagedGGUFRuntimeChoiceModal` -- no dirty-state to guard, so the mixin's
one-shot dismiss/backdrop handling is all this needs (unlike the create/
edit forms, which have their own bespoke discard guard for real field
state).
"""

from __future__ import annotations

from typing import Literal

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin

NewTaskChoice = Literal["reminder", "recurring_question"]


class NewTaskChoiceModal(SafeModalDismissMixin, ModalScreen[NewTaskChoice | None]):
    """Ask which kind of scheduled task to create."""

    BUNDLED_CSS = """
    NewTaskChoiceModal {
        align: center middle;
    }

    NewTaskChoiceModal .new-task-choice-modal {
        width: 64;
        height: auto;
        border: tall $accent;
        background: $surface;
        padding: 1 2;
    }

    NewTaskChoiceModal .new-task-choice-title {
        text-style: bold;
        margin-bottom: 1;
    }

    NewTaskChoiceModal .new-task-choice-copy {
        height: auto;
        color: $text-muted;
        margin-bottom: 1;
    }

    NewTaskChoiceModal .new-task-choice-actions {
        height: 3;
        align-horizontal: right;
    }

    NewTaskChoiceModal .new-task-choice-actions Button {
        width: auto;
        margin-left: 1;
    }
    """

    BINDINGS = [("escape", "request_safe_cancel", "Cancel")]
    SAFE_MODAL_CONTENT = ".new-task-choice-modal"

    def compose(self) -> ComposeResult:
        with Vertical(classes="new-task-choice-modal"):
            yield Static("New scheduled task", classes="new-task-choice-title", markup=False)
            yield Static(
                "A reminder fires once or on a cron schedule. A recurring "
                "question runs a scoped search on a schedule and reports "
                "what it finds.",
                classes="new-task-choice-copy",
                markup=False,
            )
            with Horizontal(classes="new-task-choice-actions"):
                yield Button("Cancel", id="new-task-choice-cancel")
                yield Button(
                    "Recurring question…", id="new-task-choice-automation"
                )
                yield Button(
                    "Reminder…", id="new-task-choice-reminder", variant="primary"
                )

    def on_mount(self) -> None:
        super().on_mount()
        self.query_one("#new-task-choice-reminder", Button).focus()

    @on(Button.Pressed)
    async def _button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "new-task-choice-reminder":
            self.dismiss_safe_once("reminder")
        elif event.button.id == "new-task-choice-automation":
            self.dismiss_safe_once("recurring_question")
        elif event.button.id == "new-task-choice-cancel":
            await self.request_safe_cancel(source="visible")
