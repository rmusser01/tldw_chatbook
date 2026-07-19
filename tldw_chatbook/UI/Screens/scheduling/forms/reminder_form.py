"""Reminder create modal form."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Static, TextArea

from tldw_chatbook.Scheduling.events import ReminderFormSubmitted
from tldw_chatbook.Scheduling.models import ScheduleKind


# TODO: support edit mode by accepting an optional ReminderTask
class ReminderForm(ModalScreen):
    """Modal form for creating a reminder."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=False),
    ]

    DEFAULT_CSS = """
    ReminderForm {
        align: center middle;
    }

    ReminderForm > Container {
        width: 80;
        height: auto;
        max-height: 45;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    .form-title {
        text-style: bold;
        text-align: center;
        padding: 1 0;
    }

    .form-label {
        color: $text-muted;
        padding: 1 0 0 0;
    }

    .error-text {
        color: $error;
        text-style: bold;
        height: auto;
        padding: 1 0;
    }

    .button-container {
        align: center middle;
        height: auto;
        padding: 1 0;
    }

    .button-container Button {
        margin: 0 1;
    }
    """

    def action_dismiss(self) -> None:
        """Dismiss the modal when the Escape key is pressed."""
        self.dismiss(None)

    def compose(self) -> ComposeResult:
        """Build the form layout."""
        with Container():
            yield Label("Create Reminder", classes="form-title")

            with Vertical():
                yield Label("Title:", classes="form-label")
                yield Input(placeholder="Enter reminder title...", id="reminder-title")

                yield Label("Body:", classes="form-label")
                yield TextArea(id="reminder-body")

                yield Label("Schedule Kind:", classes="form-label")
                yield Select(
                    self._schedule_options(),
                    allow_blank=False,
                    value=ScheduleKind.ONE_TIME.value,
                    id="reminder-kind",
                )

                yield Static("", id="reminder-errors", classes="error-text")

            with Horizontal(classes="button-container"):
                yield Button("Save", variant="success", id="reminder-save")
                yield Button("Cancel", id="reminder-cancel")

    @staticmethod
    def _schedule_options() -> list[tuple[str, str]]:
        """Return labelled options for the schedule kind selector."""
        return [
            (kind.value.replace("_", " ").title(), kind.value)
            for kind in ScheduleKind
        ]

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle save/cancel button presses."""
        if event.button.id == "reminder-cancel":
            self.dismiss(None)
        elif event.button.id == "reminder-save":
            self._save()

    def _save(self) -> None:
        """Validate the form and emit the submitted event on success."""
        error_widget = self.query_one("#reminder-errors", Static)
        title = self.query_one("#reminder-title", Input).value.strip()

        if not title:
            error_widget.update("Title is required")
            return

        body = self.query_one("#reminder-body", TextArea).text.strip()
        schedule_kind = self.query_one("#reminder-kind", Select).value

        error_widget.update("")

        form_data: dict[str, Any] = {
            "title": title,
            "body": body,
            "schedule_kind": schedule_kind,
        }

        self.post_message(ReminderFormSubmitted(form_data))
        self.dismiss(form_data)
