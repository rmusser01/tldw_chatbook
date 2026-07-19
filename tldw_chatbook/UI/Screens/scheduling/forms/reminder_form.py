"""Reminder create/edit modal form."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Static, TextArea

from tldw_chatbook.Scheduling.events import ReminderFormSubmitted
from tldw_chatbook.Scheduling.models import ReminderTask, ScheduleKind


_DEFAULT_TIMEZONE = "UTC"


class ReminderForm(ModalScreen):
    """Modal form for creating or editing a reminder."""

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
        max-height: 55;
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

    task: reactive[ReminderTask | None] = reactive(None)

    def __init__(self, task: ReminderTask | None = None) -> None:
        """Initialize the form.

        Args:
            task: Existing reminder to edit, or ``None`` to create a new one.
        """
        super().__init__()
        self.task = task

    def action_dismiss(self) -> None:
        """Dismiss the modal when the Escape key is pressed."""
        self.dismiss(None)

    def compose(self) -> ComposeResult:
        """Build the form layout."""
        with Container():
            yield Label(
                "Edit Reminder" if self.task else "Create Reminder",
                classes="form-title",
            )

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

                yield Label("Run At (ISO-8601):", classes="form-label")
                yield Input(
                    placeholder="2026-07-20T14:00:00+00:00",
                    id="reminder-run-at",
                )

                yield Label("Cron Expression:", classes="form-label")
                yield Input(placeholder="0 9 * * 1", id="reminder-cron")

                yield Label("Timezone:", classes="form-label")
                yield Input(
                    placeholder=_DEFAULT_TIMEZONE,
                    id="reminder-timezone",
                )

                yield Static("", id="reminder-errors", classes="error-text")

            with Horizontal(classes="button-container"):
                yield Button("Save", variant="success", id="reminder-save")
                yield Button("Cancel", id="reminder-cancel")

    def on_mount(self) -> None:
        """Prefill the form when editing an existing reminder."""
        if self.task is None:
            self.query_one("#reminder-timezone", Input).value = _DEFAULT_TIMEZONE
            return

        self.query_one("#reminder-title", Input).value = self.task.title
        body = self.task.body or ""
        self.query_one("#reminder-body", TextArea).text = body
        self.query_one("#reminder-kind", Select).value = self.task.schedule_kind.value
        if self.task.run_at is not None:
            self.query_one("#reminder-run-at", Input).value = self.task.run_at.isoformat()
        if self.task.cron is not None:
            self.query_one("#reminder-cron", Input).value = self.task.cron
        if self.task.timezone is not None:
            self.query_one("#reminder-timezone", Input).value = self.task.timezone
        self._update_schedule_field_visibility(self.task.schedule_kind.value)

    def on_select_changed(self, event: Select.Changed) -> None:
        """Show/hide schedule fields based on the selected schedule kind."""
        if event.select.id == "reminder-kind":
            self._update_schedule_field_visibility(str(event.value))

    def _update_schedule_field_visibility(self, kind: str) -> None:
        """Toggle which schedule inputs are visible."""
        run_at_label = self.query_one("#reminder-run-at").parent
        cron_label = self.query_one("#reminder-cron").parent
        tz_label = self.query_one("#reminder-timezone").parent
        # Labels are siblings before the input in the same Vertical; toggling the
        # input's parent Horizontal would hide label+input if they were grouped.
        # Since they are not grouped, we toggle the input widget directly and
        # leave the label visible for clarity.
        run_at_input = self.query_one("#reminder-run-at", Input)
        cron_input = self.query_one("#reminder-cron", Input)
        tz_input = self.query_one("#reminder-timezone", Input)
        if kind == ScheduleKind.ONE_TIME.value:
            run_at_input.display = True
            cron_input.display = False
            tz_input.display = False
        else:
            run_at_input.display = False
            cron_input.display = True
            tz_input.display = True
        # Avoid unused variable warnings for label containers.
        _ = (run_at_label, cron_label, tz_label)

    @staticmethod
    def _schedule_options() -> list[tuple[str, str]]:
        """Return labelled options for the schedule kind selector."""
        return [
            (kind.value.replace("_", " ").title(), kind.value) for kind in ScheduleKind
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
        body = self.query_one("#reminder-body", TextArea).text.strip()
        schedule_kind = str(self.query_one("#reminder-kind", Select).value)
        run_at = self.query_one("#reminder-run-at", Input).value.strip()
        cron = self.query_one("#reminder-cron", Input).value.strip()
        timezone = self.query_one("#reminder-timezone", Input).value.strip()

        errors: list[str] = []
        if not title:
            errors.append("Title is required")

        if schedule_kind == ScheduleKind.ONE_TIME.value:
            if not run_at:
                errors.append("Run At is required for one-time reminders")
        elif schedule_kind == ScheduleKind.RECURRING.value:
            if not cron:
                errors.append("Cron expression is required for recurring reminders")
            if not timezone:
                errors.append("Timezone is required for recurring reminders")

        if errors:
            error_widget.update("\n".join(errors))
            return

        error_widget.update("")

        form_data: dict[str, Any] = {
            "title": title,
            "body": body,
            "schedule_kind": schedule_kind,
        }
        if schedule_kind == ScheduleKind.ONE_TIME.value:
            form_data["run_at"] = run_at
        else:
            form_data["cron"] = cron
            form_data["timezone"] = timezone

        if self.task is not None:
            # Preserve the current enabled state when editing.
            form_data["enabled"] = self.task.enabled

        self.post_message(ReminderFormSubmitted(form_data, task_id=self.task.id if self.task else None))
        self.dismiss(form_data)
