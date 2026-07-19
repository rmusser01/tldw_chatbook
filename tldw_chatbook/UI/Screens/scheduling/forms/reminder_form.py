"""Reminder create/edit modal form."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from croniter import croniter
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
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

    def __init__(self, task: ReminderTask | None = None) -> None:
        """Initialize the form.

        Args:
            task: Existing reminder to edit, or ``None`` to create a new one.
        """
        super().__init__()
        self._reminder_task = task

    def action_dismiss(self) -> None:
        """Dismiss the modal when the Escape key is pressed."""
        self.dismiss(None)

    def compose(self) -> ComposeResult:
        """Build the form layout."""
        with Container():
            yield Label(
                "Edit Reminder" if self._reminder_task else "Create Reminder",
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

                with Vertical(id="reminder-run-at-group"):
                    yield Label("Run At (ISO-8601):", classes="form-label")
                    yield Input(
                        placeholder="2026-07-20T14:00:00+00:00",
                        id="reminder-run-at",
                    )

                with Vertical(id="reminder-cron-group"):
                    yield Label("Cron Expression:", classes="form-label")
                    yield Input(placeholder="0 9 * * 1", id="reminder-cron")

                with Vertical(id="reminder-timezone-group"):
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
        if self._reminder_task is None:
            self.query_one("#reminder-timezone", Input).value = _DEFAULT_TIMEZONE
            self._update_schedule_field_visibility(ScheduleKind.ONE_TIME.value)
            return

        self.query_one("#reminder-title", Input).value = self._reminder_task.title
        body = self._reminder_task.body or ""
        self.query_one("#reminder-body", TextArea).text = body
        self.query_one("#reminder-kind", Select).value = self._reminder_task.schedule_kind.value
        if self._reminder_task.run_at is not None:
            self.query_one("#reminder-run-at", Input).value = self._reminder_task.run_at.isoformat()
        if self._reminder_task.cron is not None:
            self.query_one("#reminder-cron", Input).value = self._reminder_task.cron
        if self._reminder_task.timezone is not None:
            self.query_one("#reminder-timezone", Input).value = self._reminder_task.timezone
        else:
            self.query_one("#reminder-timezone", Input).value = _DEFAULT_TIMEZONE
        self._update_schedule_field_visibility(self._reminder_task.schedule_kind.value)

    def on_select_changed(self, event: Select.Changed) -> None:
        """Show/hide schedule field groups based on the selected schedule kind."""
        if event.select.id == "reminder-kind":
            self._update_schedule_field_visibility(str(event.value))

    def _update_schedule_field_visibility(self, kind: str) -> None:
        """Toggle which schedule input groups are visible."""
        run_at_group = self.query_one("#reminder-run-at-group", Vertical)
        cron_group = self.query_one("#reminder-cron-group", Vertical)
        tz_group = self.query_one("#reminder-timezone-group", Vertical)
        if kind == ScheduleKind.ONE_TIME.value:
            run_at_group.display = True
            cron_group.display = False
            tz_group.display = False
        else:
            run_at_group.display = False
            cron_group.display = True
            tz_group.display = True

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

        parsed_run_at: datetime | None = None
        if schedule_kind == ScheduleKind.ONE_TIME.value:
            if not run_at:
                errors.append("Run At is required for one-time reminders")
            else:
                try:
                    parsed_run_at = datetime.fromisoformat(run_at)
                except ValueError:
                    errors.append("Run At must be a valid ISO-8601 datetime")
        elif schedule_kind == ScheduleKind.RECURRING.value:
            if not cron:
                errors.append("Cron expression is required for recurring reminders")
            elif not croniter.is_valid(cron):
                errors.append("Cron expression is invalid")

            if not timezone:
                errors.append("Timezone is required for recurring reminders")
            else:
                try:
                    ZoneInfo(timezone)
                except ZoneInfoNotFoundError:
                    errors.append(f"Unknown timezone: {timezone}")

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
            form_data["run_at"] = parsed_run_at
            form_data["cron"] = None
            form_data["timezone"] = None
        else:
            form_data["run_at"] = None
            form_data["cron"] = cron
            form_data["timezone"] = timezone

        if self._reminder_task is not None:
            # Preserve the current enabled state when editing.
            form_data["enabled"] = self._reminder_task.enabled

        self.post_message(ReminderFormSubmitted(form_data, task_id=self._reminder_task.id if self._reminder_task else None))
        self.dismiss(form_data)
