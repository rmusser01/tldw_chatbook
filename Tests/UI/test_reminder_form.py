"""Tests for the reminder create/edit form."""

from datetime import datetime, timezone

import pytest
from textual.app import App
from textual.widgets import Input, Select

from tldw_chatbook.Scheduling.events import ReminderFormSubmitted
from tldw_chatbook.Scheduling.models import ReminderTask, ScheduleKind
from tldw_chatbook.UI.Screens.scheduling.forms.reminder_form import ReminderForm


class FormTestApp(App):
    """Minimal app used to host the modal form under test."""

    def __init__(self) -> None:
        super().__init__()
        self.submitted: dict | None = None

    def on_reminder_form_submitted(self, event: ReminderFormSubmitted) -> None:
        self.submitted = event.form_data


@pytest.mark.asyncio
async def test_reminder_form_requires_title():
    """Clicking save with an empty title surfaces a validation error."""
    app = FormTestApp()
    async with app.run_test() as pilot:
        await app.push_screen(ReminderForm())
        await pilot.click("#reminder-save")
        error_widget = pilot.app.screen.query_one("#reminder-errors")
        # Textual 8.2.8 does not expose Static.renderable; use visual.plain instead
        assert "title is required" in error_widget.visual.plain.lower()


@pytest.mark.asyncio
async def test_reminder_form_submits_when_valid_one_time():
    """A valid one-time reminder dismisses the modal after posting the event."""
    app = FormTestApp()
    async with app.run_test() as pilot:
        await app.push_screen(ReminderForm())
        title_input = pilot.app.screen.query_one("#reminder-title", Input)
        title_input.value = "Water plants"
        run_at_input = pilot.app.screen.query_one("#reminder-run-at", Input)
        run_at_input.value = "2026-07-20T14:00:00+00:00"
        await pilot.click("#reminder-save")
        await pilot.pause()

        assert not isinstance(pilot.app.screen, ReminderForm)
        assert app.submitted is not None
        assert app.submitted["title"] == "Water plants"
        assert app.submitted["schedule_kind"] == "one_time"
        assert app.submitted["run_at"] == datetime(2026, 7, 20, 14, 0, tzinfo=timezone.utc)
        assert app.submitted["cron"] is None
        assert app.submitted["timezone"] is None


@pytest.mark.asyncio
async def test_reminder_form_submits_when_valid_recurring():
    """A valid recurring reminder dismisses the modal after posting the event."""
    app = FormTestApp()
    async with app.run_test() as pilot:
        await app.push_screen(ReminderForm())
        title_input = pilot.app.screen.query_one("#reminder-title", Input)
        title_input.value = "Weekly sync"
        kind_select = pilot.app.screen.query_one("#reminder-kind", Select)
        kind_select.value = ScheduleKind.RECURRING.value
        await pilot.pause()

        cron_input = pilot.app.screen.query_one("#reminder-cron", Input)
        cron_input.value = "0 9 * * 1"
        tz_input = pilot.app.screen.query_one("#reminder-timezone", Input)
        tz_input.value = "UTC"

        await pilot.click("#reminder-save")
        await pilot.pause()

        assert not isinstance(pilot.app.screen, ReminderForm)
        assert app.submitted is not None
        assert app.submitted["title"] == "Weekly sync"
        assert app.submitted["schedule_kind"] == "recurring"
        assert app.submitted["cron"] == "0 9 * * 1"
        assert app.submitted["timezone"] == "UTC"
        assert app.submitted["run_at"] is None


@pytest.mark.asyncio
async def test_reminder_form_requires_run_at_for_one_time():
    """A one-time reminder without a run_at value shows a validation error."""
    app = FormTestApp()
    async with app.run_test() as pilot:
        await app.push_screen(ReminderForm())
        title_input = pilot.app.screen.query_one("#reminder-title", Input)
        title_input.value = "No run_at"
        await pilot.click("#reminder-save")
        await pilot.pause()

        error_widget = pilot.app.screen.query_one("#reminder-errors")
        assert "run at is required" in error_widget.visual.plain.lower()


@pytest.mark.asyncio
async def test_reminder_form_rejects_invalid_run_at():
    """A one-time reminder with an invalid run_at shows a validation error."""
    app = FormTestApp()
    async with app.run_test() as pilot:
        await app.push_screen(ReminderForm())
        title_input = pilot.app.screen.query_one("#reminder-title", Input)
        title_input.value = "Bad run_at"
        run_at_input = pilot.app.screen.query_one("#reminder-run-at", Input)
        run_at_input.value = "not-a-datetime"
        await pilot.click("#reminder-save")
        await pilot.pause()

        error_widget = pilot.app.screen.query_one("#reminder-errors")
        assert "run at must be a valid iso-8601 datetime" in error_widget.visual.plain.lower()


@pytest.mark.asyncio
async def test_reminder_form_requires_cron_for_recurring():
    """A recurring reminder without a cron expression shows a validation error."""
    app = FormTestApp()
    async with app.run_test() as pilot:
        await app.push_screen(ReminderForm())
        title_input = pilot.app.screen.query_one("#reminder-title", Input)
        title_input.value = "No cron"
        kind_select = pilot.app.screen.query_one("#reminder-kind", Select)
        kind_select.value = ScheduleKind.RECURRING.value
        await pilot.pause()

        tz_input = pilot.app.screen.query_one("#reminder-timezone", Input)
        tz_input.value = "UTC"
        await pilot.click("#reminder-save")
        await pilot.pause()

        error_widget = pilot.app.screen.query_one("#reminder-errors")
        assert "cron expression is required" in error_widget.visual.plain.lower()


@pytest.mark.asyncio
async def test_reminder_form_rejects_invalid_cron():
    """A recurring reminder with an invalid cron expression shows a validation error."""
    app = FormTestApp()
    async with app.run_test() as pilot:
        await app.push_screen(ReminderForm())
        title_input = pilot.app.screen.query_one("#reminder-title", Input)
        title_input.value = "Bad cron"
        kind_select = pilot.app.screen.query_one("#reminder-kind", Select)
        kind_select.value = ScheduleKind.RECURRING.value
        await pilot.pause()

        cron_input = pilot.app.screen.query_one("#reminder-cron", Input)
        cron_input.value = "not-a-cron"
        tz_input = pilot.app.screen.query_one("#reminder-timezone", Input)
        tz_input.value = "UTC"
        await pilot.click("#reminder-save")
        await pilot.pause()

        error_widget = pilot.app.screen.query_one("#reminder-errors")
        assert "cron expression is invalid" in error_widget.visual.plain.lower()


@pytest.mark.asyncio
async def test_reminder_form_requires_timezone_for_recurring():
    """A recurring reminder without a timezone shows a validation error."""
    app = FormTestApp()
    async with app.run_test() as pilot:
        await app.push_screen(ReminderForm())
        title_input = pilot.app.screen.query_one("#reminder-title", Input)
        title_input.value = "No timezone"
        kind_select = pilot.app.screen.query_one("#reminder-kind", Select)
        kind_select.value = ScheduleKind.RECURRING.value
        await pilot.pause()

        cron_input = pilot.app.screen.query_one("#reminder-cron", Input)
        cron_input.value = "0 9 * * 1"
        tz_input = pilot.app.screen.query_one("#reminder-timezone", Input)
        tz_input.value = ""
        await pilot.click("#reminder-save")
        await pilot.pause()

        error_widget = pilot.app.screen.query_one("#reminder-errors")
        assert "timezone is required" in error_widget.visual.plain.lower()


@pytest.mark.asyncio
async def test_reminder_form_rejects_invalid_timezone():
    """A recurring reminder with an unknown timezone shows a validation error."""
    app = FormTestApp()
    async with app.run_test() as pilot:
        await app.push_screen(ReminderForm())
        title_input = pilot.app.screen.query_one("#reminder-title", Input)
        title_input.value = "Bad timezone"
        kind_select = pilot.app.screen.query_one("#reminder-kind", Select)
        kind_select.value = ScheduleKind.RECURRING.value
        await pilot.pause()

        cron_input = pilot.app.screen.query_one("#reminder-cron", Input)
        cron_input.value = "0 9 * * 1"
        tz_input = pilot.app.screen.query_one("#reminder-timezone", Input)
        tz_input.value = "Mars/Phobos"
        await pilot.click("#reminder-save")
        await pilot.pause()

        error_widget = pilot.app.screen.query_one("#reminder-errors")
        assert "unknown timezone" in error_widget.visual.plain.lower()


@pytest.mark.asyncio
async def test_reminder_form_preserves_enabled_state_when_editing():
    """Editing a disabled reminder preserves its enabled state in the payload."""
    app = FormTestApp()
    disabled_task = ReminderTask(
        id="task-1",
        title="Existing",
        schedule_kind=ScheduleKind.ONE_TIME,
        run_at=datetime(2026, 7, 20, 14, 0, tzinfo=timezone.utc),
        enabled=False,
    )
    async with app.run_test() as pilot:
        await app.push_screen(ReminderForm(disabled_task))
        title_input = pilot.app.screen.query_one("#reminder-title", Input)
        title_input.value = "Updated"
        await pilot.click("#reminder-save")
        await pilot.pause()

        assert app.submitted is not None
        assert app.submitted["enabled"] is False


@pytest.mark.asyncio
async def test_reminder_form_cancel_dismisses_without_submitting():
    """Clicking Cancel dismisses the modal without emitting a submission event."""
    app = FormTestApp()
    async with app.run_test() as pilot:
        await app.push_screen(ReminderForm())
        await pilot.click("#reminder-cancel")
        await pilot.pause()

        assert not isinstance(pilot.app.screen, ReminderForm)
        assert app.submitted is None
