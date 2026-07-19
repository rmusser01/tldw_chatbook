"""Tests for the reminder create/edit form."""

import pytest
from textual.app import App
from textual.widgets import Input

from tldw_chatbook.Scheduling.events import ReminderFormSubmitted
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
async def test_reminder_form_submits_when_valid():
    """A non-empty title dismisses the modal after posting the event."""
    app = FormTestApp()
    async with app.run_test() as pilot:
        await app.push_screen(ReminderForm())
        title_input = pilot.app.screen.query_one("#reminder-title", Input)
        title_input.value = "Water plants"
        await pilot.click("#reminder-save")
        await pilot.pause()

        assert not isinstance(pilot.app.screen, ReminderForm)
        assert app.submitted is not None
        assert app.submitted["title"] == "Water plants"
        assert app.submitted["schedule_kind"] == "one_time"


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
