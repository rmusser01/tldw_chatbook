"""Visible New button in the Schedule Queue pane (UX F-07).

The only create affordance used to be the `c` key in the footer. This adds
a primary button beside the "Schedule Queue" pane title, wired to the
existing `action_create_reminder`.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import (
    ReminderForm,
    SchedulesWorkbench,
)


class LocalOnlyTestApp(ConsolidatedCSSApp):
    scheduling_service = None


@pytest.mark.asyncio
async def test_new_button_exists_and_opens_create_form():
    app = LocalOnlyTestApp()
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen

        button = workbench.query_one("#scheduling-new-task", Button)
        assert "New" in str(button.label)

        pushed: list = []
        workbench.app.push_screen = lambda screen, callback=None: pushed.append(screen)
        button.press()
        await pilot.pause()

        assert pushed and isinstance(pushed[0], ReminderForm)
