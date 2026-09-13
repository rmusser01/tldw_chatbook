"""Visible Create button in the Schedule Queue rail (UX F-07).

The only create affordance used to be the `c` key in the footer. This adds
a primary button beside the "Schedule Queue" pane title. Task 5 upgraded
its behavior from a direct reminder-form push to a two-item chooser
(Reminder / Recurring question) -- the `c` key binding is unchanged and
still opens the reminder form directly. Redesign PR-2, Task 3 relabels it
`Create ▾` and repositions it in the rail header alongside the new
`Mark all read` button -- same id/handler/chooser, unchanged.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button

from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from Tests.UI.schedules_test_helpers import (
    MockSchedulingDB,
    MockSchedulingServiceMixin,
)
from tldw_chatbook.UI.Screens.scheduling.forms.automation_definition_form import (
    AutomationDefinitionForm,
)
from tldw_chatbook.UI.Screens.scheduling.forms.new_task_choice_modal import (
    NewTaskChoiceModal,
)
from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import (
    ReminderForm,
    SchedulesWorkbench,
)


class LocalOnlyTestApp(ConsolidatedCSSApp):
    scheduling_service = None


class _EmptyService(MockSchedulingServiceMixin):
    """A service is required for the recurring-question path -- unlike the
    reminder path, `action_create_automation` refuses without one."""

    def __init__(self) -> None:
        self.owner_id = "local"
        self.db = MockSchedulingDB()

    async def list_tasks(self, owner_id=None, include_projections=True):
        return []


class ServiceBackedTestApp(ConsolidatedCSSApp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scheduling_service = _EmptyService()


class LocalOnlyBundledCSSTestApp(LocalOnlyTestApp):
    """Load canonical app styles, including the Scheduling feature sheet."""

    CSS_PATH = [str(path) for path in APP_STYLESHEETS]


@pytest.mark.asyncio
async def test_new_button_exists_and_opens_choice_chooser():
    """Redesign PR-2, Task 3: `Create ▾` (relabeled from `+ New`) still
    opens the same chooser via the same id/handler."""
    app = LocalOnlyTestApp()
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen

        button = workbench.query_one("#scheduling-new-task", Button)
        assert "Create" in str(button.label)

        pushed: list = []
        workbench.app.push_screen = lambda screen, callback=None: pushed.append(screen)
        button.press()
        await pilot.pause()

        assert pushed and isinstance(pushed[0], NewTaskChoiceModal)


@pytest.mark.asyncio
async def test_new_button_chooser_reminder_choice_opens_reminder_form():
    """Picking "Reminder…" in the chooser opens the reminder form directly."""
    app = LocalOnlyTestApp()
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen

        workbench.query_one("#scheduling-new-task", Button).press()
        await pilot.pause()
        assert isinstance(pilot.app.screen, NewTaskChoiceModal)

        await pilot.click("#new-task-choice-reminder")
        await pilot.pause()

        assert isinstance(pilot.app.screen, ReminderForm)


@pytest.mark.asyncio
async def test_new_button_chooser_recurring_choice_opens_automation_form():
    """Picking "Recurring question…" opens the automation-definition form
    -- the rail's `Create ▾` button's OTHER path (redesign PR-2, Task 3:
    "both create paths still open their modals")."""
    app = ServiceBackedTestApp()
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen

        workbench.query_one("#scheduling-new-task", Button).press()
        await pilot.pause()
        assert isinstance(pilot.app.screen, NewTaskChoiceModal)

        await pilot.click("#new-task-choice-automation")
        await pilot.pause()

        assert isinstance(pilot.app.screen, AutomationDefinitionForm)


@pytest.mark.asyncio
async def test_new_button_row_flattens_to_one_line_in_compact_mode():
    """Compact mode (see on_resize) reclaims the header's ROWS, not its
    buttons.

    Redesign PR-4, task 6 (ruling 6: "the rail degrades readably at 80")
    DELIBERATELY replaces this test's previous claim -- that the whole
    header hid below 84 columns. It hid `Create ▾`, `Mark all read` and
    `Results` with it, which are the only routes to those operations that
    need no selected row, and the row it bought back came from the
    3-row bordered Button box rather than from the buttons themselves.
    The header now flattens to a single row: the title yields (the screen
    header already names the pane) and the buttons lose their border.
    The narrow-width floor's own reachability claims are pinned in
    `test_schedules_responsive_floor.py`.
    """
    app = LocalOnlyBundledCSSTestApp()
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen

        # Same class on_resize applies to #scheduling-workbench once
        # hide_detail is true (width < 84).
        workbench.query_one("#scheduling-workbench").add_class("compact")
        await pilot.pause()

        header = workbench.query_one("#scheduling-list-header")
        assert header.display is True
        assert header.region.height == 1
        assert workbench.query_one("#scheduling-list-title").display is False
        for button_id in ("#scheduling-new-task", "#scheduling-results-badge"):
            button = workbench.query_one(button_id, Button)
            assert button.display is True
            assert button.region.height == 1
