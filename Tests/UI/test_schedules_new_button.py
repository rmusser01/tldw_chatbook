"""Visible New button in the Schedule Queue pane (UX F-07).

The only create affordance used to be the `c` key in the footer. This adds
a primary button beside the "Schedule Queue" pane title, wired to the
existing `action_create_reminder`.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button

from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import (
    ReminderForm,
    SchedulesWorkbench,
)


class LocalOnlyTestApp(ConsolidatedCSSApp):
    scheduling_service = None


class LocalOnlyBundledCSSTestApp(LocalOnlyTestApp):
    """Loads the app-level CSS bundle so `.compact` display rules resolve.

    ``ConsolidatedCSSApp`` only registers the screen/modal sheets by
    default; the scheduling feature CSS (``_scheduling.tcss``) lives in the
    app bundle, so a test asserting on its `display: none` rules needs this
    tier too (same pattern as ``Tests/UI/test_trace_responsive.py``).
    """

    CSS_PATH = BUNDLED_STYLESHEET


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


@pytest.mark.asyncio
async def test_new_button_row_hidden_in_compact_mode():
    """Compact mode (see on_resize) reclaims the header's row for the table.

    Hiding the whole header -- not just the title -- keeps the pre-change
    row budget: the New button stays reachable via the `c` key, which the
    footer advertises, instead of eating the one spare row at 80x24.
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
        assert header.display is False
