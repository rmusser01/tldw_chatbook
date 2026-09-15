"""Opening backup settings must survive an obsolete Console progress tick."""

import pytest
from textual.widgets import Button

from Tests.UI.test_console_rail_reconciliation import _RailHarness
from tldw_chatbook.UI.Console_Modules.left_rail import ConsoleLeftRail


class _ProgressRailHarness(_RailHarness):
    def build_rail(self) -> ConsoleLeftRail:
        rail = super().build_rail()
        rail._open_agent_progress = lambda: None
        rail._agent_progress_state = lambda: (3, {"current": 3})
        return rail


@pytest.mark.asyncio
async def test_progress_tick_during_descendant_replacement_keeps_console_usable():
    """A queued tick skips a removed child and later paints the current count."""
    async with _ProgressRailHarness().run_test() as pilot:
        rail = pilot.app.screen.query_one(ConsoleLeftRail)
        assert rail._progress_timer is not None
        await rail.query_one("#console-agent-progress", Button).remove()

        # Recomposition removes descendants while their parent still owns its
        # timer. Deliver the real callback in that gap through the message pump.
        rail.call_later(rail._sync_progress_count)
        await pilot.pause()
        assert pilot.app.is_running

        await rail.recompose()
        rail._sync_progress_count()
        assert str(rail.query_one("#console-agent-progress", Button).label) == (
            "Progress: 3 queued"
        )
