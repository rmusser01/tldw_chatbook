"""The agent-progress timer must survive its button not being there.

task-32639. ``ConsoleLeftRail._sync_progress_count`` runs on a 0.5s interval
and used to call ``query_one("#console-agent-progress")`` unguarded. That
button is composed only while ``_open_agent_progress`` is set, so a tick
landing while the agent section is between recomposes raised ``NoMatches``
out of the timer callback -- Textual re-raises a timer exception at the app,
which exits. Seen as a CI failure in
``test_library_opens_within_budget_on_a_seeded_profile``, whose only crime
was being slow enough to hit the window.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_console_rail_reconciliation import (
    _all_open_rail_state,
    _workspace_state,
)
from tldw_chatbook.Chat.console_session_settings import ConsoleSettingsSummaryState
from tldw_chatbook.UI.Console_Modules.left_rail import ConsoleLeftRail


class _ProgressRailHost(App[None]):
    """A rail whose agent-progress button exists, so the timer arms."""

    def __init__(self) -> None:
        super().__init__()
        self.progress_counts = {"current": 3}
        self.navigation = Static("Pending: 0", id="progress-navigation")

    def refresh_progress_navigation(self) -> None:
        self.navigation.update(f"Pending: {self.progress_counts.get('current', 0)}")

    def compose(self) -> ComposeResult:
        yield self.navigation
        yield ConsoleLeftRail(
            rail_state=_all_open_rail_state(),
            workspace_context_state=_workspace_state(),
            settings_summary_state=ConsoleSettingsSummaryState(
                model_row="Model: test",
                context_row="Context: 0",
                sampling_row="T 0.7 · max_tokens 100",
                identity_row="Identity: character",
            ),
            system_line_text="System: none",
            system_line_dim=True,
            fleet_line="1 agent running",
            agent_status_line="Running",
            agent_steps_text="one",
            agent_drilldown_active=False,
            agent_full_log_available=False,
            open_agent_progress=lambda: None,
            agent_progress_state=lambda: (
                self.progress_counts.get("current", 0),
                dict(self.progress_counts),
            ),
            refresh_progress_navigation=self.refresh_progress_navigation,
            show_character_section=False,
            character_avatar_widget_builder=(
                lambda _box=None, **_kwargs: Static("avatar")
            ),
            character_avatar_name="Samira",
        )


@pytest.mark.ui
@pytest.mark.asyncio
@private_profile_test
async def test_progress_tick_survives_the_button_leaving_the_tree(
    request: pytest.FixtureRequest,
) -> None:
    """Missing labels never block counts/navigation or outlive their rail."""

    app = _ProgressRailHost()
    async with app.run_test(size=(80, 40)) as pilot:
        rail = app.query_one(ConsoleLeftRail)
        button = app.query_one("#console-agent-progress", Button)
        assert str(button.label) == "Progress: 3 queued"
        assert str(app.navigation.content) == "Pending: 3"

        await button.remove()
        app.progress_counts = {"current": 5}
        # Deliver the queued callback through the mounted message pump.
        rail.call_later(rail._sync_progress_count)
        await pilot.pause()

        assert not app.query("#console-agent-progress")
        assert rail._progress_counts == {"current": 5}
        assert str(app.navigation.content) == "Pending: 5"
        assert app.is_running

        await rail.recompose()
        await pilot.pause(0.6)
        assert str(rail.query_one("#console-agent-progress", Button).label) == (
            "Progress: 5 queued"
        )

        await rail.remove()
        app.progress_counts = {"current": 8}
        await pilot.pause(0.6)
        assert str(app.navigation.content) == "Pending: 5"
        assert app._exception is None
