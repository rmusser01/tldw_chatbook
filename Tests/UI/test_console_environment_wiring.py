"""Screen-wiring tests for the Console Environment/Tasks panel (task-13).

This is the BOTH-SEAMS gate for the Environment redesign. Every projection
in ``Chat/console_environment_state.py`` and every dispatch rule in
``UI/Console_Modules/environment.py`` is already unit-tested against pure
fakes -- and all of those tests pass with ``ChatScreen`` completely
unwired. A prior fix in this area shipped broken for exactly that reason,
so the tests here drive the REAL screen through the production Console
harness and assert on the real seam each action is supposed to move: the
composer's draft text, the Change Review opener's kwargs, the rail
preference writer, the controller's own ``request_refresh``, and the
mounted sections' DOM.

Harness is ``Tests/UI/test_console_right_rail.py``'s ``make_console_pilot``
(a send-ready ``ChatScreen`` at 160x45, Inspect rail starting from its
persisted default -- closed). Data is always a CANNED
``EnvironmentSnapshot``: nothing here shells out to git or ``gh``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

from Tests.UI.test_console_right_rail import make_console_pilot
from tldw_chatbook.Chat.console_environment_state import (
    ENV_ROW_CHANGES,
    ENV_ROW_CHECKS_FIX,
    ENV_ROW_COMMIT_PUSH,
    ENV_ROW_PR_ADD,
    ENV_ROW_PR_OPEN,
    EnvironmentSnapshot,
    EnvSourceAvailability,
    GitEnvState,
    PrCheck,
    PrEnvState,
    TASKS_ROW_ADD,
    TasksEnvState,
    BranchTaskState,
)
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_inspector_section import (
    ConsoleInspectorSection,
    ConsoleInspectorSectionState,
    InspectorSectionRow,
)
from tldw_chatbook.Workspaces.change_tracking import ChangedFile

_FLEET_BUSY_LINE = "1 other agents running, 0 waiting for approval."


@asynccontextmanager
async def _console_screen(**kwargs):
    """Yield ``(pilot, screen)`` -- the harness itself yields only ``pilot``."""
    async with make_console_pilot(**kwargs) as pilot:
        yield pilot, pilot.app.screen


def _snapshot(*, files=(), pr=None, tasks=None, branch="feat/task-1-x"):
    """Build a canned OK-git snapshot (the only data these tests use)."""
    return EnvironmentSnapshot(
        git=GitEnvState(
            availability=EnvSourceAvailability.OK,
            root="/w",
            branch=branch,
            adds=sum(f.adds for f in files),
            dels=sum(f.dels for f in files),
            files=tuple(files),
        ),
        pr=pr or PrEnvState(),
        tasks=tasks or TasksEnvState(),
    )


def _row_ids(section: ConsoleInspectorSection) -> list[str]:
    return [row.row_id for row in section.rows]


# ---------------------------------------------------------------------------
# Row actions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fix_row_inserts_failure_text_into_composer():
    """The failing-checks "Fix" row must reach the real composer draft."""
    async with _console_screen() as (pilot, screen):
        snapshot = _snapshot(
            pr=PrEnvState(
                availability=EnvSourceAvailability.OK,
                number=7,
                title="T",
                state="OPEN",
                url="https://x/pull/7",
                checks=(PrCheck("ci-tests", "failure", "https://ci/1"),),
            )
        )
        screen._console_environment.snapshot = snapshot
        screen._land_console_environment(snapshot)
        await pilot.pause()

        screen._handle_console_environment_row("environment", ENV_ROW_CHECKS_FIX)
        await pilot.pause()

        composer = screen.query_one("#console-native-composer", ConsoleComposerBar)
        draft = composer.draft_text()
        assert "ci-tests" in draft and "https://ci/1" in draft


@pytest.mark.asyncio
async def test_pr_add_row_inserts_pr_summary_into_composer():
    """The PR "Add to chat" row inserts the PR summary, not the checks text."""
    async with _console_screen() as (pilot, screen):
        snapshot = _snapshot(
            pr=PrEnvState(
                availability=EnvSourceAvailability.OK,
                number=7,
                title="Wire the panel",
                state="OPEN",
                url="https://x/pull/7",
            )
        )
        screen._console_environment.snapshot = snapshot
        screen._land_console_environment(snapshot)
        await pilot.pause()

        screen._handle_console_environment_row("environment", ENV_ROW_PR_ADD)
        await pilot.pause()

        composer = screen.query_one("#console-native-composer", ConsoleComposerBar)
        draft = composer.draft_text()
        assert "PR #7" in draft and "Wire the panel" in draft
        assert "https://x/pull/7" in draft


@pytest.mark.asyncio
async def test_tasks_add_row_inserts_branch_task_into_composer():
    """The Tasks section's "Add task to chat" row routes through the SAME handler."""
    async with _console_screen() as (pilot, screen):
        snapshot = _snapshot(
            tasks=TasksEnvState(
                availability=EnvSourceAvailability.OK,
                branch_task=BranchTaskState(
                    task_id="13",
                    title="Screen wiring",
                    status="In Progress",
                    path="backlog/tasks/task-13.md",
                ),
            )
        )
        screen._console_environment.snapshot = snapshot
        screen._land_console_environment(snapshot)
        await pilot.pause()

        screen._handle_console_environment_row("tasks", TASKS_ROW_ADD)
        await pilot.pause()

        composer = screen.query_one("#console-native-composer", ConsoleComposerBar)
        draft = composer.draft_text()
        assert "task-13" in draft and "Screen wiring" in draft
        assert "backlog/tasks/task-13.md" in draft


@pytest.mark.asyncio
async def test_row_activation_toggles_expansion_and_rerenders():
    """Activating an expandable row re-projects and re-renders the section."""
    async with _console_screen() as (pilot, screen):
        snapshot = _snapshot(
            files=(
                ChangedFile(path="a.py", status="M", adds=3, dels=1),
                ChangedFile(path="b.py", status="A", adds=7, dels=0),
            )
        )
        screen._console_environment.snapshot = snapshot
        screen._land_console_environment(snapshot)
        await pilot.pause()
        section = screen.query_one(
            "#console-environment-section", ConsoleInspectorSection
        )
        assert not any(rid.startswith("env-file-") for rid in _row_ids(section))

        screen._handle_console_environment_row("environment", ENV_ROW_CHANGES)
        await pilot.pause()
        assert "env-file-1" in _row_ids(section)
        assert section.query("#console-inspector-section-environment-row-1")

        screen._handle_console_environment_row("environment", ENV_ROW_CHANGES)
        await pilot.pause()
        assert not any(rid.startswith("env-file-") for rid in _row_ids(section))


@pytest.mark.asyncio
async def test_commit_push_row_opens_change_review_in_current_mode():
    """"Commit or push" must open Change Review straight onto the working tree."""
    async with _console_screen() as (pilot, screen):
        captured: list[dict] = []
        screen._open_change_review = lambda *a, **kw: captured.append(dict(kw))
        snapshot = _snapshot(
            files=(ChangedFile(path="a.py", status="M", adds=3, dels=1),)
        )
        screen._console_environment.snapshot = snapshot
        screen._land_console_environment(snapshot)
        await pilot.pause()

        screen._handle_console_environment_row("environment", ENV_ROW_COMMIT_PUSH)
        await pilot.pause()
        assert captured == [{"initial_current_mode": True}]


@pytest.mark.asyncio
async def test_changes_review_row_opens_change_review_in_default_mode():
    """The expanded "Review in Change Review" row opens the ordinary view."""
    async with _console_screen() as (pilot, screen):
        captured: list[dict] = []
        screen._open_change_review = lambda *a, **kw: captured.append(dict(kw))
        screen._handle_console_environment_row("environment", "env-changes-review")
        await pilot.pause()
        assert captured == [{}]


@pytest.mark.asyncio
async def test_pr_open_row_uses_app_open_url_and_never_raises():
    """PR "Open in browser" goes through ``app.open_url`` (never webbrowser)."""
    async with _console_screen() as (pilot, screen):
        opened: list[str] = []
        screen.app.open_url = lambda url, **kw: opened.append(url)
        snapshot = _snapshot(
            pr=PrEnvState(
                availability=EnvSourceAvailability.OK,
                number=7,
                title="T",
                state="OPEN",
                url="https://x/pull/7",
            )
        )
        screen._console_environment.snapshot = snapshot
        screen._handle_console_environment_row("environment", ENV_ROW_PR_OPEN)
        await pilot.pause()
        assert opened == ["https://x/pull/7"]

        def _boom(url, **kw):
            raise RuntimeError("no browser")

        screen.app.open_url = _boom
        screen._handle_console_environment_row("environment", ENV_ROW_PR_OPEN)
        await pilot.pause()  # a failing opener must not escape the handler


@pytest.mark.asyncio
async def test_row_activation_message_routes_through_the_screen_handler():
    """A real ``RowActivated`` for the environment section reaches the handler.

    The fleet section already owns that handler; this proves the section-id
    fan-out, not just the helper method the other tests call directly.
    """
    async with _console_screen() as (pilot, screen):
        snapshot = _snapshot(
            files=(ChangedFile(path="a.py", status="M", adds=3, dels=1),)
        )
        screen._console_environment.snapshot = snapshot
        screen._land_console_environment(snapshot)
        await pilot.pause()
        section = screen.query_one(
            "#console-environment-section", ConsoleInspectorSection
        )
        section.post_message(
            ConsoleInspectorSection.RowActivated("environment", ENV_ROW_CHANGES)
        )
        await pilot.pause()
        assert "env-file-0" in _row_ids(section)


# ---------------------------------------------------------------------------
# Section-level messages
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collapse_toggle_persists_via_rail_preferences():
    async with _console_screen() as (pilot, screen):
        captured: list[dict] = []
        screen._set_console_rail_preference = lambda **kw: captured.append(kw)
        section = screen.query_one(
            "#console-environment-section", ConsoleInspectorSection
        )
        section.set_open(False)
        await pilot.pause()
        assert captured and captured[0]["section_updates"] == {"environment": False}
        assert captured[0]["notify_on_failure"] is False


@pytest.mark.asyncio
async def test_tasks_collapse_toggle_persists_under_its_own_id():
    async with _console_screen() as (pilot, screen):
        captured: list[dict] = []
        screen._set_console_rail_preference = lambda **kw: captured.append(kw)
        screen.query_one("#console-tasks-section", ConsoleInspectorSection).set_open(
            False
        )
        await pilot.pause()
        assert captured and captured[0]["section_updates"] == {"tasks": False}


@pytest.mark.asyncio
async def test_refresh_view_all_forces_net_tier():
    async with _console_screen() as (pilot, screen):
        captured: list[dict] = []
        screen._console_environment.request_refresh = lambda **kw: captured.append(kw)
        section = screen.query_one(
            "#console-environment-section", ConsoleInspectorSection
        )
        section.post_message(ConsoleInspectorSection.ViewAllRequested("environment"))
        await pilot.pause()
        assert captured == [{"include_net": True, "force_net": True}]


# ---------------------------------------------------------------------------
# Landing / degradation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_landing_for_stale_root_does_not_touch_sections():
    """No root and no mounted Environment section: land quietly, change nothing."""
    async with _console_screen() as (pilot, screen):
        screen._review_selection._console_change_review_workspace_roots = lambda: None
        assert screen._console_environment_root() is None

        tasks_section = screen.query_one(
            "#console-tasks-section", ConsoleInspectorSection
        )
        assert tasks_section.rows == ()
        await screen.query_one("#console-environment-section").remove()
        await pilot.pause()

        snapshot = _snapshot(
            tasks=TasksEnvState(
                availability=EnvSourceAvailability.OK,
                branch_task=BranchTaskState(
                    task_id="13", title="Screen wiring", status="In Progress"
                ),
            )
        )
        screen._land_console_environment(snapshot)  # must not raise
        await pilot.pause()
        assert tasks_section.rows == ()  # untouched, not half-applied


@pytest.mark.asyncio
async def test_poll_timer_and_root_accessor_are_wired():
    """The 10s local poll exists and the root accessor reads the review roots."""
    async with _console_screen() as (pilot, screen):
        assert screen._console_environment_poll_timer is not None
        screen._review_selection._console_change_review_workspace_roots = (
            lambda: ("/w/one", "/w/two")
        )
        assert screen._console_environment_root() == "/w/one"


@pytest.mark.asyncio
async def test_environment_workers_cannot_panic_the_app():
    """Gathers run with ``exit_on_error=False``.

    Textual's default panics the whole app on a raising worker, and these
    jobs shell out to git/``gh`` and marshal back through
    ``call_from_thread`` (which itself raises on a torn-down screen). A
    status panel must never be able to take the app down.
    """
    async with _console_screen() as (pilot, screen):
        captured: list[dict] = []
        screen.run_worker = lambda job, **kw: captured.append(kw)
        screen._console_environment._dispatch_local("/w/one")
        assert captured and captured[0]["exit_on_error"] is False
        assert captured[0]["thread"] is True and captured[0]["exclusive"] is True


@pytest.mark.asyncio
async def test_fleet_sync_and_app_focus_nudge_the_local_tier():
    """Both spec'd nudges reach the controller (its own guard handles a closed rail)."""
    async with _console_screen() as (pilot, screen):
        calls: list[dict] = []
        screen._console_environment.request_refresh = lambda **kw: calls.append(kw)

        screen._run_coalesced_console_agent_fleet_sync()
        assert calls == [{}]

        calls.clear()
        screen.on_console_app_focus_environment_refresh(None)
        assert calls == [{}]


# ---------------------------------------------------------------------------
# Addition A: fleet activity re-opens the Inspect rail (and its own section)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fleet_activity_auto_opens_the_inspector_rail():
    async with _console_screen() as (pilot, screen):
        assert screen._current_console_rail_state().right_open is False
        screen._agent._console_agent_fleet_summary_line = lambda: _FLEET_BUSY_LINE
        assert screen._current_console_rail_state().right_open is True


@pytest.mark.asyncio
async def test_manual_collapse_while_fleet_busy_sticks_until_the_fleet_goes_idle():
    async with _console_screen() as (pilot, screen):
        screen._agent._console_agent_fleet_summary_line = lambda: _FLEET_BUSY_LINE
        assert screen._current_console_rail_state().right_open is True

        screen._set_console_rail_preference(right_open=False)
        assert screen._console_fleet_inspector_dismissed is True
        # The tick-recomputed force must not fight the manual collapse.
        assert screen._current_console_rail_state().right_open is False

        # ...and the stickiness is scoped to THIS busy window.
        screen._agent._console_agent_fleet_summary_line = lambda: ""
        assert screen._current_console_rail_state().right_open is False
        assert screen._console_fleet_inspector_dismissed is False
        screen._agent._console_agent_fleet_summary_line = lambda: _FLEET_BUSY_LINE
        assert screen._current_console_rail_state().right_open is True


@pytest.mark.asyncio
async def test_fleet_section_opens_itself_on_first_activity():
    """The moved fleet section defaults collapsed; first activity must open it."""
    async with _console_screen() as (pilot, screen):
        fleet_section = screen.query_one(
            "#console-agent-section-subagents", ConsoleInspectorSection
        )
        assert fleet_section.open is False

        payload = list(screen._agent._console_agent_section_payload())
        payload[2] = ConsoleInspectorSectionState(
            rows=(
                InspectorSectionRow(
                    row_id="fleet-1", primary_text="child agent", clickable=True
                ),
            ),
            summary="1 working",
        )
        payload[3] = _FLEET_BUSY_LINE
        screen._agent._console_agent_section_payload = lambda: tuple(payload)
        screen._console_agent_section_last = None
        screen._sync_console_agent_section()
        await pilot.pause()
        assert fleet_section.open is True

        # A manual collapse while still busy is not re-forced by the next tick.
        fleet_section.set_open(False)
        await pilot.pause()
        screen._console_agent_section_last = None
        screen._sync_console_agent_section()
        await pilot.pause()
        assert fleet_section.open is False
