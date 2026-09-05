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
from textual.app import App
from textual.events import AppFocus

from Tests.UI.test_console_fleet_panel import _FleetBridge
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_console_right_rail import make_console_pilot
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Agents.fleet_coordinator import FleetHandle
from tldw_chatbook.app import TldwCli
from tldw_chatbook.UI.Console_Modules.environment import UNKNOWN_ROOT
from tldw_chatbook.Chat.console_environment_state import (
    ENV_ROW_CHANGES,
    ENV_ROW_CHECKS_FIX,
    ENV_ROW_COMMIT_PUSH,
    ENV_ROW_PENDING,
    ENV_ROW_PR,
    ENV_ROW_PR_ADD,
    ENV_ROW_PR_OPEN,
    ENV_ROW_UNBOUND,
    ENV_ROW_UNBOUND_NOTE,
    EnvironmentSnapshot,
    EnvSourceAvailability,
    GitEnvState,
    PrCheck,
    PrEnvState,
    TASKS_ROW_ADD,
    TasksEnvState,
    BranchTaskState,
    unbound_snapshot,
)
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_inspector_section import (
    ConsoleInspectorSection,
    ConsoleInspectorSectionRow,
)
from tldw_chatbook.Workspaces.change_tracking import ChangedFile

_FLEET_BUSY_LINE = "1 other agents running, 0 waiting for approval."


def _right_rail_open(screen) -> bool:
    rail = screen.query_one("#console-right-rail")
    return bool(rail.display) and rail.styles.display != "none"


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


@pytest.mark.asyncio
async def test_expanding_a_row_keeps_focus_on_that_row(monkeypatch):
    """F2: Enter-Enter must collapse the ROW, never the whole section.

    Expanding recomposes the section (structural key change) and unmounts
    the focused row; Textual's focus reset then landed the caret on the
    section's collapse chevron, so the second Enter collapsed the entire
    Environment section. The section must stay open and focus must return
    to the same row.
    """
    async with _console_screen() as (pilot, screen):
        await screen.action_toggle_console_inspector_rail()
        await pilot.pause()
        assert _right_rail_open(screen)

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

        def _changes_row():
            return next(
                widget
                for widget in section.query(ConsoleInspectorSectionRow)
                if widget.row_id == ENV_ROW_CHANGES
            )

        _changes_row().focus()
        await pilot.pause()
        assert screen.focused is _changes_row()

        # Expand.
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        assert "env-file-1" in _row_ids(section)
        focused = screen.focused
        assert isinstance(focused, ConsoleInspectorSectionRow)
        assert focused.row_id == ENV_ROW_CHANGES  # NOT the collapse toggle
        assert focused.clickable  # still the expandable row, still activatable

        # Collapse again with a second Enter: the row collapses, the SECTION
        # stays open, and focus stays put.
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        assert not any(rid.startswith("env-file-") for rid in _row_ids(section))
        assert section.open  # the section itself was never collapsed
        focused = screen.focused
        assert isinstance(focused, ConsoleInspectorSectionRow)
        assert focused.row_id == ENV_ROW_CHANGES


@pytest.mark.asyncio
async def test_expansion_from_an_unfocused_row_does_not_steal_focus():
    """Negative control: a click-driven expansion never grabs the caret."""
    async with _console_screen() as (pilot, screen):
        await screen.action_toggle_console_inspector_rail()
        await pilot.pause()
        snapshot = _snapshot(
            files=(ChangedFile(path="a.py", status="M", adds=3, dels=1),)
        )
        screen._console_environment.snapshot = snapshot
        screen._land_console_environment(snapshot)
        await pilot.pause()

        screen._focus_console_workbench_target("console-native-composer")
        await pilot.pause()
        focused_before = screen.focused
        assert focused_before is not None

        screen._handle_console_environment_row("environment", ENV_ROW_CHANGES)
        await pilot.pause()
        await pilot.pause()
        assert screen.focused is focused_before


# ---------------------------------------------------------------------------
# TASK-31661: poll-driven sync must not steal rail focus
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_poll_landing_that_adds_a_row_keeps_focus_on_the_same_row_id():
    """TASK-31661 AC#1: a background sync that ADDS a row must not move focus.

    The 10s Environment poll calls `_land_console_environment` with
    whatever the controller last gathered -- an external file change can
    grow the Environment section's row set (a new `env-file-N` row)
    while the rail still has focus parked on a row further down (the
    live defect's own example: "Review in Change Review"). The row set
    changing is a STRUCTURAL change (`ConsoleInspectorSection.
    _structural_key`), so `sync_state` recomposes, and Textual's own
    focus reset would otherwise land the caret on a widget above the
    section header with no visible focus indication.
    """
    async with _console_screen() as (pilot, screen):
        await screen.action_toggle_console_inspector_rail()
        await pilot.pause()

        one_file = _snapshot(
            files=(ChangedFile(path="a.py", status="M", adds=3, dels=1),)
        )
        screen._console_environment.snapshot = one_file
        screen._land_console_environment(one_file)
        await pilot.pause()

        # Expand "Changes" so "Review in Change Review" mounts.
        screen._handle_console_environment_row("environment", ENV_ROW_CHANGES)
        await pilot.pause()
        section = screen.query_one(
            "#console-environment-section", ConsoleInspectorSection
        )
        assert "env-changes-review" in _row_ids(section)

        def _review_row():
            return next(
                widget
                for widget in section.query(ConsoleInspectorSectionRow)
                if widget.row_id == "env-changes-review"
            )

        _review_row().focus()
        await pilot.pause()
        assert screen.focused is _review_row()

        # A poll lands a snapshot with an ADDED file -- structural change.
        two_files = _snapshot(
            files=(
                ChangedFile(path="a.py", status="M", adds=3, dels=1),
                ChangedFile(path="b.py", status="A", adds=7, dels=0),
            )
        )
        screen._console_environment.snapshot = two_files
        screen._land_console_environment(two_files)
        await pilot.pause()
        await pilot.pause()

        assert "env-file-1" in _row_ids(section)
        focused = screen.focused
        assert isinstance(focused, ConsoleInspectorSectionRow)
        assert focused.row_id == "env-changes-review"


@pytest.mark.asyncio
async def test_poll_landing_that_removes_the_focused_row_lands_on_the_nearest_survivor():
    """TASK-31661 AC#1/#2: removing the focused row falls back to its neighbor.

    Focus is parked on "Open in browser" (``env-pr-open``, clickable/
    focusable) inside the expanded PR row. A poll lands a snapshot where
    the PR has disappeared entirely (closed/merged and no longer
    reported) -- the whole PR sub-tree (``env-pr``, ``env-pr-title``,
    ``env-pr-open``, ``env-pr-add``) is gone, so the exact row_id cannot
    be restored. The candidate ladder (same index, then index-1, then the
    first row) all overshoot the now-3-row list, so this exercises the
    final "first row" rung -- the restore must still land on a row with
    a real focus indicator, never on whatever Textual's unmount-triggered
    reset happened to pick.

    File rows (``env-file-N``) are deliberately NOT used for this
    scenario: they are not ``clickable``, so they can never actually hold
    focus in the first place (`InspectorSectionRow.clickable` defaults to
    `False`, and only `clickable or cancellable` rows get `can_focus`).
    """
    async with _console_screen() as (pilot, screen):
        await screen.action_toggle_console_inspector_rail()
        await pilot.pause()

        with_pr = _snapshot(
            pr=PrEnvState(
                availability=EnvSourceAvailability.OK,
                number=7,
                title="Add feature",
                state="OPEN",
                url="https://x/pull/7",
            )
        )
        screen._console_environment.snapshot = with_pr
        screen._land_console_environment(with_pr)
        await pilot.pause()
        screen._handle_console_environment_row("environment", ENV_ROW_PR)
        await pilot.pause()
        section = screen.query_one(
            "#console-environment-section", ConsoleInspectorSection
        )
        assert ENV_ROW_PR_OPEN in _row_ids(section)

        def _row(row_id):
            return next(
                widget
                for widget in section.query(ConsoleInspectorSectionRow)
                if widget.row_id == row_id
            )

        _row(ENV_ROW_PR_OPEN).focus()
        await pilot.pause()
        assert screen.focused is _row(ENV_ROW_PR_OPEN)

        # A poll lands a snapshot where the PR is gone entirely -- the
        # whole PR sub-tree, including the focused row_id, disappears.
        without_pr = _snapshot()
        screen._console_environment.snapshot = without_pr
        screen._land_console_environment(without_pr)
        await pilot.pause()
        await pilot.pause()

        assert ENV_ROW_PR_OPEN not in _row_ids(section)
        focused = screen.focused
        assert isinstance(focused, ConsoleInspectorSectionRow)
        assert focused.section_id == "environment"
        # Neighborhood ladder overshoots (previous index 5, new list has 3
        # rows) -- falls all the way back to the first row.
        assert focused.row_id == ENV_ROW_CHANGES


@pytest.mark.asyncio
async def test_poll_landing_never_hijacks_focus_outside_the_rail():
    """TASK-31661 negative control: focus outside the rail is untouched.

    Zero overhead is part of the spec: when focus was never inside the
    Environment/Tasks sections, `_land_console_environment` must not
    fight the user by dragging focus back into the rail.
    """
    async with _console_screen() as (pilot, screen):
        await screen.action_toggle_console_inspector_rail()
        await pilot.pause()
        one_file = _snapshot(
            files=(ChangedFile(path="a.py", status="M", adds=3, dels=1),)
        )
        screen._console_environment.snapshot = one_file
        screen._land_console_environment(one_file)
        await pilot.pause()
        screen._handle_console_environment_row("environment", ENV_ROW_CHANGES)
        await pilot.pause()
        section = screen.query_one(
            "#console-environment-section", ConsoleInspectorSection
        )
        assert "env-changes-review" in _row_ids(section)

        screen._focus_console_workbench_target("console-native-composer")
        await pilot.pause()
        focused_before = screen.focused
        assert focused_before is not None

        two_files = _snapshot(
            files=(
                ChangedFile(path="a.py", status="M", adds=3, dels=1),
                ChangedFile(path="b.py", status="A", adds=7, dels=0),
            )
        )
        screen._console_environment.snapshot = two_files
        screen._land_console_environment(two_files)
        await pilot.pause()
        await pilot.pause()

        assert "env-file-1" in _row_ids(section)
        assert screen.focused is focused_before


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
    """No root and no mounted Environment section: land quietly, change nothing.

    Updated pin (TASK-31660 round 1): stubs the DEFINITIVE empty answer
    ``()`` rather than ``None``. ``None`` from this accessor now means "could
    not determine" and maps to ``UNKNOWN_ROOT``, which is a different case
    (covered by its own tests below); this test is about a real no-root
    landing, so it must ask for the real no-root answer.
    """
    async with _console_screen() as (pilot, screen):
        screen._review_selection._console_change_review_workspace_roots = lambda: ()
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
async def test_unbound_landing_replaces_the_previous_repos_paint():
    """TASK-31660 AC #1/#3 at the SCREEN seam.

    The reported P0 was a repaint failure, not a projection failure: after a
    switch to an unbound workspace the mounted rows still read the previous
    repository's branch and counts, and still offered "Commit or push - N
    files". Drives the real sections through the real landing path.
    """
    async with _console_screen() as (pilot, screen):
        bound = _snapshot(
            files=(ChangedFile(path="a.py", status="M", adds=3, dels=1),),
            branch="feat/previous-repo",
            tasks=TasksEnvState(
                availability=EnvSourceAvailability.OK,
                branch_task=BranchTaskState(
                    task_id="31660", title="State honesty", status="In Progress"
                ),
            ),
        )
        screen._console_environment.snapshot = bound
        screen._land_console_environment(bound)
        await pilot.pause()
        section = screen.query_one(
            "#console-environment-section", ConsoleInspectorSection
        )
        tasks_section = screen.query_one(
            "#console-tasks-section", ConsoleInspectorSection
        )
        assert ENV_ROW_COMMIT_PUSH in _row_ids(section)
        assert tasks_section.styles.display == "block"

        unbound = unbound_snapshot()
        screen._console_environment.snapshot = unbound
        screen._land_console_environment(unbound)
        await pilot.pause()

        assert _row_ids(section) == [ENV_ROW_UNBOUND, ENV_ROW_UNBOUND_NOTE]
        assert ENV_ROW_COMMIT_PUSH not in _row_ids(section)
        painted = " ".join(
            str(static.renderable)
            for static in section.query(".console-inspector-section-row-primary")
        )
        assert "No folder is bound to this conversation's workspace" in painted
        assert "not a report that nothing changed" in painted
        assert "feat/previous-repo" not in painted
        assert "Commit or push" not in painted
        # The Tasks card goes with it -- it described the other repo's backlog.
        assert tasks_section.rows == ()
        assert tasks_section.styles.display == "none"


@pytest.mark.asyncio
async def test_cold_start_rail_never_paints_the_no_git_workspace_negative():
    """AC #2 at the screen seam: PENDING, not a claim, before anything lands."""
    async with _console_screen() as (pilot, screen):
        section = screen.query_one(
            "#console-environment-section", ConsoleInspectorSection
        )
        assert _row_ids(section) == [ENV_ROW_PENDING]
        painted = " ".join(
            str(static.renderable)
            for static in section.query(".console-inspector-section-row-primary")
        )
        assert "Checking workspace" in painted
        assert "No git workspace" not in painted


@pytest.mark.asyncio
async def test_refresh_while_unbound_re_lands_through_the_real_controller():
    """AC #4: pressing Refresh with no bound folder is not a no-op.

    Drives the real "Refresh" tail message through the real handler and the
    real controller -- the seam the old early-return made inert.
    """
    async with _console_screen() as (pilot, screen):
        screen._review_selection._console_change_review_workspace_roots = lambda: ()
        assert screen._console_environment_root() is None
        await screen.action_toggle_console_inspector_rail()
        await pilot.pause()
        assert _right_rail_open(screen)

        landed: list = []
        screen._console_environment._on_snapshot = lambda snap: (
            landed.append(snap) or screen._land_console_environment(snap)
        )
        section = screen.query_one(
            "#console-environment-section", ConsoleInspectorSection
        )
        section.post_message(ConsoleInspectorSection.ViewAllRequested("environment"))
        await pilot.pause()

        assert landed and all(
            snap.git.availability is EnvSourceAvailability.UNBOUND for snap in landed
        )
        assert _row_ids(section) == [ENV_ROW_UNBOUND, ENV_ROW_UNBOUND_NOTE]


@pytest.mark.asyncio
async def test_unresolvable_root_reads_as_unknown_not_as_unbound():
    """TASK-31660 round 1: a swallowed exception must not assert "not bound".

    `_console_change_review_workspace_roots` returns ``None`` on ANY
    exception, and the layer under it returns ``None`` when the chat
    controller is absent or has no active session. Those are "cannot tell";
    only ``()`` -- what `resolve_turn_execution_context` returns for a
    workspace that binds no folder -- is "answered: nothing bound".
    """
    async with _console_screen() as (pilot, screen):
        def _boom():
            raise RuntimeError("roots accessor exploded")

        # Drive the REAL degradation path (the try/except in
        # `_console_change_review_workspace_roots`), not a stub of its result.
        screen._review_selection._workspace_roots_accessor = _boom
        assert screen._review_selection._console_change_review_workspace_roots() is None
        assert screen._console_environment_root() is UNKNOWN_ROOT

        # ...and the definitive empty answer is still plain None.
        screen._review_selection._workspace_roots_accessor = lambda: ()
        assert screen._console_environment_root() is None
        screen._review_selection._workspace_roots_accessor = lambda: ("/w/one",)
        assert screen._console_environment_root() == "/w/one"


@pytest.mark.asyncio
async def test_unknown_root_never_paints_the_unbound_copy():
    """The on-screen half of the same finding, through the real controller."""
    async with _console_screen() as (pilot, screen):
        bound = _snapshot(
            files=(ChangedFile(path="a.py", status="M", adds=3, dels=1),),
            branch="feat/still-here",
        )
        screen._console_environment.snapshot = bound
        screen._land_console_environment(bound)
        await pilot.pause()
        section = screen.query_one(
            "#console-environment-section", ConsoleInspectorSection
        )
        assert ENV_ROW_COMMIT_PUSH in _row_ids(section)

        def _boom():
            raise RuntimeError("roots accessor exploded")

        screen._review_selection._workspace_roots_accessor = _boom
        await screen.action_toggle_console_inspector_rail()  # opens -> refresh
        await pilot.pause()
        screen._console_environment.poll_tick()
        await pilot.pause()

        # Previous paint stands; no negative was asserted on a failure.
        assert ENV_ROW_COMMIT_PUSH in _row_ids(section)
        assert ENV_ROW_UNBOUND not in _row_ids(section)
        painted = " ".join(
            str(static.renderable)
            for static in section.query(".console-inspector-section-row-primary")
        )
        assert "No folder is bound" not in painted
        assert "feat/still-here" in painted
        assert screen._console_environment.snapshot is bound


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
async def test_fleet_sync_nudges_the_local_tier():
    """The coalesced fleet sync nudges the controller (its guard handles a closed rail)."""
    async with _console_screen() as (pilot, screen):
        calls: list[dict] = []
        screen._console_environment.request_refresh = lambda **kw: calls.append(kw)
        screen._run_coalesced_console_agent_fleet_sync()
        assert calls == [{}]


@asynccontextmanager
async def _app_focus_screen():
    """Host the real ChatScreen under an App carrying the PRODUCTION handler.

    Every Console UI test runs under ``ConsoleHarness`` -- a plain
    ``ConsolidatedCSSApp``, NOT ``TldwCli`` -- so the app-level handler the
    real binary uses is unreachable from the usual harness. Binding the
    production function object itself onto the host keeps the whole route
    real: post to the App's queue, Textual's own name dispatch, the
    production body, the real mounted ChatScreen, the real controller. The
    assertion below pins the other half -- that ``TldwCli`` is where that
    function actually lives, so this cannot pass against a handler that has
    been renamed or deleted out of production.
    """
    assert "on_app_focus" in TldwCli.__dict__, (
        "TldwCli must own the AppFocus handler: AppFocus is bubble=False and "
        "the driver posts it only to the App, so nothing below the App can "
        "observe a focus regain."
    )

    class _FocusHarness(ConsoleHarness):
        on_app_focus = TldwCli.__dict__["on_app_focus"]

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = _FocusHarness(app)
    async with host.run_test(size=(160, 45)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await pilot.pause(0.2)
        yield pilot, console


@pytest.mark.asyncio
async def test_terminal_focus_regain_reaches_the_controller_through_the_real_event():
    """`AppFocus` posted for real -- never a direct handler call.

    `AppFocus` is `bubble=False` and the driver posts it ONLY to the App;
    events travel UP the DOM, never down, so a screen-level `@on(AppFocus)`
    handler can never run. An earlier revision of this wiring shipped exactly
    that dead handler, and its test passed because it called the method
    directly. This one posts the production event and asserts it lands.
    """
    async with _app_focus_screen() as (pilot, screen):
        calls: list[dict] = []
        screen._console_environment.request_refresh = lambda **kw: calls.append(kw)

        pilot.app.post_message(AppFocus())
        await pilot.pause()
        await pilot.pause()

        assert calls == [{}]


@pytest.mark.asyncio
async def test_app_focus_forwarding_tolerates_a_screen_that_does_not_want_it():
    """The forwarding is an opt-in: a screen without the method is a no-op.

    Same production function object, a plain default screen instead of the
    Console. A focus event must never be able to raise out of the App.
    """

    class _BareApp(App):
        on_app_focus = TldwCli.__dict__["on_app_focus"]

    app = _BareApp()
    async with app.run_test(size=(80, 24)) as pilot:
        assert not hasattr(pilot.app.screen, "notify_terminal_focus_regained")
        pilot.app.post_message(AppFocus())
        await pilot.pause()
        await pilot.pause()
        assert pilot.app.is_running  # nothing raised out of the handler


@pytest.mark.asyncio
async def test_opening_the_inspect_rail_refreshes_both_tiers():
    """I1: drive the real accelerator, not the controller method.

    Closed -> open must fetch (an opened rail showing 10s-stale data is the
    defect); open -> closed must dispatch nothing.
    """
    async with _console_screen() as (pilot, screen):
        opens: list[int] = []
        screen._console_environment.notify_rail_opened = lambda: opens.append(1)
        assert not _right_rail_open(screen)

        await screen.action_toggle_console_inspector_rail()
        await pilot.pause()
        assert _right_rail_open(screen)
        assert opens == [1]

        await screen.action_toggle_console_inspector_rail()
        await pilot.pause()
        assert not _right_rail_open(screen)
        assert opens == [1]  # closing dispatches nothing


# ---------------------------------------------------------------------------
# Addition A: fleet activity re-opens the Inspect rail (and its own section)
# ---------------------------------------------------------------------------


def _running_handle(handle_id: str) -> FleetHandle:
    return FleetHandle(
        handle_id=handle_id,
        run_id=f"run-{handle_id}",
        agent="researcher",
        task="find pricing",
        status="running",
        started_at=1000.0,
    )


async def _wire_fleet_bridge(pilot, screen, bridge) -> None:
    """Attach the established fake fleet bridge (test_console_fleet_panel's)."""
    screen._console_agent_bridge = bridge
    screen._console_agent_drilldown_run_id = None
    screen._character._current_console_rail_conversation_id = lambda: "conv-A"
    screen._agent._console_agent_drilldown_conversation_id = "conv-A"
    await pilot.pause()


async def _real_fleet_sync(pilot, screen) -> None:
    """Run one REAL agent-section sync (rows derived from the bridge)."""
    screen._console_agent_section_last = None
    screen._sync_console_agent_section()
    await pilot.pause()


@pytest.mark.asyncio
async def test_fleet_rows_drive_the_section_and_rail_auto_open_lifecycle():
    """The whole busy-window cycle, driven through the REAL derivation.

    Rows come from `ConsoleAgentBridge.fleet_snapshot` via
    `_console_agent_fleet_section_state`; nothing here monkeypatches the
    payload (an earlier revision did, and that patch removed exactly the
    call that was broken). Note the summary line stays EMPTY throughout --
    this is the ordinary single-session case, which is precisely where
    keying the force on other-sessions activity misbehaved.
    """
    bridge = _FleetBridge((_running_handle("h1"),))
    async with _console_screen() as (pilot, screen):
        await _wire_fleet_bridge(pilot, screen, bridge)
        fleet_section = screen.query_one(
            "#console-agent-section-subagents", ConsoleInspectorSection
        )
        assert fleet_section.open is False
        assert screen._agent._console_agent_fleet_summary_line() == ""

        # Rows appear -> the section opens itself, once.
        await _real_fleet_sync(pilot, screen)
        assert fleet_section.rows  # real rows, from the real derivation
        assert fleet_section.open is True

        # The user collapses it while the rows persist -> stays collapsed
        # across subsequent REAL syncs (this is the case that regressed).
        fleet_section.set_open(False)
        await pilot.pause()
        assert screen._console_fleet_inspector_dismissed is True
        await _real_fleet_sync(pilot, screen)
        assert fleet_section.open is False
        await _real_fleet_sync(pilot, screen)
        assert fleet_section.open is False

        # Rows go empty -> busy window over, both flags reset.
        bridge._handles = []
        await _real_fleet_sync(pilot, screen)
        assert screen._console_fleet_rows_present is False
        assert screen._console_fleet_inspector_dismissed is False
        assert screen._console_fleet_section_auto_opened is False

        # Next busy window auto-opens again.
        fleet_section.set_open(False)
        await pilot.pause()
        bridge._handles = [_running_handle("h2")]
        await _real_fleet_sync(pilot, screen)
        assert fleet_section.open is True


@pytest.mark.asyncio
async def test_fleet_rows_auto_open_the_inspect_rail_and_refresh_it():
    """Rows appearing while the rail is closed reveal it AND refresh it."""
    bridge = _FleetBridge((_running_handle("h1"),))
    async with _console_screen() as (pilot, screen):
        await _wire_fleet_bridge(pilot, screen, bridge)
        opens: list[int] = []
        screen._console_environment.notify_rail_opened = lambda: opens.append(1)
        assert not _right_rail_open(screen)

        await _real_fleet_sync(pilot, screen)
        assert screen._console_fleet_rows_present is True
        rail_state = screen._current_console_rail_state()
        assert rail_state.right_open is True
        screen._sync_console_rail_visibility_if_changed(rail_state)
        await pilot.pause()
        assert _right_rail_open(screen)
        assert opens == [1]  # an auto-opened rail is not left on stale data

        # The user closes it during the SAME busy window -> not re-forced.
        screen._set_console_rail_preference(right_open=False)
        await pilot.pause()
        assert screen._console_fleet_inspector_dismissed is True
        assert screen._current_console_rail_state().right_open is False
        await _real_fleet_sync(pilot, screen)
        assert screen._current_console_rail_state().right_open is False


@pytest.mark.asyncio
async def test_other_session_activity_alone_does_not_touch_the_inspect_rail():
    """The other-sessions summary line drives nothing on the right rail.

    Its display target is the pinned Static beside the LEFT rail's header.
    Keying the right-rail force on it opened a rail whose fleet section was
    `display: none` -- revealing nothing.
    """
    async with _console_screen() as (pilot, screen):
        screen._agent._console_agent_fleet_summary_line = lambda: _FLEET_BUSY_LINE
        assert screen._console_fleet_rows_present is False
        assert screen._current_console_rail_state().right_open is False
