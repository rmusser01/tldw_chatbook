"""A run tick pays for the Conversations browser build at most every other poll tick.

TASK-36571.1 (measured live 2026-11-05, dev 29592ecc9b, Anthropic haiku through
the review proxy, 160x45): every projection built the whole workspace context
(the Conversations browser, the workspace tree, canonical-owner bookkeeping)
to read one string, ``workspace_label``, for the rail badge and the
Inspector's conversation rows; the Agent section built the whole rail state,
Inspector included, to read one flag; and the Conversations tray push rebuilt
the browser on every 0.2 s tick to find it unchanged. These tests pin each
cheaper component (they fail on that base) and guard what the rails show:

* a projection, and a control-bar sync, build no workspace context;
* the label read matches the build's label in every registry state;
* the Agent section's sync builds no rail state, and its open flag equals
  the rail state's in every case that rule distinguishes;
* settled poll ticks build on alternate ticks, never twice in a row from one
  build, and never over a push made outside the tick;
* an input the fingerprint covers reaches the tray on the next tick; one it
  leaves out reaches it within one more tick.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from Tests.UI.test_console_dictation import _ready_host
from Tests.UI.test_console_run_tick_workspace_reads import APP_SIZE, _settled_console
from tldw_chatbook.UI.Console_Modules import config_sync
from tldw_chatbook.UI.Console_Modules import workspace as workspace_module
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console.console_workspace_context import (
    ConsoleWorkspaceContextTray,
)
from tldw_chatbook.Workspaces.display_state import build_console_workspace_state

pytestmark = pytest.mark.bootstrap_profile


def _count_builds(monkeypatch) -> list[int]:
    """Count real workspace-context builds at their uncached seam."""
    builds: list[int] = []
    original = workspace_module.build_console_workspace_state

    def counting(**kwargs):
        builds.append(1)
        return original(**kwargs)

    monkeypatch.setattr(workspace_module, "build_console_workspace_state", counting)
    return builds


class _CallCounter:
    """Count calls of given code objects with a profile hook (no rebinding)."""

    def __init__(self, *functions) -> None:
        self._codes = {function.__code__: function.__name__ for function in functions}
        self.calls: dict[str, int] = dict.fromkeys(self._codes.values(), 0)

    def __enter__(self) -> "_CallCounter":
        self._previous = sys.getprofile()

        def observe(frame, event, _argument):
            if event == "call":
                name = self._codes.get(frame.f_code)
                if name is not None:
                    self.calls[name] += 1

        sys.setprofile(observe)
        return self

    def __exit__(self, *_exc) -> None:
        sys.setprofile(self._previous)


async def _poll_tick(console):
    """Run one sync the way the 0.2 s poll runs it; False when it deferred."""
    return await config_sync.run_periodic_console_sync(
        console._sync_native_console_chat_ui
    )


async def _completed_poll_tick(console, pilot, attempts: int = 8) -> None:
    """Run poll ticks until one completes (a cold checked display defers some)."""
    for _ in range(attempts):
        if await _poll_tick(console) is True:
            return
        await pilot.pause()
    raise AssertionError("no poll tick completed")


async def _fresh_build_then_poll_tick(console, pilot, builds) -> int:
    """A completed explicit sync, then at once a completed poll tick.

    Returns the poll tick's own builds. Retries the pair when either sync
    defers (a cold checked display hands it to a whole-sync replay).
    """
    for _ in range(8):
        await pilot.pause()
        if await console._sync_native_console_chat_ui() is not True:
            continue
        before = len(builds)
        if await _poll_tick(console) is True:
            return len(builds) - before
    raise AssertionError("no explicit sync and poll tick completed back to back")


def _tray(console) -> ConsoleWorkspaceContextTray:
    return console.query_one("#console-workspace-context", ConsoleWorkspaceContextTray)


def _chats_collapsed(state) -> bool:
    browser = state.conversation_browser
    section = next(s for s in browser.sections if s.section_id == "chats")
    return bool(section.collapsed)


# ---------------------------------------------------------------------------
# 1. The projection reads the label, not the build
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_run_tick_projection_builds_no_workspace_context(monkeypatch):
    """The rail badge and the Inspector rows read ``workspace_label`` alone."""
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console = await _settled_console(host, pilot)
        builds = _count_builds(monkeypatch)
        for _ in range(8):  # A cold checked display defers a projection.
            await pilot.pause()
            builds.clear()
            with console._workspace.tick_workspace_build_scope():
                if console._sync_console_rail_and_controls() is True:
                    break
        else:
            raise AssertionError("no projection completed")
        assert builds == [], (
            f"the projection built the workspace context {len(builds)}x"
        )


@pytest.mark.asyncio
async def test_a_control_bar_sync_builds_no_workspace_context(monkeypatch):
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console = await _settled_console(host, pilot)
        builds = _count_builds(monkeypatch)
        for _ in range(8):  # A cold checked display defers the sync.
            await pilot.pause()
            builds.clear()
            if console._sync_console_control_bar() is True:
                break
        else:
            raise AssertionError("no control-bar sync completed")
        assert builds == [], (
            f"a control-bar sync built the workspace context {len(builds)}x"
        )


class _Registry:
    def __init__(self, active):
        self._active = active

    def get_active_workspace(self):
        if isinstance(self._active, Exception):
            raise self._active
        return self._active

    def list_workspaces(self):
        return ()

    def list_workspace_memberships(self, *_args, **_kwargs):
        return ()

    def list_runtime_bindings(self, *_args, **_kwargs):
        return ()


@pytest.mark.parametrize(
    "registry",
    [
        None,
        _Registry(RuntimeError("unreadable")),
        _Registry(None),
        "active",
    ],
    ids=["no-registry", "unreadable", "no-active-workspace", "active"],
)
def test_the_label_read_matches_the_build_in_every_registry_state(registry):
    from tldw_chatbook.Workspaces.display_state import console_workspace_label

    if registry == "active":
        from tldw_chatbook.Workspaces.models import WorkspaceRecord

        record = WorkspaceRecord(workspace_id="workspace-7", name="Seven")
        registry = _Registry(record)
    state = build_console_workspace_state(
        registry_service=registry, current_conversation=None
    )
    assert console_workspace_label(registry) == state.workspace_label


@pytest.mark.asyncio
async def test_the_rails_show_the_builds_label_after_a_workspace_switch():
    """The label read follows a registry change the next sync, like the build."""
    app, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console = await _settled_console(host, pilot)
        registry = app.workspace_registry_service
        registry.create_workspace(workspace_id="workspace-gamma", name="Gamma")
        registry.set_active_workspace("workspace-gamma")
        await console._sync_native_console_chat_ui()
        await pilot.pause()

        build_label = (
            console._workspace._build_console_workspace_context_state().workspace_label
        )
        assert build_label == "Workspace: Gamma"
        assert console._workspace.console_workspace_label() == build_label
        rows = {
            row.label: row.value
            for row in console._selected_console_conversation_inspector_rows()
        }
        assert rows["Workspace"] == "Gamma"
        assert console._current_console_rail_state().left_badge == (
            console._build_console_rail_state(
                staged_context_state=console._build_console_staged_context_state(None),
                inspector_state=console._build_console_inspector_state(None),
                workspace_label=build_label,
            ).left_badge
        )


# ---------------------------------------------------------------------------
# 2. The Agent section reads its open flag, not the rail state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_agent_section_sync_builds_no_rail_state():
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console = await _settled_console(host, pilot)
        console._console_agent_section_last = None  # Force the full derivation.
        with _CallCounter(
            ChatScreen._current_console_rail_state,
            ChatScreen._build_console_inspector_state,
        ) as counter:
            console._sync_console_agent_section()
        assert counter.calls == {
            "_current_console_rail_state": 0,
            "_build_console_inspector_state": 0,
        }, counter.calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stored_open", "fleet_line", "dismissed"),
    [
        (False, "", False),
        (True, "", False),
        (False, "", True),
        (False, "1 running", False),
        (True, "1 running", True),
        (False, "1 running", True),
    ],
)
async def test_the_agent_section_open_flag_equals_the_rail_states(
    stored_open, fleet_line, dismissed
):
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console = await _settled_console(host, pilot)
        console._set_console_rail_preference(
            section_updates={"agent": stored_open}, notify_on_failure=False
        )
        console._agent._console_agent_fleet_summary_line = lambda: fleet_line
        console._agent_section_user_dismissed_while_busy = dismissed
        flag = console._current_console_agent_section_open()
        console._agent_section_user_dismissed_while_busy = dismissed
        assert flag is console._current_console_rail_state().agent_open
        assert flag is (stored_open or bool(fleet_line and not dismissed))


# ---------------------------------------------------------------------------
# 3. The tray push reuses the previous tick's build, once
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_poll_tick_reuses_the_build_its_predecessor_made(monkeypatch):
    """The tray push finds the previous tick's build unchanged and reuses it."""
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console = await _settled_console(host, pilot)
        builds = _count_builds(monkeypatch)
        assert await _fresh_build_then_poll_tick(console, pilot, builds) == 0
        # The reuse is spent: the next completed tick builds.
        before = len(builds)
        result = await _poll_tick(console)
        if result is True:
            assert len(builds) - before == 1


@pytest.mark.asyncio
async def test_an_explicit_sync_never_reuses_a_carried_build(monkeypatch):
    """Only the poll's own tick reuses; a send transition's full sync builds."""
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console = await _settled_console(host, pilot)
        builds = _count_builds(monkeypatch)
        completed = 0
        for _ in range(8):
            await pilot.pause()
            before = len(builds)
            if await console._sync_native_console_chat_ui() is True:
                completed += 1
                assert len(builds) - before >= 1
        assert completed >= 2


@pytest.mark.asyncio
async def test_a_push_made_outside_the_tick_is_not_overwritten_by_the_carried_build():
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console = await _settled_console(host, pilot)
        tray = _tray(console)
        for _ in range(8):
            await pilot.pause()
            if await console._sync_native_console_chat_ui() is True:
                break  # Its build is the one the next poll tick could reuse.
        assert not _chats_collapsed(tray.state)

        # A handler's push: the collapse preference is outside the fingerprint.
        console._workspace._toggle_console_conversation_browser_section("section:chats")
        assert _chats_collapsed(tray.state)

        await _completed_poll_tick(console, pilot)
        assert _chats_collapsed(tray.state), "the tick pushed the older carried build"


@pytest.mark.asyncio
async def test_an_input_the_fingerprint_leaves_out_reaches_the_tray_within_one_tick():
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console = await _settled_console(host, pilot)
        tray = _tray(console)
        for _ in range(8):
            await pilot.pause()
            if await console._sync_native_console_chat_ui() is True:
                break
        assert not _chats_collapsed(tray.state)

        # Written with no push: only a later tick's build can show it.
        console._workspace._set_console_conversation_browser_group_collapsed(
            "section:chats", True
        )
        await _completed_poll_tick(console, pilot)  # Base shows it; this may reuse.
        await _completed_poll_tick(console, pilot)  # Never two reuses in a row.
        assert _chats_collapsed(tray.state)


@pytest.mark.asyncio
async def test_an_input_the_fingerprint_covers_reaches_the_tray_on_the_next_tick():
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console = await _settled_console(host, pilot)
        for _ in range(8):
            await pilot.pause()
            if await console._sync_native_console_chat_ui() is True:
                break
        store = console._console_chat_store
        session = store.sessions()[0]
        store.rename_session(session.id, "Renamed mid-run")
        await _completed_poll_tick(console, pilot)
        titles = {
            row.title
            for section in _tray(console).state.conversation_browser.sections
            for row in section.rows
        }
        assert "Renamed mid-run" in titles, titles


def test_a_carried_build_is_taken_once_and_only_by_the_next_tick(monkeypatch):
    controller = SimpleNamespace(_console_carried_workspace_build=None)
    take = workspace_module.ConsoleWorkspaceController._take_carried_workspace_build
    clock = [100.0]
    monkeypatch.setattr(workspace_module.time, "monotonic", lambda: clock[0])
    state = object()

    def carry(serial, at=100.0, fingerprint=("f",)):
        controller._console_carried_workspace_build = (serial, at, fingerprint, state)

    carry(4)
    assert take(controller, ("f",), 5) is state
    assert take(controller, ("f",), 5) is None  # Taken once.
    assert take(controller, ("f",), 6) is None  # A reuse is never reused.
    carry(4)
    assert take(controller, ("f",), 6) is None  # Not the next tick.
    carry(4)
    assert take(controller, ("g",), 5) is None  # Inputs moved.
    carry(
        4, at=100.0 - workspace_module.CONSOLE_TICK_BUILD_CARRY_MAX_AGE_SECONDS - 0.01
    )
    assert take(controller, ("f",), 5) is None  # A poll that stopped and restarted.
    carry(4)
    assert take(controller, ("f",), None) is None
