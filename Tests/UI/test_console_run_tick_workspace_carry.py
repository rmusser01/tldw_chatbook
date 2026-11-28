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
from tldw_chatbook.Chat.console_chat_models import ConsoleRunState, ConsoleRunStatus
from tldw_chatbook.UI.Console_Modules import config_sync
from tldw_chatbook.UI.Console_Modules import workspace as workspace_module
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console.console_workspace_context import (
    ConsoleWorkspaceContextTray,
)
from tldw_chatbook.Workspaces.display_state import build_console_workspace_state

pytestmark = pytest.mark.bootstrap_profile


def _count_builds(monkeypatch, console) -> list[int]:
    """Count real workspace-context builds: each one runs the browser pipeline once."""
    builds: list[int] = []
    controller = console._workspace
    original = controller._with_console_conversation_browser_state

    def counting(*args, **kwargs):
        builds.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        controller, "_with_console_conversation_browser_state", counting
    )
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


async def _fresh_build_then_poll_tick(console, pilot, builds, between=None) -> int:
    """A completed explicit sync, then at once a completed poll tick.

    Returns the poll tick's own builds. Retries the pair when either sync
    defers (a cold checked display hands it to a whole-sync replay).

    Args:
        between: Called after the explicit sync and before the poll tick.
    """
    for _ in range(8):
        await pilot.pause()
        start = len(builds)
        if (
            await console._sync_native_console_chat_ui() is not True
            or len(builds) == start
        ):
            continue  # Deferred, or it reused its own predecessor's build.
        if between is not None:
            between()
        before = len(builds)
        if await _poll_tick(console) is True:
            return len(builds) - before
    raise AssertionError("no explicit sync and poll tick completed back to back")


def _set_run_status(console, status: ConsoleRunStatus) -> None:
    controller = console._ensure_console_chat_controller()
    controller._set_run_state(ConsoleRunState(status, f"{status.value} (test)"))


def _row_markers(console) -> set[str]:
    browser = _tray(console).state.conversation_browser
    return {row.run_marker for section in browser.sections for row in section.rows}


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
        builds = _count_builds(monkeypatch, console)
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
        builds = _count_builds(monkeypatch, console)
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
        builds = _count_builds(monkeypatch, console)
        assert await _fresh_build_then_poll_tick(console, pilot, builds) == 0
        # The reuse is spent: the next completed tick builds.
        for _ in range(8):  # A cold checked display defers some ticks.
            before = len(builds)
            if await _poll_tick(console) is True:
                break
            await pilot.pause()
        else:
            raise AssertionError("no poll tick completed after the reuse")
        assert len(builds) - before == 1


@pytest.mark.asyncio
async def test_a_run_status_flip_that_moves_no_marker_still_reuses_the_build(
    monkeypatch,
):
    """A send flips VALIDATING -> STREAMING; the browser shows the same marker."""
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console = await _settled_console(host, pilot)
        _set_run_status(console, ConsoleRunStatus.VALIDATING)
        builds = _count_builds(monkeypatch, console)
        reused = await _fresh_build_then_poll_tick(
            console,
            pilot,
            builds,
            between=lambda: _set_run_status(console, ConsoleRunStatus.STREAMING),
        )
        assert reused == 0


@pytest.mark.asyncio
async def test_a_run_status_flip_that_moves_a_marker_rebuilds_on_the_next_tick(
    monkeypatch,
):
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console = await _settled_console(host, pilot)
        builds = _count_builds(monkeypatch, console)
        idle_markers = None

        def start_run():
            nonlocal idle_markers
            idle_markers = _row_markers(console)
            _set_run_status(console, ConsoleRunStatus.STREAMING)

        assert await _fresh_build_then_poll_tick(console, pilot, builds, start_run) == 1
        assert _row_markers(console) != idle_markers  # The running marker shows.


@pytest.mark.asyncio
async def test_a_send_transitions_full_sync_reuses_its_predecessors_build_once(
    monkeypatch,
):
    """Explicit full syncs are run ticks too: the second of two reuses the first's."""
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console = await _settled_console(host, pilot)
        builds = _count_builds(monkeypatch, console)
        for _ in range(8):
            await pilot.pause()
            start = len(builds)
            if (
                await console._sync_native_console_chat_ui() is not True
                or len(builds) == start
            ):
                continue  # Deferred, or it reused its own predecessor's build.
            before = len(builds)
            if await console._sync_native_console_chat_ui() is True:
                assert len(builds) - before == 0
                before = len(builds)
                if await console._sync_native_console_chat_ui() is True:
                    assert len(builds) - before >= 1  # Never two reuses in a row.
                break
        else:
            raise AssertionError("no two full syncs completed back to back")


@pytest.mark.asyncio
async def test_an_unmoved_registry_view_reuses_the_workspace_state_builder(monkeypatch):
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console = await _settled_console(host, pilot)
        calls = []
        original = workspace_module.build_console_workspace_state

        def counting(**kwargs):
            calls.append(1)
            return original(**kwargs)

        monkeypatch.setattr(workspace_module, "build_console_workspace_state", counting)
        controller = console._workspace
        first = controller._build_console_workspace_context_state()
        second = controller._build_console_workspace_context_state()
        assert len(calls) <= 1, f"the builder ran {len(calls)}x for an unmoved view"
        assert second == first


@pytest.mark.asyncio
async def test_the_builder_memo_follows_a_registry_rename():
    app, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console = await _settled_console(host, pilot)
        registry = app.workspace_registry_service
        controller = console._workspace
        registry.create_workspace(workspace_id="workspace-memo", name="Memo")
        registry.set_active_workspace("workspace-memo")
        assert (
            controller._build_console_workspace_context_state().workspace_name == "Memo"
        )
        registry.rename_workspace("workspace-memo", "Memo Renamed")
        assert (
            controller._build_console_workspace_context_state().workspace_name
            == "Memo Renamed"
        )


def test_the_view_display_key_moves_with_what_its_reads_return():
    class Service:
        mutation_generation = 1

        def get_active_workspace(self):
            return "active"

        def list_workspaces(self):
            return ("w",)

    service = Service()
    view = workspace_module._ConsoleRegistryDisplayReads(service)
    key = view.display_key()
    assert view.display_key() == key  # Nothing read: unmoved.
    view.get_active_workspace()
    filled = view.display_key()
    assert filled != key  # A live read filled the cache.
    view.get_active_workspace()
    assert view.display_key() == filled  # A cached read moves nothing.
    service.mutation_generation = 2
    renewed = view.display_key()
    assert renewed != filled  # A new generation.
    view.stale_reader = lambda *_args: None
    assert view.display_key() != renewed  # Stale reads are their own key.
    view.stale_reader = None
    assert view.install(2, {("workspaces",): ("w",)}) is True
    assert view.display_key() != renewed  # An installed re-read.
    service.mutation_generation = True  # A double: caching does not apply.
    assert view.display_key() is None


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
    controller = SimpleNamespace(
        _console_carried_workspace_build=None,
        _console_workspace_carry_key=lambda fingerprint: fingerprint,
    )
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
    carry(4, fingerprint=None)
    assert take(controller, ("f",), 5) is None  # An unreadable run display.


@pytest.mark.asyncio
async def test_an_unchanged_folder_availability_reread_pushes_nothing():
    """The once-a-second availability re-read found nothing new and still pushed."""
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console = await _settled_console(host, pilot)
        controller = console._workspace
        pushes = []
        original = controller._sync_workspace_context_fn
        controller._sync_workspace_context_fn = lambda: (pushes.append(1), original())
        try:
            for _ in range(2):
                controller._workspace_files_availability_cached_at = 0.0
                controller._request_workspace_files_availability_refresh(
                    (workspace_module.DEFAULT_WORKSPACE_ID,)
                )
                await host.workers.wait_for_complete()
                await pilot.pause()
            before = dict(controller._workspace_files_availability_by_id)
            assert len(pushes) <= 1, f"an unchanged re-read pushed {len(pushes) - 1}x"
            pushed = len(pushes)

            # A changed snapshot still pushes.
            controller._workspace_files_availability_by_id = (
                workspace_module.MappingProxyType(
                    {key: not value for key, value in before.items()}
                )
            )
            controller._workspace_files_availability_cached_at = 0.0
            controller._request_workspace_files_availability_refresh(
                (workspace_module.DEFAULT_WORKSPACE_ID,)
            )
            await host.workers.wait_for_complete()
            await pilot.pause()
            assert len(pushes) == pushed + 1
            assert dict(controller._workspace_files_availability_by_id) == before
        finally:
            controller._sync_workspace_context_fn = original


@pytest.mark.asyncio
async def test_a_cached_label_read_enters_no_stale_read_context(monkeypatch):
    """The label read the projection makes several times a tick is a dict lookup."""
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console = await _settled_console(host, pilot)
        controller = console._workspace
        entered = []
        original = controller._stale_display_reads

        def counting():
            entered.append(1)
            return original()

        monkeypatch.setattr(controller, "_stale_display_reads", counting)
        with controller.tick_workspace_build_scope():
            label = controller.console_workspace_label()  # May fill the cache once.
            entered.clear()
            assert controller.console_workspace_label() == label
            assert entered == [], "a cached label read entered the stale-read context"


def test_a_reused_builder_result_still_asks_again_for_a_failed_stale_reread():
    """A stale read asks for its off-loop re-read on every build; a reuse must too."""

    class Service:
        mutation_generation = 1

        def get_active_workspace(self):
            return "active"

    service = Service()
    view = workspace_module._ConsoleRegistryDisplayReads(service)
    view.get_active_workspace()  # Cached for generation 1.
    service.mutation_generation = 2  # The cache becomes the previous generation's.
    asked = []
    view.stale_reader = lambda reads, generation: asked.append(generation)
    view.renew_stale_requests()
    assert asked == [2]
    view.stale_reader = None
    view.renew_stale_requests()  # Live reads: nothing to ask for.
    assert asked == [2]
    view.stale_reader = lambda reads, generation: asked.append(generation)
    assert view.install(2, {("active",): "active"}) is True
    view.renew_stale_requests()  # Re-read landed: nothing stale left.
    assert asked == [2]


@pytest.mark.asyncio
async def test_a_reused_workspace_state_builder_renews_stale_requests(monkeypatch):
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console = await _settled_console(host, pilot)
        renewed = []
        monkeypatch.setattr(
            workspace_module._ConsoleRegistryDisplayReads,
            "renew_stale_requests",
            lambda self: renewed.append(self),
        )
        controller = console._workspace
        controller._build_console_workspace_context_state()
        renewed.clear()
        controller._build_console_workspace_context_state()  # The builder's memo hit.
        assert renewed == [controller._console_registry_reads_view()]


@pytest.mark.asyncio
async def test_the_builder_memo_follows_a_conversation_switch():
    """The memo is keyed on the current conversation too: the selected-chat
    row's 'Conversation id' tooltip (``scope_detail``) shows the new id."""
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console = await _settled_console(host, pilot)
        controller = console._workspace
        original = controller._current_conversation_id_accessor
        try:
            for conversation_id in ("conv-memo-1", "conv-memo-2"):
                controller._current_conversation_id_accessor = (
                    lambda conversation_id=conversation_id: conversation_id
                )
                state = controller._build_console_workspace_context_state()
                assert state.scope_detail == conversation_id
        finally:
            controller._current_conversation_id_accessor = original


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["server adapter", "acp handoff", "folder bindings"])
async def test_the_builder_memo_follows_each_of_its_other_inputs(changed):
    from tldw_chatbook.Workspaces.display_state import (
        ConsoleWorkspaceACPHandoffState,
        ConsoleWorkspaceServerAdapterState,
    )

    app, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console = await _settled_console(host, pilot)
        controller = console._workspace
        controller._build_console_workspace_context_state()  # The memo is warm.
        with _CallCounter(build_console_workspace_state) as counter:
            if changed == "server adapter":
                app.workspace_server_adapter_state = ConsoleWorkspaceServerAdapterState(
                    available=False, detail="Adapter down (test)."
                )
                state = controller._build_console_workspace_context_state()
                assert state.server_readiness_label == "Server: unavailable"
            elif changed == "acp handoff":
                app.workspace_acp_handoff_state = ConsoleWorkspaceACPHandoffState(
                    status="ready", detail="Ready (test)."
                )
                state = controller._build_console_workspace_context_state()
                assert state.acp_handoff_label == "ACP task/run: ready"
            else:
                controller._workspace_files_runtime_bindings_by_id = (
                    workspace_module.MappingProxyType(
                        dict(controller._workspace_files_runtime_bindings_by_id)
                    )
                )
                controller._build_console_workspace_context_state()
        assert counter.calls == {"build_console_workspace_state": 1}, (
            f"a changed {changed} reused the memo: {counter.calls}"
        )


# ---------------------------------------------------------------------------
# 4. Outside a run tick, the cheaper reads still ask for the build's re-reads
# ---------------------------------------------------------------------------


def _run_idle_sync(console, sync: str) -> bool:
    """Run one of the syncs an idle Console runs; False when it deferred."""
    if sync == "settings summary":
        console._sync_console_settings_summary()
        return True
    if sync == "control bar":
        return console._sync_console_control_bar() is True
    console._console_agent_section_last = None  # Force the full derivation.
    console._sync_console_agent_section()
    return True


@pytest.mark.asyncio
@pytest.mark.parametrize("sync", ["settings summary", "control bar", "agent section"])
async def test_an_idle_sync_past_the_ttls_still_rereads_folder_availability_and_saved_rows(
    sync,
):
    """TASK-36571.1 review: idle, each of these syncs built the workspace context
    on the base, and each build asked for the folder-availability snapshot (1 s
    TTL) and the saved Conversations rows (10 s TTL) once they aged. The label
    and open-flag reads that replaced those builds asked for neither, so an idle
    Console showed a deleted folder, or a conversation written elsewhere, only
    once some event-driven sync built.
    """
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console = await _settled_console(host, pilot)
        controller = console._workspace
        for _ in range(8):  # A cold checked display defers the control-bar sync.
            await host.workers.wait_for_complete()
            await pilot.pause()
            if (
                controller._workspace_files_availability_refresh_in_flight
                or controller._console_persisted_rows_refresh_key is not None
            ):
                continue  # A re-read some other sync asked for is still landing.
            # As if the Console had sat idle past both TTLs.
            controller._workspace_files_availability_cached_at = 0.0
            controller._console_persisted_rows_cache_at = 0.0
            if _run_idle_sync(console, sync):
                break
        else:
            raise AssertionError(f"no {sync} sync completed")
        asked = {
            "folder availability": (
                controller._workspace_files_availability_refresh_in_flight
            ),
            "saved rows": controller._console_persisted_rows_refresh_key is not None,
        }
        assert asked == {"folder availability": True, "saved rows": True}, (
            f"the idle {sync} sync asked for {asked}"
        )
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert controller._workspace_files_availability_cached_at > 0.0
        assert controller._console_persisted_rows_cache_at > 0.0


@pytest.mark.asyncio
async def test_a_run_ticks_label_and_open_flag_reads_ask_for_no_reread():
    """Inside a run tick the tick's own build asks (or a reuse of a build that
    just asked), so the projection's several cheap reads add nothing."""
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console = await _settled_console(host, pilot)
        controller = console._workspace
        await host.workers.wait_for_complete()
        await pilot.pause()
        controller._workspace_files_availability_cached_at = 0.0
        controller._console_persisted_rows_cache_at = 0.0
        with _CallCounter(
            workspace_module.ConsoleWorkspaceController._request_workspace_files_availability_refresh,
            workspace_module.ConsoleWorkspaceController._sync_persisted_console_browser_rows,
        ) as counter:
            with controller.tick_workspace_build_scope():
                controller.console_workspace_label()
                console._current_console_agent_section_open()
        assert counter.calls == {
            "_request_workspace_files_availability_refresh": 0,
            "_sync_persisted_console_browser_rows": 0,
        }, counter.calls


@pytest.mark.asyncio
async def test_an_idle_read_rereads_the_browsers_aged_subagent_counts(tmp_path):
    """The third re-read a build asks for once it ages: the sub-agent counts of
    the rows the last count read covered (TASK-36571.1 review). A real bridge
    over a real run store takes the off-loop path, whose landing publishes."""
    from Tests.UI.test_console_agent_controller import (
        _bridge_over,
        _seed_done_primary_with_subagents,
    )

    db_path = tmp_path / "agent_runs.db"
    _seed_done_primary_with_subagents(db_path, tasks=("t1",))
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console = await _settled_console(host, pilot)
        bridge = _bridge_over(db_path)
        console._console_agent_bridge = bridge
        queries = []
        original = bridge._db.count_subagents_by_conversation

        def counting(ids):
            if sorted(ids) == ["conv-A"]:  # Not the mounted browser's own rows.
                queries.append(sorted(ids))
            return original(ids)

        bridge._db.count_subagents_by_conversation = counting
        rows = (SimpleNamespace(conversation_id="conv-A"),)
        agent = console._agent
        workspace = console._workspace
        agent._console_subagent_counts_for_rows(bridge, rows)
        await host.workers.wait_for_complete()
        await pilot.pause()
        await host.workers.wait_for_complete()
        assert queries == [["conv-A"]]
        # The last count read (a build's): fresh, so nothing is re-read.
        assert agent._console_subagent_counts_for_rows(bridge, rows) == {"conv-A": 1}
        state = agent._console_subagent_counts_read[frozenset({"conv-A"})]
        workspace.request_due_console_context_rereads()
        assert state["pending"] is False, "fresh counts were re-read"

        state["at"] = 0.0  # Aged past the TTL.
        workspace.request_due_console_context_rereads()
        assert agent._console_subagent_counts_read[frozenset({"conv-A"})]["pending"]
        await host.workers.wait_for_complete()
        assert queries == [["conv-A"], ["conv-A"]], "the aged counts were not re-read"

        agent._console_subagent_counts_for_rows(bridge, rows)
        agent._console_subagent_counts_read[frozenset({"conv-A"})]["at"] = 0.0
        console._console_agent_bridge = _bridge_over(db_path)  # A new bridge.
        workspace.request_due_console_context_rereads()
        await host.workers.wait_for_complete()
        assert len(queries) == 2, "the replaced bridge's rows were re-read"
