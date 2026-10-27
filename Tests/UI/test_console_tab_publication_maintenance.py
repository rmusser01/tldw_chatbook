"""Evidence-only real original publication/drain control; root runs serially."""

import asyncio
import hashlib
import inspect
import sys
import time
from pathlib import Path
from types import FunctionType

import pytest

from Tests.private_profile import private_profile_test
from Tests.UI.test_console_session_tab_close import (
    _SIZE,
    ProductionConsoleHarness,
    _mounted_console,
    _ready_app,
    _settle,
    _show_tabs,
)
from Tests.UI.test_console_tab_publication_read_barrier import (
    _OriginalReadGate,
    _shape,
    _source_code,
)


class _OriginalSurfaceEntry:
    """Observe the exact original surface method entering on one actual task."""

    def __init__(self, screen, surface):
        from tldw_chatbook.UI.Screens import chat_screen
        from tldw_chatbook.Widgets.Console import console_session_surface as module

        assert type(surface) is module.ConsoleSessionSurface
        self.surface, self.screen, self.task, self.frame = surface, screen, None, None
        self.entered = asyncio.Event()
        self.monitor, self.tool, self.active, self.restored = (
            sys.monitoring,
            None,
            False,
            False,
        )
        self.bindings, self.modules = [], []
        for defining, owner, name in (
            (module, module.ConsoleSessionSurface, "sync_sessions"),
            (chat_screen, chat_screen.ChatScreen, "_sync_console_native_session_tabs"),
            (chat_screen, chat_screen.ChatScreen, "_sync_native_console_chat_ui"),
            (
                chat_screen,
                chat_screen.ChatScreen,
                "_console_sync_maintenance_close_admission",
            ),
            (chat_screen, chat_screen.ChatScreen, "_console_sync_maintenance_drain"),
            (chat_screen, chat_screen.ChatScreen, "_console_sync_maintenance_resume"),
        ):
            function = inspect.getattr_static(owner, name)
            assert (
                type(function) is FunctionType
                and function.__globals__ is defining.__dict__
            )
            assert function.__closure__ is None
            assert _shape(function.__code__) == _shape(
                _source_code(defining, function.__qualname__)
            )
            self.bindings.append(
                (
                    owner,
                    name,
                    function,
                    function.__code__,
                    function.__globals__,
                    function.__defaults__,
                    function.__kwdefaults__,
                )
            )
            if defining not in [item[0] for item in self.modules]:
                path = Path(defining.__file__)
                self.modules.append(
                    (defining, path, hashlib.sha256(path.read_bytes()).hexdigest())
                )
        self.code = self.bindings[0][3]

    def _start(self, code, offset):
        if not self.active or asyncio.current_task() is not self.task:
            return
        frame = sys._getframe(1)
        assert frame.f_code is code
        if frame.f_locals["self"] is self.surface:
            self.frame = frame
            self.entered.set()

    def start(self):
        self.tool = next(
            slot for slot in range(6) if self.monitor.get_tool(slot) is None
        )
        self.monitor.use_tool_id(self.tool, "tldw-exact-tab-maintenance-entry")
        assert self.monitor.get_events(self.tool) == 0
        assert self.monitor.get_local_events(self.tool, self.code) == 0
        self.callback = self._start
        assert (
            self.monitor.register_callback(
                self.tool, self.monitor.events.PY_START, self.callback
            )
            is None
        )
        self.monitor.set_local_events(
            self.tool, self.code, self.monitor.events.PY_START
        )
        self.active = True

    def stop(self):
        assert self.monitor.get_events(self.tool) == 0
        assert (
            self.monitor.get_local_events(self.tool, self.code)
            == self.monitor.events.PY_START
        )
        self.monitor.set_local_events(self.tool, self.code, 0)
        assert (
            self.monitor.register_callback(
                self.tool, self.monitor.events.PY_START, None
            )
            is self.callback
        )
        self.monitor.free_tool_id(self.tool)
        self.active, self.restored = False, True
        assert all(
            inspect.getattr_static(owner, name) is function
            and function.__code__ is code
            and function.__globals__ is namespace
            and function.__defaults__ is defaults
            and function.__kwdefaults__ is kwdefaults
            and function.__closure__ is None
            for owner, name, function, code, namespace, defaults, kwdefaults in self.bindings
        )
        assert all(
            hashlib.sha256(path.read_bytes()).hexdigest() == digest
            for _, path, digest in self.modules
        )


@pytest.mark.asyncio
@pytest.mark.ui
@pytest.mark.timeout(180)
@private_profile_test
async def test_original_maintenance_drain_waits_for_coalesced_actual_tab_publication(
    request,
):
    from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend
    from tldw_chatbook.Widgets.Console.console_session_surface import (
        ConsoleSessionSurface,
    )

    app = _ready_app()
    host = ProductionConsoleHarness(app)
    read_gate = entry = None
    whole = coalesced = drain = warming = None
    lock_owned = False
    async with host.run_test(size=_SIZE) as pilot:
        console = await _mounted_console(host, pilot, "#console-native-composer")
        controller = console._ensure_console_chat_controller()
        keeper = controller.store.active_session_id
        target = controller.new_session(title="Maintenance publication control")
        controller.switch_session(keeper)
        await _show_tabs(console, pilot, {keeper, target.id})
        projection = spend.ConsoleReadinessConfigProjection.for_screen(console)
        assert (
            type(projection) is spend.ConsoleReadinessConfigProjection
            and projection.max_age == 1.0
        )
        assert await _quiet(pilot, console, projection)
        surface = console.query_one("#console-session-surface", ConsoleSessionSurface)
        assert type(surface._session_sync_lock) is asyncio.Lock
        entry = _OriginalSurfaceEntry(console, surface)
        read_gate = _OriginalReadGate(
            projection, controller.store, target.id, "coalesced"
        )
        read_gate.start()
        entry.start()
        try:
            controller.switch_session(target.id)
            # No await since `_quiet`; re-check that the switch queued
            # nothing for the screen's pump before taking the lock.
            assert _publication_quiet(console, projection)
            # Hold the actual original Surface await, because current cold
            # readiness correctly returns False instead of blocking FULL.
            await surface._session_sync_lock.acquire()
            lock_owned = True
            whole = asyncio.create_task(console._sync_native_console_chat_ui())
            entry.task = whole
            warming = asyncio.create_task(projection.warm())
            assert await _loop_wait(entry.entered.is_set)
            assert await _loop_wait(
                read_gate.entered.is_set
            ), "original stock reader not admitted"
            assert console._console_sync_in_progress
            coalesced = asyncio.create_task(console._sync_native_console_chat_ui())
            assert await _loop_wait(lambda: console._console_session_tabs_sync_calls >= 2)
            assert not coalesced.done() and surface._session_sync_lock.locked()
            console._console_sync_maintenance_close_admission()
            read_gate.release.set()
            assert not whole.done() and not coalesced.done()
            # Both original FULL callers retain their actual tab custody.
            # The separate idle-whole control proves count-only draining.
            drain = asyncio.create_task(
                console._console_sync_maintenance_drain(time.monotonic() + 2.0)
            )
            await asyncio.sleep(0)
            assert (
                not drain.done()
            ), "maintenance reported completion while original publication was blocked"
            surface._session_sync_lock.release()
            lock_owned = False
            await asyncio.gather(whole, coalesced, warming)
            assert await drain is True
            assert console._console_session_tabs_sync_calls == 0
            assert entry.frame is not None and entry.frame.f_code is entry.code
        finally:
            if read_gate is not None:
                read_gate.release.set()
            if lock_owned:
                surface._session_sync_lock.release()
            try:
                assert await _settle(
                    pilot, read_gate.reader_returned.is_set
                ), "exact released original held reader did not return"
                assert read_gate.native_calls > 0 and read_gate.completed_checked_read
                pending = [
                    task for task in (whole, coalesced, drain, warming) if task is not None
                ]
                if pending:
                    await asyncio.wait_for(
                        asyncio.gather(*pending, return_exceptions=True), 8
                    )
            finally:
                console._console_sync_maintenance_resume()
                try:
                    if entry is not None and entry.active:
                        entry.stop()
                finally:
                    if read_gate is not None and read_gate.tool is not None:
                        read_gate.stop()
        assert entry.restored and read_gate.restored
        assert read_gate.native_calls > 0 and read_gate.completed_checked_read


@pytest.mark.asyncio
@pytest.mark.ui
@pytest.mark.timeout(180)
@private_profile_test
async def test_original_maintenance_waits_for_actual_tabs_with_idle_whole_pass(request):
    """Prove tab custody directly without depending on a cold reader's return."""
    from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend
    from tldw_chatbook.Widgets.Console.console_session_surface import (
        ConsoleSessionSurface,
    )

    app = _ready_app()
    host = ProductionConsoleHarness(app)
    entry = publication = drain = None
    lock_owned = False
    async with host.run_test(size=_SIZE) as pilot:
        console = await _mounted_console(host, pilot, "#console-native-composer")
        controller = console._ensure_console_chat_controller()
        keeper = controller.store.active_session_id
        target = controller.new_session(title="Direct publication custody")
        controller.switch_session(keeper)
        await _show_tabs(console, pilot, {keeper, target.id})
        projection = spend.ConsoleReadinessConfigProjection.for_screen(console)
        assert await _quiet(pilot, console, projection)
        surface = console.query_one("#console-session-surface", ConsoleSessionSurface)
        assert type(surface._session_sync_lock) is asyncio.Lock
        entry = _OriginalSurfaceEntry(console, surface)
        entry.start()
        try:
            assert _publication_quiet(console, projection)
            await surface._session_sync_lock.acquire()
            lock_owned = True
            publication = asyncio.create_task(
                console._sync_console_native_session_tabs()
            )
            entry.task = publication
            assert await _loop_wait(
                entry.entered.is_set
            ), "actual original surface was not reached"
            assert not publication.done() and not console._console_sync_in_progress
            console._console_sync_maintenance_close_admission()
            drain = asyncio.create_task(
                console._console_sync_maintenance_drain(time.monotonic() + 2.0)
            )
            await asyncio.sleep(0)
            assert (
                not drain.done()
            ), "original drain finished while actual owned publication was blocked"
            assert getattr(console, "_console_session_tabs_sync_calls", 0) > 0
            surface._session_sync_lock.release()
            lock_owned = False
            await publication
            assert await drain is True
            assert console._console_session_tabs_sync_calls == 0
        finally:
            if lock_owned:
                surface._session_sync_lock.release()
            try:
                pending = [task for task in (publication, drain) if task is not None]
                if pending:
                    await asyncio.wait_for(asyncio.gather(*pending), 8)
            finally:
                console._console_sync_maintenance_resume()
                if entry is not None and entry.active:
                    entry.stop()
        assert entry.restored


def _awaiting(task) -> list[str]:
    """Qualified names along ``task``'s await chain, innermost last."""
    names = []
    awaitable = task.get_coro() if task is not None else None
    while (code := getattr(awaitable, "cr_code", None)) is not None:
        names.append(code.co_qualname)
        awaitable = awaitable.cr_await
    return names


def _publication_quiet(console, projection) -> bool:
    """Nothing is left that would publish tabs on the screen's own pump.

    A test may take a session lock only when this holds. TASK-36611 measured
    why "no sync in progress" is not enough: the first whole sync that
    completes the attach ends by scheduling ``_reconcile_console_after_attach``
    with ``call_after_refresh``, and that reconciliation queues one more whole
    sync the same way (``_start_console_view_after_reconciliation``). Textual
    runs every after-refresh callback on the screen's message pump and awaits
    it there. A test that took the lock in that gap parked the screen's pump
    behind its own lock and then waited on Pilot, which waits for that pump:
    ``WaitForScreenTimeout`` (dev 5b5920b8e2, 1 run in 4). A queued callback
    sits first in the screen's message queue (``InvokeLater``), then in its
    after-refresh list, then in a local copy while earlier callbacks run, so
    quiet asks for the screen's pump itself to be waiting for a message with
    none queued, besides the attach reconciled, no sync running, owed or
    replaying, and no Console worker still running.

    This covers the screen's pump only, not pending timers. Textual runs a
    ``set_timer`` callback on the timer's own task, off the screen's pump, so
    a timer cannot park the pump. These pins wait with ``_loop_wait`` while
    they hold the lock, so a timer that started a publication after the lock
    was taken would fail the pin after 10 s instead of hanging it. Nothing
    here proves that no view-start timer publishes tabs. The coalesced
    control-bar replay is the one timer path checked above, through
    ``_console_control_bar_replay_whole_sync``.

    The pump reads lean on Textual 8.x internals (``_message_queue``,
    ``_next_callbacks``, ``_callbacks``, ``_task`` and the
    ``MessagePump._get_message`` qualname). If Textual renames one, this
    raises or never holds, so ``_quiet`` fails loudly instead of passing.
    """
    return (
        console._console_attach_reconciled
        and not console._console_attach_reconcile_running
        and console._message_queue.empty()
        and not console._next_callbacks
        and not console._callbacks
        and "MessagePump._get_message" in _awaiting(console._task)
        and not console._console_sync_requested
        and not console._console_sync_in_progress
        and not console._console_session_tabs_sync_calls
        and not getattr(console, "_console_control_bar_replay_whole_sync", False)
        and not projection.pending
        and not any(
            worker.node is console and not worker.is_finished
            for worker in console.app.workers
        )
    )


async def _quiet(pilot, console, projection) -> bool:
    """Wait for ``_publication_quiet``; it still holds when this returns True."""
    return await _settle(pilot, lambda: _publication_quiet(console, projection))


async def _loop_wait(predicate, *, timeout: float = 10.0) -> bool:
    """Poll on the event loop alone, for waits made while a lock is held.

    Pilot's waits need every mounted pump, the screen's included, and a
    screen pump parked on a lock the waiter holds never answers them.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return bool(predicate())


@pytest.mark.asyncio
@pytest.mark.ui
@pytest.mark.timeout(180)
@private_profile_test
async def test_admitted_real_tab_publication_rechecks_replaced_controller(request):
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_chat_models import ConsoleRunMarker
    from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend
    from tldw_chatbook.Widgets.Console.console_session_surface import (
        ConsoleSessionSurface,
    )

    app = _ready_app()
    host = ProductionConsoleHarness(app)
    publication = replacement = entry = None
    lock_owned = False
    async with host.run_test(size=_SIZE) as pilot:
        console = await _mounted_console(host, pilot, "#console-native-composer")
        original = console._ensure_console_chat_controller()
        store = original.store
        session_id = store.active_session_id
        await _show_tabs(console, pilot, {session_id})
        projection = spend.ConsoleReadinessConfigProjection.for_screen(console)
        assert await _quiet(pilot, console, projection)
        surface = console.query_one("#console-session-surface", ConsoleSessionSurface)
        entry = _OriginalSurfaceEntry(console, surface)
        entry.start()
        try:
            # `_quiet` returned with nothing queued and no await since, so
            # this acquire does not suspend and no other publication can be
            # waiting for the lock.
            assert _publication_quiet(console, projection)
            await surface._session_sync_lock.acquire()
            lock_owned = True
            publication = asyncio.create_task(console._sync_console_native_session_tabs())
            entry.task = publication
            assert await _loop_wait(entry.entered.is_set), (
                "publication never reached the surface await; another publication "
                "holds the tab lock: "
                f"{console._console_session_tabs_sync_calls} tab publications open"
            )
            assert not publication.done()
            assert original.run_marker_for(session_id) is ConsoleRunMarker.NONE
            console._console_sync_maintenance_close_admission()
            replacement = ConsoleChatController(store=store, provider_gateway=original.provider_gateway)
            replacement.add_pending_round(session_id, "replacement-round")
            assert replacement.run_marker_for(session_id) is ConsoleRunMarker.NEEDS_APPROVAL
            console._console_runtime().set_chat_controller(replacement)
            assert console._console_chat_store is store
            assert console._console_chat_controller is replacement
            surface._session_sync_lock.release()
            lock_owned = False
            await asyncio.wait_for(publication, 8)
            button = surface.query_one(f"#console-session-tab-{session_id}")
            assert "waiting for approval" in str(button.tooltip), str(button.tooltip)
            assert console._console_session_tabs_sync_calls == 0
            # Released, the screen answers Pilot again: a stall would raise here.
            await pilot.pause()
        finally:
            if lock_owned:
                surface._session_sync_lock.release()
            try:
                if publication is not None:
                    await asyncio.wait_for(publication, 8)
            finally:
                console._console_runtime().set_chat_controller(original)
                if replacement is not None:
                    replacement.discard_pending_round(session_id, "replacement-round")
                    await replacement.shutdown()
                console._console_sync_maintenance_resume()
                if entry.active:
                    entry.stop()
        assert entry.restored
