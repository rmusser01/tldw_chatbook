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
    ProductionConsoleHarness,
    _SIZE,
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
    whole = coalesced = drain = None
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
        assert await _settle(
            pilot,
            lambda: not projection.pending
            and not console._console_sync_in_progress
            and not getattr(console, "_console_control_bar_replay_whole_sync", False),
        )
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
            whole = asyncio.create_task(console._sync_native_console_chat_ui())
            assert await _settle(
                pilot, read_gate.entered.is_set
            ), "original stock reader not admitted"
            assert console._console_sync_in_progress
            # Hold the actual original surface lock, never replace its method.
            await surface._session_sync_lock.acquire()
            lock_owned = True
            coalesced = asyncio.create_task(console._sync_native_console_chat_ui())
            entry.task = coalesced
            assert await _settle(
                pilot, entry.entered.is_set
            ), "coalesced original caller returned before actual tab publication"
            assert not coalesced.done() and surface._session_sync_lock.locked()
            assert getattr(console, "_console_session_tabs_sync_calls", 0) > 0
            console._console_sync_maintenance_close_admission()
            read_gate.release.set()
            assert await _settle(
                pilot, whole.done
            ), "older admitted whole pass did not finish"
            await whole
            assert not console._console_sync_in_progress
            assert (
                not coalesced.done()
                and getattr(console, "_console_session_tabs_sync_calls", 0) > 0
            )
            # The old boolean is now False. Only the real admitted publication
            # count can keep the unchanged original drain awaiting completion.
            drain = asyncio.create_task(
                console._console_sync_maintenance_drain(time.monotonic() + 2.0)
            )
            await asyncio.sleep(0)
            assert (
                not drain.done()
            ), "maintenance reported completion while original publication was blocked"
            surface._session_sync_lock.release()
            lock_owned = False
            await coalesced
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
                    task for task in (whole, coalesced, drain) if task is not None
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
        assert await _settle(
            pilot,
            lambda: not projection.pending
            and not console._console_sync_in_progress
            and not getattr(console, "_console_control_bar_replay_whole_sync", False),
        )
        surface = console.query_one("#console-session-surface", ConsoleSessionSurface)
        assert type(surface._session_sync_lock) is asyncio.Lock
        entry = _OriginalSurfaceEntry(console, surface)
        entry.start()
        try:
            await surface._session_sync_lock.acquire()
            lock_owned = True
            publication = asyncio.create_task(
                console._sync_console_native_session_tabs()
            )
            entry.task = publication
            assert await _settle(
                pilot, entry.entered.is_set
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
