"""A tab publication holding the session locks lets go without the screen's pump.

TASK-36611. A whole sync queued with ``call_after_refresh`` (the attach
reconciliation's view start, ``ChatScreen._start_console_view_after_
reconciliation``) runs on the Console screen's own message pump: Textual
awaits every after-refresh callback there. When another publication holds the
tab locks at that moment -- here the worker whole sync that the real
maintenance resume replays -- the screen's pump waits on
``_console_session_tabs_sync_lock`` until that publication is done, and the
screen processes no messages meanwhile. That is a short wait, not a hang,
only while the holder's critical section (the tab removals and mounts in
``ConsoleSessionSurface.sync_sessions``) can finish without the screen's
pump. This test parks the screen's pump behind a real holder first, then lets
the holder do its real remount while the test holds nothing. A holder that
waits on the screen's pump inside the lock (checked with a temporary
``screen.call_later`` wait in ``sync_sessions``) fails it after 10 s.

``test_console_tab_publication_maintenance.py`` once timed out because the
test itself held the surface lock while waiting on Pilot, which waits for the
parked screen pump: a circular wait only a test can make. No production code
takes the session locks except the tab publication itself.
"""

import time

import pytest

from Tests.private_profile import private_profile_test
from Tests.UI.test_console_session_tab_close import (
    _SIZE,
    ProductionConsoleHarness,
    _mounted_console,
    _open_tab_ids,
    _ready_app,
    _show_tabs,
)
from Tests.UI.test_console_tab_publication_maintenance import (
    _awaiting,
    _loop_wait,
    _quiet,
)

_AWAITING_LOCK = ["_ContextManagerMixin.__aenter__", "Lock.acquire"]


@pytest.mark.asyncio
@pytest.mark.ui
@pytest.mark.timeout(180)
@private_profile_test
async def test_tab_holder_finishes_while_the_screen_pump_waits_for_it(request):
    from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend
    from tldw_chatbook.Widgets.Console.console_session_surface import (
        ConsoleSessionSurface,
    )

    app = _ready_app()
    host = ProductionConsoleHarness(app)
    gate_held = False
    async with host.run_test(size=_SIZE) as pilot:
        console = await _mounted_console(host, pilot, "#console-native-composer")
        controller = console._ensure_console_chat_controller()
        keeper = controller.store.active_session_id
        await _show_tabs(console, pilot, {keeper})
        projection = spend.ConsoleReadinessConfigProjection.for_screen(console)
        assert await _quiet(pilot, console, projection)
        surface = console.query_one("#console-session-surface", ConsoleSessionSurface)
        added = {
            controller.new_session(title=f"Held publication {number}").id
            for number in range(3)
        }
        controller.switch_session(keeper)
        # The holder has a real remount to do: three tabs to add.
        assert _open_tab_ids(console) == {keeper}
        # Only a gate: the holder parks on it, so the screen's pump can be
        # parked behind the holder before the holder does any DOM work.
        await surface._session_sync_lock.acquire()
        gate_held = True
        try:
            console._console_sync_maintenance_close_admission()
            assert await console._sync_native_console_chat_ui() is False
            assert console._console_sync_requested
            # The resume replays the coalesced request on a worker: the holder.
            console._console_sync_maintenance_resume()
            holder = next(
                worker
                for worker in console.app.workers
                if worker.node is console
                and worker.group == "console-sync"
                and not worker.is_finished
            )
            assert await _loop_wait(
                lambda: _awaiting(holder._task)[-3:]
                == ["ConsoleSessionSurface.sync_sessions", *_AWAITING_LOCK]
            ), f"holder never reached the surface lock: {_awaiting(holder._task)}"
            assert console._console_session_tabs_sync_lock.locked()
            # Queue a whole sync exactly as the view start does.
            assert console.call_after_refresh(console._sync_native_console_chat_ui)
            assert await _loop_wait(
                lambda: _awaiting(console._task)[-3:]
                == ["ChatScreen._sync_console_native_session_tabs", *_AWAITING_LOCK]
            ), f"screen pump never queued behind the holder: {_awaiting(console._task)}"
            # From here the test holds nothing; only the holder holds a lock.
            surface._session_sync_lock.release()
            gate_held = False
            released = time.monotonic()
            assert await _loop_wait(
                lambda: "ChatScreen._sync_console_native_session_tabs"
                not in _awaiting(console._task)
            ), (
                "the screen's pump stayed parked behind the holder for "
                f"{time.monotonic() - released:.1f}s; holder: {_awaiting(holder._task)}"
            )
            # Pilot's own wait needs the screen's pump: a stall raises here.
            await pilot.pause()
            assert _open_tab_ids(console) == {keeper, *added}
            assert await _quiet(pilot, console, projection)
        finally:
            if gate_held:
                surface._session_sync_lock.release()
            console._console_sync_maintenance_resume()
