"""task-15470: Console sidebar-state persistence must not do synchronous file
I/O on the event loop, and must never lose a toggle to a quit.

Before this task, `ChatScreen.watch_sidebar_state` called `_save_sidebar_state()`
directly -- an open+parse+rewrite of `ui_state.toml` on the event loop, once
per `Collapsible.Toggled` (every sidebar toggle, plus expand-all/collapse-all/
reset). These tests pin the replacement: the reactive assignment only marks
the state dirty and arms a debounce timer (no write); the write itself runs
off the loop via `asyncio.to_thread`; and `on_unmount` (the screen's quit
path -- see `ChatScreen.on_unmount`, which Textual's `App._shutdown` ->
`_close_all` -> `_prune` drives for the active screen) force-flushes any
pending write so a toggle immediately followed by quit is not lost.
"""

from __future__ import annotations

import asyncio
import threading

import toml

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.config import _get_effective_config_path

APP_SIZE = (160, 48)


def _ui_state_path():
    return _get_effective_config_path().parent / "ui_state.toml"


async def _mounted_console_screen(pilot):
    screen = pilot.app.screen_stack[-1]
    await _wait_for_selector(screen, pilot, "#console-native-composer")
    return screen


async def test_toggle_does_not_write_synchronously_on_the_event_loop():
    """AC #2: a sidebar toggle performs no synchronous file I/O.

    Reassigning `sidebar_state` (what every `Collapsible.Toggled` handler
    does) must only arm the debounce timer -- the file must not exist (or
    must not yet carry the new id) immediately afterwards, before the
    debounce timer has had any chance to fire.
    """
    app = _build_test_app()
    async with ConsoleHarness(app).run_test(size=APP_SIZE) as pilot:
        screen = await _mounted_console_screen(pilot)

        screen.ui_state.set_collapsible_state("task-15470-probe", True)
        screen.sidebar_state = dict(screen.ui_state.collapsible_states)

        # No `await pilot.pause(...)` long enough for the 0.5s debounce to
        # fire -- the assertions below run before any timer could have.
        ui_state_path = _ui_state_path()
        if ui_state_path.exists():
            on_disk = toml.load(ui_state_path)
            collapsible_states = on_disk.get("sidebar", {}).get(
                "collapsible_states", {}
            )
            assert "task-15470-probe" not in collapsible_states, (
                "the toggle wrote to disk synchronously instead of "
                "debouncing"
            )
        assert screen._sidebar_state_dirty is True
        assert screen._sidebar_state_save_timer is not None


async def test_debounced_write_lands_after_the_timer_fires():
    """The debounce mechanism itself must actually persist -- not just skip.

    A mutant that deleted the timer callback's write (or never armed a
    timer at all) would pass the first test above; this one requires the
    write to actually happen once the debounce interval elapses.
    """
    app = _build_test_app()
    async with ConsoleHarness(app).run_test(size=APP_SIZE) as pilot:
        screen = await _mounted_console_screen(pilot)

        screen.ui_state.set_collapsible_state("task-15470-lands", True)
        screen.sidebar_state = dict(screen.ui_state.collapsible_states)

        # Debounce is 0.5s (SIDEBAR_STATE_SAVE_DEBOUNCE_SECONDS); wait past it
        # and let the dispatched worker's `to_thread` write complete.
        await pilot.pause(0.7)
        for _ in range(20):
            if not screen._sidebar_state_dirty:
                break
            await pilot.pause(0.05)

        on_disk = toml.load(_ui_state_path())
        assert (
            on_disk["sidebar"]["collapsible_states"]["task-15470-lands"] is True
        )


async def test_quit_immediately_after_toggle_flushes_the_pending_write():
    """AC #2 flush test: toggle, then quit before the debounce fires.

    Exiting the harness's `run_test()` context tears the app down through
    `App._shutdown` -> `_close_all` -> `_prune`, which unmounts the active
    screen and (per `ChatScreen.on_unmount`'s own docstring) runs
    `_flush_sidebar_state_now` before any other teardown step. No
    `pilot.pause()` long enough for the natural debounce timer is used here
    -- if quit relied on that timer, this test would still be waiting on a
    write that never happened.
    """
    app = _build_test_app()
    harness = ConsoleHarness(app)
    async with harness.run_test(size=APP_SIZE) as pilot:
        screen = await _mounted_console_screen(pilot)

        screen.ui_state.set_collapsible_state("task-15470-quit-flush", True)
        screen.sidebar_state = dict(screen.ui_state.collapsible_states)

        assert screen._sidebar_state_dirty is True
        # Exiting the `async with` block below tears the app down (quit),
        # deliberately with no further pause -- the pending write must
        # survive on the unmount path alone.

    on_disk = toml.load(_ui_state_path())
    assert (
        on_disk["sidebar"]["collapsible_states"]["task-15470-quit-flush"] is True
    )


async def test_reset_settings_schedules_a_write_even_from_an_empty_state():
    """`handle_reset_settings` must not silently drop its persistence.

    Resetting from an already-empty `sidebar_state` reassigns `{}` to `{}`,
    which Textual's reactive treats as a no-op and never calls
    `watch_sidebar_state` for -- so the reset handler must explicitly
    schedule a save rather than relying on the reactive alone. This flushes
    on unmount the same as a real toggle would.
    """
    app = _build_test_app()
    harness = ConsoleHarness(app)
    async with harness.run_test(size=APP_SIZE) as pilot:
        screen = await _mounted_console_screen(pilot)
        # Confirm the premise: sidebar_state is already the reactive default.
        assert screen.sidebar_state == {}

        # Call the handler directly (matches the `@on(Button.Pressed, ...)`
        # decorator's dispatch signature but does not depend on the reset
        # button being reachable/visible in this harness's default view --
        # the method body never reads `event`).
        screen.handle_reset_settings(None)
        await pilot.pause()

        assert screen._sidebar_state_dirty is True


async def test_toggle_during_in_flight_write_survives_a_quit():
    """Review round (task-15470): a toggle landing WHILE a debounced write
    is still in flight must survive a quit -- not just a toggle landing
    before the debounce timer ever fires (the case
    `test_quit_immediately_after_toggle_flushes_the_pending_write` covers).

    Reproduced without the fix: `_persist_sidebar_state_off_loop` cleared
    `_sidebar_state_dirty` only AFTER its write completed, so toggle 2's
    `dirty=True` (set while toggle 1's write was still inside `to_thread`)
    was silently clobbered back to False the instant toggle 1's write
    finished. Separately, `_flush_sidebar_state_now` `return`ed immediately
    after awaiting an in-flight worker, never re-checking
    `_sidebar_state_dirty` at all -- so even a dirty flag that legitimately
    survived to that point would have been dropped. Both fixes are
    required; this test forces the exact race by blocking toggle 1's write
    on a `threading.Event` and calling `_flush_sidebar_state_now` directly
    while it is still blocked.
    """
    app = _build_test_app()
    harness = ConsoleHarness(app)
    async with harness.run_test(size=APP_SIZE) as pilot:
        screen = await _mounted_console_screen(pilot)

        write_started = threading.Event()
        proceed = threading.Event()
        real_write = screen._write_sidebar_state_snapshot

        def slow_write(snapshot):
            write_started.set()
            assert proceed.wait(timeout=5), "test stalled waiting to proceed"
            real_write(snapshot)

        screen._write_sidebar_state_snapshot = slow_write

        # Toggle 1: goes through the real debounce + worker dispatch.
        screen.ui_state.set_collapsible_state("task-15470-inflight-1", True)
        screen.sidebar_state = dict(screen.ui_state.collapsible_states)
        await pilot.pause(0.7)  # past SIDEBAR_STATE_SAVE_DEBOUNCE_SECONDS

        assert write_started.wait(timeout=2), "worker never started its write"
        worker = screen._sidebar_state_persist_worker
        assert worker is not None and not worker.is_finished, (
            "toggle 1's worker must still be in flight for this test to "
            "mean anything"
        )

        # Toggle 2 lands while toggle 1's write is still blocked in flight.
        screen.ui_state.set_collapsible_state("task-15470-inflight-2", True)
        screen.sidebar_state = dict(screen.ui_state.collapsible_states)
        assert screen._sidebar_state_dirty is True

        # Call the real flush path directly, while toggle 1's worker is
        # still blocked -- it must wait for that worker, THEN notice
        # toggle 2's dirty flag and flush it too.
        flush_task = asyncio.create_task(screen._flush_sidebar_state_now())
        await pilot.pause()
        await pilot.pause()
        assert not flush_task.done(), (
            "_flush_sidebar_state_now returned without waiting for the "
            "in-flight worker -- this test is not exercising the race it "
            "claims to"
        )

        proceed.set()
        await flush_task

    on_disk = toml.load(_ui_state_path())
    collapsible_states = on_disk["sidebar"]["collapsible_states"]
    assert collapsible_states.get("task-15470-inflight-1") is True, (
        "toggle 1 should have landed"
    )
    assert collapsible_states.get("task-15470-inflight-2") is True, (
        "toggle 2 LOST"
    )
