"""A Console sync tick must not outlive -- or kill -- its own screen.

task-15860, cross-suite leak. The wake-fires-headless landing turned
`Tests/UI/test_console_headless_wake_fires.py` +
`Tests/UI/test_console_store_continuity.py`, run together, from 5 passed
into `1 failed, 4 passed`: the four-way agreement test could no longer
navigate back to Console ("navigating to 'chat' never reached ChatScreen;
stuck on LibraryScreen"), while every file passed alone.

What the instrumentation found is NOT test pollution. Traced live
(`ConsoleSessionSurface.sync_sessions` wrapped, `App._handle_exception`
wrapped, `run_worker` origins captured):

    tick ENTER  screen_running=True   <- a console-sync tick starts
    ...                                  the user navigates away
    run_worker(console-sync) from chat_screen.py (the tick's own `finally`)
    tick EXIT-OK screen_running=False <- the screen is already closed
    tick ENTER  screen_running=False  <- the re-armed worker runs anyway
    sync_sessions RAISED NoMatches("#console-native-tab-strip")
    App._handle_exception(WorkerFailed(...))   <- the app EXITS

Three production facts compose into an app-killing crash:

1. `_sync_native_console_chat_ui` re-arms itself from its own `finally`
   whenever a coalesced request came in mid-tick (a wake turn landing
   rows while the user navigates is exactly that). That `run_worker`
   call runs AFTER Textual's unmount sweep (`Widget._on_unmount` ->
   `workers.cancel_node(self)`), so the worker it creates is never in the
   cancelled set.
2. The tick queries the DOM (`_sync_console_native_session_tabs` ->
   `ConsoleSessionSurface.sync_sessions` ->
   `query_one("#console-native-tab-strip")`), which navigating away has
   removed.
3. Textual workers default to `exit_on_error=True`, so the resulting
   `NoMatches` reaches `App._handle_exception` and takes the whole TUI
   down. Post-mortem the app reported `running=False closing=True
   closed=True`, every `post_message` was silently dropped, and the
   pending navigation never happened -- which is precisely the shape the
   contaminated suite saw.

Nothing above is harness-only: real `NavigateToScreen` routing, the real
screen teardown, the real 0.2s-plus-on-append sync tick, Textual's own
worker defaults. The test suite supplied timing pressure, not the bug.
`is_mounted` is no defence -- the removed surface still reported
`is_mounted=True` while its pump reported `is_running=False`.
"""

from __future__ import annotations

import pytest
from textual.css.query import NoMatches

from Tests.Chat.test_console_fleet_wake import _settle
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_fleet_wake_wiring import _attach_real_dbs
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_console_store_continuity import (
    _StallingWakeGateway,
    _navigate,
    _seed_console,
)
from tldw_chatbook.UI.Screens.chat_screen import (
    ChatScreen,
    _console_screen_is_torn_down,
)


def _console_sync_workers(app):
    """Live workers in the screen-owned `console-sync` group."""
    return [worker for worker in app.workers if worker.group == "console-sync"]


async def _console_app(tmp_path):
    """Return `(app, gateway)` -- a real app wired for a real Console.

    Not a fixture: the pilot has to stay open for the assertions, so each
    test owns its own `async with app.run_test(...)`.
    """
    app = _build_test_app()
    _attach_real_dbs(app, tmp_path)
    _configure_native_ready_console(app)
    gateway = _StallingWakeGateway()
    app.console_provider_gateway_factory = lambda: gateway
    app.app_config.setdefault("console", {})["agent_runtime"] = False
    return app, gateway


@pytest.mark.asyncio
async def test_a_console_sync_tick_scheduled_after_teardown_cannot_kill_the_app(
    tmp_path,
):
    """The observed crash, driven deterministically.

    RED before the fix: the worker runs a full sync against the removed
    surface, raises `NoMatches`, and `exit_on_error=True` hands that to
    `App._handle_exception` -- after which the app is closed, the posted
    navigation is dropped, and the user is stuck on Library.
    """
    app, gateway = await _console_app(tmp_path)

    async with app.run_test(size=(160, 48)) as pilot:
        chat, _controller, _store, _session_id, _conversation_id = await _seed_console(
            app, pilot, gateway
        )
        await _navigate(app, pilot, "library", expect="LibraryScreen")
        assert chat not in app.screen_stack, "Console must actually unmount"
        assert _console_screen_is_torn_down(chat), (
            "harness precondition: the navigation must really have closed this "
            "screen's pump -- a test that ran against a live screen would prove "
            "nothing"
        )
        assert chat.is_mounted, (
            "harness precondition: `is_mounted` is still True for the closed "
            "screen -- this is why a mount check cannot be the guard"
        )

        # Exactly what the tick's own `finally` did: schedule one more
        # console-sync worker on the screen Textual already closed.
        chat.run_worker(
            chat._sync_native_console_chat_ui(),
            exclusive=True,
            group="console-sync",
        )
        assert await _settle(lambda: not _console_sync_workers(app), seconds=10.0), (
            "the post-teardown console-sync worker never finished"
        )

        assert app.is_running, (
            "a console-sync tick on a torn-down screen killed the app: "
            f"{app._exception!r}"
        )
        # The user-visible consequence of that death, asserted directly.
        chat2 = await _navigate(app, pilot, "chat", expect="ChatScreen")
        assert isinstance(chat2, ChatScreen), type(chat2).__name__


@pytest.mark.asyncio
async def test_a_torn_down_screen_runs_no_sync_work_and_never_re_arms(tmp_path):
    """The two guards, each with its own observable.

    * a tick entered on a closed screen does NO work -- it must not reach
      the sync steps at all (a dead screen repainting nothing is the
      point; `_console_screen_displayed()`-style view semantics rely on
      it);
    * the coalesced re-arm must not schedule a worker the unmount sweep
      can never cancel.
    """
    app, gateway = await _console_app(tmp_path)

    async with app.run_test(size=(160, 48)) as pilot:
        chat, _controller, _store, _session_id, _conversation_id = await _seed_console(
            app, pilot, gateway
        )
        await _navigate(app, pilot, "library", expect="LibraryScreen")
        assert _console_screen_is_torn_down(chat), "harness precondition"

        reached: list[object] = []
        chat._sync_console_control_bar = lambda *a, **k: reached.append(a)

        # A coalesced request is what a wake turn landing rows during the
        # navigation leaves behind.
        chat._console_sync_requested = True
        await chat._sync_native_console_chat_ui()

        assert reached == [], (
            "a torn-down screen still ran its Console UI sync"
        )
        assert not _console_sync_workers(app), (
            "a torn-down screen re-armed its own sync worker; Textual's unmount "
            "sweep has already run, so nothing will ever cancel it"
        )
        assert app.is_running, f"the app died: {app._exception!r}"


async def _arm_midtick_injection(app, pilot, gateway, *, navigate_away: bool):
    """Seed Console and schedule a DOM failure part-way through the next tick.

    The entry guard cannot catch a tick that was already past it when the
    screen closed -- the traced crash showed exactly that transition
    (`tick EXIT-OK screen_running=False`). Reproducing it by real timing
    would be a race, so the failure is injected at a fixed point instead:
    `_refresh_active_character_avatar_if_scope_changed` is the last awaited
    step before the DOM-touching `_sync_console_native_session_tabs`.
    Deliberately private-method-coupled: if that step is renamed or moved
    after the tab sync, this stops proving anything, and it should fail
    loudly rather than pass vacuously.

    `navigate_away=True` performs the REAL navigation (and therefore the
    real screen teardown) inside that window -- no flag is poked. Where a
    test does have to set `_closing` by hand (the partial-teardown case
    below, which a real navigation cannot stage), it must restore it:
    `MessagePump._close_messages` early-returns when the flag is already
    set, so a screen left flagged never actually closes and the app hangs
    at shutdown -- measured, it wedged this file's first draft past a
    120s timeout.
    """
    chat, _controller, _store, _session_id, _conversation_id = await _seed_console(
        app, pilot, gateway
    )
    original = chat._refresh_active_character_avatar_if_scope_changed
    injected: list[bool] = []

    async def _break_the_tab_strip():
        await original()
        injected.append(navigate_away)
        if navigate_away:
            await _navigate(app, pilot, "library", expect="LibraryScreen")
            # A coalesced follow-up request, exactly as a wake turn landing
            # transcript rows during the navigation leaves behind. This is
            # what the tick's `finally` would re-arm on.
            chat._console_sync_requested = True
        else:
            for strip in chat.query("#console-native-tab-strip"):
                await strip.remove()

    # A direct call would coalesce into a tick already in flight.
    assert await _settle(
        lambda: not chat._console_sync_in_progress, seconds=10.0
    ), "a console-sync tick never finished; the direct call would coalesce"
    chat._console_sync_requested = False
    chat._refresh_active_character_avatar_if_scope_changed = _break_the_tab_strip
    return chat, injected


@pytest.mark.asyncio
async def test_a_live_screens_console_sync_failure_still_propagates(tmp_path):
    """The CONTROL for the teardown-scoped `except`.

    Without this, the fix would be indistinguishable from a blanket
    `except Exception: pass` that silently eats every real Console sync
    bug.
    """
    app, gateway = await _console_app(tmp_path)

    async with app.run_test(size=(160, 48)) as pilot:
        chat, injected = await _arm_midtick_injection(
            app, pilot, gateway, navigate_away=False
        )
        with pytest.raises(NoMatches):
            await chat._sync_native_console_chat_ui()
        assert injected == [False], (
            f"the control never reached the injected failure: {injected}"
        )
        assert not _console_screen_is_torn_down(chat), (
            "control precondition: that screen was alive throughout"
        )


@pytest.mark.asyncio
async def test_a_real_navigation_arriving_mid_tick_is_absorbed(tmp_path):
    """The user navigates away WHILE a sync tick is running."""
    app, gateway = await _console_app(tmp_path)

    async with app.run_test(size=(160, 48)) as pilot:
        chat, injected = await _arm_midtick_injection(
            app, pilot, gateway, navigate_away=True
        )
        await chat._sync_native_console_chat_ui()
        # Checked BEFORE any pump: the `finally` re-arm creates its worker
        # synchronously, so a leaked one is visible the moment the tick
        # returns.
        assert not _console_sync_workers(app), (
            "the tick re-armed a console-sync worker on the screen the "
            "navigation had just closed; Textual's unmount sweep has already "
            "run, so nothing will ever cancel it"
        )
        assert injected == [True], (
            f"the teardown case never reached the injected navigation: {injected}"
        )
        assert _console_screen_is_torn_down(chat), (
            "the injected navigation must have really closed this screen"
        )
        assert app.is_running, f"the app died: {app._exception!r}"
        chat2 = await _navigate(app, pilot, "chat", expect="ChatScreen")
        assert isinstance(chat2, ChatScreen), type(chat2).__name__


@pytest.mark.asyncio
async def test_a_partly_dismantled_screen_mid_tick_is_absorbed(tmp_path):
    """The window the entry guard structurally cannot cover.

    `MessagePump._close_messages` sets `_closing` as its FIRST statement
    and children come down after it, so a tick already past the entry
    guard can find a surface whose own children are gone -- which is
    exactly the state the traced crash raised from
    (`sync_sessions` -> `query_one("#console-native-tab-strip")`, on a
    surface that still answered `is_mounted=True`). A real navigation
    cannot stage this from a test, because by the time the awaited
    navigation returns the whole surface has gone and
    `_sync_console_native_session_tabs` short-circuits on its own
    `QueryError` guard before reaching the raising call. So the two halves
    are staged directly, in Textual's own order.

    `_closing` is restored before the test leaves: `_close_messages`
    early-returns when it is already set, so a screen left flagged never
    actually closes and the app hangs at shutdown (measured -- it wedged
    this file's first draft past a 120s timeout).
    """
    app, gateway = await _console_app(tmp_path)

    async with app.run_test(size=(160, 48)) as pilot:
        chat, _controller, _store, _session_id, _conversation_id = await _seed_console(
            app, pilot, gateway
        )
        original = chat._refresh_active_character_avatar_if_scope_changed
        injected: list[bool] = []

        async def _dismantle_mid_tick():
            await original()
            injected.append(True)
            for strip in chat.query("#console-native-tab-strip"):
                await strip.remove()
            chat._closing = True

        assert await _settle(
            lambda: not chat._console_sync_in_progress, seconds=10.0
        ), "a console-sync tick never finished; the direct call would coalesce"
        chat._console_sync_requested = False
        chat._refresh_active_character_avatar_if_scope_changed = _dismantle_mid_tick
        try:
            await chat._sync_native_console_chat_ui()
        finally:
            chat._closing = False
            chat._refresh_active_character_avatar_if_scope_changed = original

        assert injected == [True], (
            f"the tick never reached the injected dismantling: {injected}"
        )
        assert app.is_running, f"the app died: {app._exception!r}"
