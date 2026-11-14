"""Native refresh phases must leave room for input, teardown and cancellation."""

from __future__ import annotations

import asyncio

import pytest

from Tests.UI.test_console_mcp_approval import (
    _build_ready_production_console_app,
    _wait_for_production_console_ready,
)
from tldw_chatbook import config
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@pytest.mark.parametrize("phase", ["scope", "controls", "tabs"])
@pytest.mark.parametrize("transition", ["continue", "teardown", "cancel"])
@pytest.mark.parametrize("dispatch_mode", ["ready", "timer"])
async def test_native_refresh_honors_input_between_phases(
    monkeypatch, phase, transition, dispatch_mode
):
    """Removing a yield or its teardown fence must fail on the real pipeline."""
    app = _build_ready_production_console_app()
    async with app.run_test(size=(120, 50)) as pilot:
        screen = await _wait_for_production_console_ready(app, pilot)
        store = screen._console_chat_store
        session_id = store.active_session_id
        task = None
        observed = []
        entered = []
        loop_lock_owners = []
        message = None

        def interrupt():
            nonlocal message
            observed.append(True)
            if transition == "teardown":
                screen._closing = True
            elif transition == "cancel":
                task.cancel()
            else:
                loop_lock_owners.append(
                    (
                        config._settings_rebuild_lock()._is_owned(),
                        config._config_file_lock()._is_owned(),
                    )
                )
                message = store.append_message(
                    session_id,
                    role=ConsoleMessageRole.USER,
                    content="Current phase input",
                )

        owner, producer_name, consumer_name = {
            "scope": (
                screen._character,
                "_refresh_active_character_avatar_if_scope_changed",
                "_sync_console_rail_and_controls",
            ),
            "controls": (
                screen,
                "_sync_console_rail_and_controls",
                "_sync_console_settings_recovery_surfaces",
            ),
            "tabs": (
                screen,
                "_sync_console_native_session_tabs",
                "_dispatch_active_console_roleplay_refresh",
            ),
        }[phase]
        producer = getattr(owner, producer_name)
        consumer = getattr(screen, consumer_name)

        def schedule_input():
            if asyncio.current_task() is task:
                loop = asyncio.get_running_loop()
                if dispatch_mode == "timer":
                    loop.call_later(0, interrupt)
                else:
                    loop.call_soon(interrupt)

        if phase == "controls":

            def producing(*args, **kwargs):
                result = producer(*args, **kwargs)
                if asyncio.current_task() is task and result is True:
                    schedule_input()
                return result

        else:

            async def producing(*args, **kwargs):
                result = await producer(*args, **kwargs)
                schedule_input()
                return result

        def consuming(*args, **kwargs):
            if asyncio.current_task() is task:
                entered.append(bool(observed))
            return consumer(*args, **kwargs)

        monkeypatch.setattr(owner, producer_name, producing)
        monkeypatch.setattr(screen, consumer_name, consuming)
        try:
            # A supported busy-config deferral has not reached the phase yet.
            async with asyncio.timeout(10):
                while not observed:
                    task = asyncio.create_task(screen._sync_native_console_chat_ui())
                    try:
                        await task
                    except asyncio.CancelledError:
                        if transition != "cancel" or not task.cancelled():
                            raise
                    else:
                        if observed and transition == "cancel":
                            pytest.fail("Queued cancellation missed the refresh phase")
                    if not observed:
                        await _wait_for_production_console_ready(app, pilot)
            assert observed == [True]
            assert entered == ([True] if transition == "continue" else [])
            assert store.active_session_id == session_id
            if transition == "continue":
                # A competing writer may hold a lock; this loop must not.
                assert loop_lock_owners == [(False, False)]
        finally:
            screen._closing = False
            if task is not None and not task.done():
                task.cancel()
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
        # The target's finally may legitimately dispatch a coalesced successor.
        await _wait_for_production_console_ready(app, pilot)
        assert not screen._console_sync_in_progress
        assert screen._workspace._console_tick_builds is None
        if transition == "continue":
            transcript = screen.query_one("#console-native-transcript")
            assert [
                item.content for item in transcript._messages if item.id == message.id
            ] == ["Current phase input"]
