"""Separate exact original detached-view native Character lifetime route."""

from __future__ import annotations
import asyncio
import inspect
import pytest
from types import CoroutineType
from Tests.private_profile import private_profile_test
from Tests.UI.test_original_character_teardown_lifetime import (
    _stock_private_app,
    _wait_for_exact_pair,
    _retire_exact_reader,
    _violations,
    _write,
    _accept,
)

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.timeout(300)]


@pytest.mark.asyncio
@private_profile_test
async def test_real_app_detached_character_worker_retires_before_owned_exit(
    request, tmp_path
):
    """The actual App retains its held callback after original view detachment."""
    from textual.screen import Screen
    from Tests.Performance._character_teardown_original_gate import (
        OriginalCharacterTeardownGate,
    )
    from Tests.UI.background_signals import await_background_task
    from Tests.UI import test_console_session_tab_close as original
    from Tests.UI.app_factory import drain_active_service_patches, drain_created_dirs
    from tldw_chatbook.Backup_Recovery.participants import _close_settled_core_cache

    app = _stock_private_app()
    gate = None
    violations = []
    try:
        async with app.run_test(size=(160, 44)) as pilot:
            startup = app._initial_screen_setup_task
            assert type(startup) is asyncio.Task
            assert startup.get_loop() is asyncio.get_running_loop()
            producer = inspect.getattr_static(
                type(app), "_run_no_splash_post_mount_setup"
            )
            coroutine = startup.get_coro()
            assert (
                type(coroutine) is CoroutineType
                and coroutine.cr_code is producer.__code__
            )
            if coroutine.cr_frame is not None:
                assert (
                    coroutine.cr_frame.f_globals is producer.__globals__
                    and coroutine.cr_frame.f_locals.get("self") is app
                )
            await await_background_task(startup, what="original initial Console setup")
            assert (
                startup.done()
                and not startup.cancelled()
                and startup.exception() is None
            )
            assert app._initial_screen_pushed is True
            console = await original._mounted_console(
                app, pilot, "#console-native-composer"
            )
            await console._sync_native_console_chat_ui()
            database = app.chachanotes_db
            creator_connection = getattr(database._local, "conn", None)
            assert creator_connection is not None
            # A real suspend/resume consumes the original mount one-shot token;
            # no cache/clock/attachment flag is overwritten by this test.
            await app.push_screen(Screen())
            await pilot.pause()
            gate = OriginalCharacterTeardownGate(
                console,
                app,
                inspect.getattr_static(type(app), "_shutdown_console_runtime"),
                "console-character-context-refresh",
            )
            runtime = app.console_runtime
            assert runtime.view is console and runtime.app is app
            attachment_generation = console._console_runtime_attachment_generation
            detach = inspect.getattr_static(type(runtime), "detach_view")
            gate._pin(detach)
            gate.slots.append((type(runtime), "detach_view", detach))
            gate.install()
            try:
                await app.pop_screen()
                gate.mark_stage("real_resume_issued")
                assert await _wait_for_exact_pair(gate)
                assert app.screen is console and console.app_instance is app
                # This is the exact original detach API/generation used by
                # stock on_unmount, while this known finite callback is live.
                assert detach(runtime, console, attachment_generation) is True
                assert runtime.view is None and runtime._attached_generation is None
                assert app.console_runtime is runtime and gate.worker.node is console
                assert not gate.worker._task.done() and not gate.release.is_set()
                held = gate.boundary_facts()
                assert (
                    held["native_live"]
                    and held["exact_operation_counted"]
                    and held["exact_lease_live"]
                    and not held["pause_active"]
                )
                # This is the exact shipping App-exit stage, not a substitute
                # runtime/close function or a narrowed storage authority guard.
                await app._shutdown_console_runtime()
                closed = _close_settled_core_cache(database)
                if not closed:
                    facts = gate.boundary_facts()
                    assert (
                        facts["native_live"]
                        and facts["exact_operation_counted"]
                        and facts["exact_lease_live"]
                        and not facts["pause_active"]
                    )
                violations = _violations(gate)
            finally:
                if gate.entered.is_set():
                    await _retire_exact_reader(gate)
                else:
                    gate.release.set()
            assert _close_settled_core_cache(
                database
            ), "actual declared creator owner did not retire after callback completion"
        # All ordinary original App lifecycle and fixture-owned cache cleanup
        # must succeed before the sole causal oracle below is evaluated.
    finally:
        if gate is not None:
            gate.release.set()
        try:
            drain_active_service_patches()
            drain_created_dirs()
        finally:
            if gate is not None:
                receipt = _write(gate, tmp_path, "real-app-detached-resume", violations)
    _accept(receipt, violations)
