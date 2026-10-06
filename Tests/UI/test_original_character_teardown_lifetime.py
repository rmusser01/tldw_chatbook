"""Two distinct original private-child lifetime controls, not a timing census."""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
from types import CoroutineType

import pytest

from Tests.private_profile import private_profile_test

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.timeout(300)]


def _stock_private_app():
    """Persist supported private inputs before the unpatched constructor."""
    from Tests.Performance.console_storage_unit_observer import (
        OriginalStorageUnitObserver,
    )
    from tldw_chatbook import config
    from tldw_chatbook.Constants import TAB_CHAT

    for section, key, value in (
        ("splash_screen", "enabled", False),
        ("first_run", "setup_completed", True),
        ("_first_run", "setup_completed", True),
        ("console", "onboarding", {"first_send_completed": True}),
        ("general", "default_tab", TAB_CHAT),
    ):
        assert config.save_setting_to_cli_config(section, key, value)
    loaded = config.load_cli_config_and_ensure_existence()
    assert loaded["splash_screen"]["enabled"] is False
    assert loaded["first_run"]["setup_completed"] is True
    assert loaded["_first_run"]["setup_completed"] is True
    assert loaded["console"]["onboarding"]["first_send_completed"] is True
    from tldw_chatbook import app as app_module
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    assert app_module.load_settings is config.load_settings
    proof = OriginalStorageUnitObserver({}, lambda: False, lambda _name: None)
    for name in (
        "__init__",
        "_init_notes_service",
        "_init_prompts_service",
        "_init_providers_models",
        "_init_media_db",
    ):
        function = inspect.getattr_static(app_module.TldwCli, name)
        proof._pin(function)
        proof.slots.append((app_module.TldwCli, name, function))
    app = app_module.TldwCli()
    assert app.notes_service is not None
    assert (
        type(app.chachanotes_db) is CharactersRAGDB
        and not app.chachanotes_db.is_memory_db
    )
    assert proof.close()["original_source_current"]
    return app


async def _wait_for_exact_pair(gate):
    """Poll only the known causal event, within the original ten-second bound."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 10.0
    while loop.time() < deadline:
        if gate.entered.is_set():
            task = gate.worker._task
            assert type(task) is asyncio.Task and task.get_loop() is loop
            return True
        await asyncio.sleep(0.01)
    return False


async def _retire_exact_reader(gate):
    gate.release.set()
    for _ in range(1000):
        if gate.invoke_returned.is_set() and gate.future.done():
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("original Character invoke did not physically retire")
    await asyncio.gather(gate.worker._task, return_exceptions=True)


def _write(gate, tmp_path, route, violations):
    receipt = gate.close()
    receipt["route"] = route
    receipt["violations"] = violations
    receipt["original_scope_body_and_guards_replaced"] = False
    (tmp_path / (route + "-character-teardown.json")).write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def _accept(receipt, violations):
    assert (
        receipt["complete"]
        and receipt["original_source_current"]
        and not receipt["invalid"]
    ), receipt
    assert receipt["global_events"] == 0 and receipt["hooks_retired_before_inactive"]
    assert (
        receipt["actual_original_pair_held"] and receipt["original_exit_rows"]
    ), receipt
    assert receipt["original_invoke_returned"] and receipt["actual_native_future_done"]
    assert (
        receipt["actual_native_connection_closed"]
        and receipt["actual_lease_retired"]
        and receipt["actual_operation_retired"]
    )
    assert receipt["original_worker_task_done"]
    assert not violations, violations


def _violations(gate):
    result = []
    for row in gate.shutdown_rows:
        if (
            row["native_live"]
            and row["exact_operation_counted"]
            and row["exact_lease_live"]
            and not row["release_set"]
        ):
            result.append(
                "original owner exit returned before its exact native Character callback retired"
            )
    for row in gate.closer_rows:
        if (
            not row["allowed"]
            and row["native_live"]
            and row["exact_operation_counted"]
            and row["exact_lease_live"]
            and not row["pause_active"]
        ):
            result.append(
                "original settled-close correctly refused the still-live exact Character operation"
            )
    return result


@pytest.mark.asyncio
@private_profile_test
async def test_real_app_resume_character_worker_retires_before_owned_exit(
    request, tmp_path
):
    """The actual App lifecycle must own its original resume Character callback."""
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
            gate.install()
            try:
                await app.pop_screen()
                gate.mark_stage("real_resume_issued")
                assert await _wait_for_exact_pair(gate)
                assert app.screen is console and console.app_instance is app
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
                receipt = _write(gate, tmp_path, "real-app-resume", violations)
    _accept(receipt, violations)


@pytest.mark.asyncio
@private_profile_test
async def test_prepared_host_console_sync_retires_before_creator_close(
    request, tmp_path
):
    """The stock harness must retain the screen callback through real close."""
    from textual.app import App
    from Tests.Performance._character_teardown_original_gate import (
        OriginalCharacterTeardownGate,
    )
    from Tests.UI import test_console_session_tab_close as original
    from Tests.UI.app_factory import drain_active_service_patches, drain_created_dirs

    gate = None
    violations = []
    context = original._pending_close_app(request, "chat_create")
    app = await context.__aenter__()
    owner = app._pending_close_owned_resources
    try:
        host = original.ProductionConsoleHarness(app)
        try:
            async with host.run_test(size=(160, 44)) as pilot:
                console = await original._mounted_console(
                    host, pilot, "#console-native-composer"
                )
                await console._sync_native_console_chat_ui()
                controller = console._console_chat_controller
                gate = OriginalCharacterTeardownGate(
                    console, host, App._shutdown, "console-sync"
                )
                gate.install()
                # New selected-session ownership genuinely changes the display
                # key. No forced expiry, private memo reset, or fake receiver.
                selected = controller.new_session(title="Held teardown scope")
                gate.mark_stage("new_session_returned")
                controller.switch_session(selected.id)
                gate.mark_stage("switch_session_returned")
                console.run_worker(
                    console._sync_native_console_chat_ui(),
                    exclusive=True,
                    group="console-sync",
                )
                gate.mark_stage("stock_sync_worker_issued")
                assert await _wait_for_exact_pair(gate)
                owner.adopt_runtime_runs()
            # The original host has exited, while the actual owner loop remains
            # alive exactly as in the prepared-Close fixture's original finally.
            await owner.dispose_runtime()
            try:
                owner.close_creators()
            except RuntimeError as error:
                assert error.args == ("prepared_close_database_not_retired",)
                facts = gate.boundary_facts()
                assert (
                    facts["native_live"]
                    and facts["exact_operation_counted"]
                    and facts["exact_lease_live"]
                    and not facts["pause_active"]
                )
            violations = _violations(gate)
        finally:
            if gate is not None:
                if gate.entered.is_set():
                    await _retire_exact_reader(gate)
                else:
                    gate.release.set()
        owner.close_creators()
    finally:
        # Original context performs its exact second settled check and removes
        # only its declared contained sandbox after actual physical retirement.
        try:
            await context.__aexit__(*sys.exc_info())
            drain_active_service_patches()
            drain_created_dirs()
        finally:
            if gate is not None:
                receipt = _write(gate, tmp_path, "prepared-host-sync", violations)
    _accept(receipt, violations)
