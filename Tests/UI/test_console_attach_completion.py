"""A deferred full refresh cannot acknowledge Console attachment."""

import asyncio
from contextlib import asynccontextmanager

import pytest

from Tests.UI.test_console_runtime_ownership import _attach_reconciliation_screen
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

pytestmark = pytest.mark.bootstrap_profile


@asynccontextmanager
async def _mounted_console(host):
    """Own the runtime and finite callbacks absent from this small test App."""
    from tldw_chatbook.UI.Console_Modules.view_workers import (
        capture_console_view_workers,
        drain_console_view_workers,
    )

    async with host.run_test(size=(160, 48)) as pilot:
        try:
            yield pilot
        finally:
            runtime = getattr(host.app_instance, "console_runtime", None)

            async def retire():
                await drain_console_view_workers(capture_console_view_workers(host))
                if runtime is not None:
                    await runtime.dispose()

            task = asyncio.create_task(retire())
            host.app_instance._console_runtime_shutdown_task = task
            await asyncio.shield(task)


@pytest.mark.asyncio
@pytest.mark.parametrize("deferred", ["maintenance", "replay", "coalesced"])
async def test_original_deferred_refresh_does_not_finish_attachment(deferred):
    async def no_op():
        return None

    effects = []
    screen, runtime, _scheduled = _attach_reconciliation_screen(
        no_op, start=lambda: effects.append("started")
    )
    runtime.finish_view_reconciliation = (
        lambda *_args: effects.append("finished") or True
    )
    screen._sync_native_console_chat_ui = (
        ChatScreen._sync_native_console_chat_ui.__get__(screen)
    )
    screen._sync_console_native_session_tabs = no_op
    screen._console_sync_maintenance_paused = deferred == "maintenance"
    screen._console_control_bar_replay_whole_sync = deferred == "replay"
    screen._console_sync_in_progress = deferred == "coalesced"

    await screen._reconcile_console_after_attach()

    assert not screen._console_attach_sync_complete
    assert not screen._console_attach_reconciled
    assert effects == []


@pytest.mark.asyncio
async def test_coalesced_caller_accepts_full_refresh_completion_from_same_visit():
    entered, release = asyncio.Event(), asyncio.Event()

    async def coalesced():
        entered.set()
        await release.wait()
        return False

    effects = []
    screen, _runtime, scheduled = _attach_reconciliation_screen(
        coalesced, start=lambda: effects.append("started")
    )
    pending = asyncio.create_task(screen._reconcile_console_after_attach())
    await entered.wait()
    # Publication by the other full refresh while this tabs-only call waits.
    screen._console_attach_sync_complete = True
    release.set()
    await pending
    assert screen._console_attach_reconciled
    assert effects == ["started"]
    assert scheduled == []


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["runtime", "generation", "hidden", "closed"])
async def test_refresh_cannot_finish_a_replaced_or_hidden_attachment(change):
    entered, release = asyncio.Event(), asyncio.Event()

    async def held_sync():
        entered.set()
        await release.wait()
        return True

    effects = []
    screen, runtime, scheduled = _attach_reconciliation_screen(
        held_sync, start=lambda: effects.append("started")
    )
    runtime.finish_view_reconciliation = lambda *_: effects.append("finished") or True
    screen.call_after_refresh = lambda callback: scheduled.append((0, callback))
    pending = asyncio.create_task(screen._reconcile_console_after_attach())
    await entered.wait()
    if change == "runtime":
        runtime.view = object()
    elif change == "generation":
        runtime._attached_generation += 1
    elif change == "hidden":
        screen._is_active_console_screen = lambda: False
    else:
        screen._closed = True
    release.set()
    await pending
    assert not screen._console_attach_reconciled
    assert not screen._console_attach_sync_complete
    assert effects == []
    assert scheduled == []


@pytest.mark.asyncio
async def test_ordered_resume_keeps_its_original_presentation_priority():
    async def unexpected_sync():
        pytest.fail("ordinary startup refresh ran before ordered Resume")

    screen, _runtime, scheduled = _attach_reconciliation_screen(unexpected_sync)
    screen._resume_navigation_startup_in_progress = True
    await screen._reconcile_console_after_attach()
    assert screen._console_attach_reconciled
    assert scheduled == []


@pytest.mark.asyncio
async def test_mounted_original_refresh_replays_before_runtime_attachment(monkeypatch):
    from Tests.UI.test_console_initial_draft_ownership import DeferredInitialConsole
    from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector

    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    host = DeferredInitialConsole(app, "")
    async with _mounted_console(host) as pilot:
        screen = host.screen
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        await screen._reconcile_console_after_attach()
        assert not screen._console_attach_reconciled
        assert not screen._console_attach_sync_complete
        runtime = screen._console_runtime()
        assert runtime._reconciled_view is not screen
        original_finish = runtime.finish_view_reconciliation
        completions = []

        def finish(view, generation):
            completions.append((view, screen._console_attach_sync_complete))
            return original_finish(view, generation)

        monkeypatch.setattr(runtime, "finish_view_reconciliation", finish)
        screen._console_sync_maintenance_resume()
        for _ in range(100):
            if screen._console_attach_reconciled:
                break
            await pilot.pause(0.05)
        assert screen._console_attach_reconciled, {
            "full_pending": screen._console_sync_requested,
            "replay": getattr(screen, "_console_control_bar_replay_whole_sync", False),
            "scheduled": getattr(screen, "_console_control_bar_sync_scheduled", False),
            "running": screen._console_sync_in_progress,
            "projection_pending": getattr(getattr(screen, "_console_readiness_config_projection", None), "pending", None),
        }
        assert runtime._reconciled_view is screen
        assert completions == [(screen, True)]
        assert host.changed_owners == [screen._console_chat_store.active_session_id]

        # A later passive refresh must not bypass exhausted attach retries.
        # The first successful sync already published completion for this visit.
        screen._console_attach_reconciled = False
        screen._console_attach_reconcile_retry_exhausted = True
        queued = []
        original_after_refresh = screen.call_after_refresh

        def observe_reconcile(callback, *args, **kwargs):
            if callback == screen._start_console_attach_reconciliation:
                queued.append(callback)
                return True
            return original_after_refresh(callback, *args, **kwargs)

        monkeypatch.setattr(screen, "call_after_refresh", observe_reconcile)
        for _ in range(100):
            completed = await screen._sync_native_console_chat_ui()
            if completed:
                break
            await pilot.pause(0.05)
        assert completed is True
        assert queued == []


@pytest.mark.asyncio
async def test_mounted_suspend_resume_rearms_after_old_reconcile_retires(monkeypatch):
    from textual.screen import Screen

    from Tests.UI.test_console_initial_draft_ownership import DeferredInitialConsole
    from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector

    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    host = DeferredInitialConsole(app, "")
    async with _mounted_console(host) as pilot:
        screen = host.screen
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        await screen._reconcile_console_after_attach()
        assert not screen._console_attach_reconciled
        entered, release = asyncio.Event(), asyncio.Event()
        calls = 0

        async def held_sync():
            nonlocal calls
            calls += 1
            if calls == 1:
                entered.set()
                await release.wait()
            return True

        queued = []
        monkeypatch.setattr(screen, "_sync_native_console_chat_ui", held_sync)
        monkeypatch.setattr(
            screen, "call_after_refresh", lambda callback: queued.append(callback)
        )
        pending = asyncio.create_task(screen._reconcile_console_after_attach())
        try:
            await entered.wait()
            old_visit = screen._console_attach_visit_generation
            await host.push_screen(Screen())
            await pilot.pause()
            assert screen._console_attach_visit_generation > old_visit
            await host.pop_screen()
            await pilot.pause()
            # Resume really queued an attempt, but the old await still owns
            # the running flag. This is the lost-wake window.
            retries = [
                cb for cb in queued if cb == screen._start_console_attach_reconciliation
            ]
            assert retries
            queued.clear()
            retries[-1]()
            assert not screen._console_attach_reconciled
            release.set()
            await pending
            assert not screen._console_attach_reconciled
            retries = [
                cb for cb in queued if cb == screen._start_console_attach_reconciliation
            ]
            assert len(retries) == 1
            # A queued retry must also recheck visibility when it actually runs.
            await host.push_screen(Screen())
            await pilot.pause()
            retries[0]()
            await pilot.pause()
            assert not screen._console_attach_reconciled
            assert calls == 1
            await host.pop_screen()
            await pilot.pause()
            await screen._reconcile_console_after_attach()
            assert screen._console_attach_reconciled
            assert calls == 2
        finally:
            release.set()
            await pending


@pytest.mark.asyncio
async def test_suspended_original_refresh_keeps_dirty_work_for_resume(monkeypatch):
    from textual.screen import Screen

    from Tests.UI.test_console_initial_draft_ownership import DeferredInitialConsole
    from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
    from tldw_chatbook.UI.Console_Modules import config_sync
    from tldw_chatbook.Widgets.Console import ConsoleComposerBar

    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    host = DeferredInitialConsole(app, "")
    async with _mounted_console(host) as pilot:
        screen = host.screen
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        screen._console_sync_maintenance_resume()
        for _ in range(100):
            if (
                screen._console_attach_reconciled
                and not screen._console_sync_in_progress
            ):
                break
            await pilot.pause(0.05)
        assert screen._console_attach_reconciled
        assert not screen._console_sync_in_progress
        runtime = screen._console_runtime()
        store = screen._console_chat_store
        session_id = store.active_session_id
        generation = runtime._attached_generation
        hidden_passes, completions = [], []
        entered, release = asyncio.Event(), asyncio.Event()
        pending = None
        original_prepare = config_sync.prepare_console_presentation_sync
        original_started = screen._record_ui_worker_started
        original_finished = runtime.finish_view_reconciliation

        async def hold_original_preparation(current):
            result = await original_prepare(current)
            if current is screen and asyncio.current_task() is pending:
                # Only this FULL waits here, after its original preparation retires.
                entered.set()
                await release.wait()
            return result

        def record_started(name):
            if name == "console-sync" and not screen._is_active_console_screen():
                hidden_passes.append(name)
            return original_started(name)

        def finish(view, attachment):
            completions.append((view, attachment))
            return original_finished(view, attachment)

        monkeypatch.setattr(
            config_sync, "prepare_console_presentation_sync", hold_original_preparation
        )
        monkeypatch.setattr(screen, "_record_ui_worker_started", record_started)
        monkeypatch.setattr(runtime, "finish_view_reconciliation", finish)
        async def refresh_until_admitted():
            while True:
                completed = await screen._sync_native_console_chat_ui()
                if entered.is_set():
                    return completed
                assert completed is False
                await asyncio.sleep(0.05)

        pending = asyncio.create_task(refresh_until_admitted())
        try:
            await asyncio.wait_for(entered.wait(), 5)
            assert not pending.done()
            await host.push_screen(Screen())
            assert not screen._is_active_console_screen()
            assert runtime.view is screen
            assert runtime._attached_generation == generation
            assert screen._console_runtime() is runtime
            assert runtime.chat_store is store

            store.set_session_draft(session_id, "older hidden revision")
            assert await screen._sync_native_console_chat_ui() is False
            store.set_session_draft(session_id, "latest hidden revision")
            release.set()
            assert await pending is False
            # Let the existing 0.2s coalesced replay actually reach its timer.
            await asyncio.sleep(0.25)
            assert hidden_passes == [], (
                "a suspended Console admitted another FULL refresh"
            )
            assert screen._console_sync_requested
            assert completions == []

            await host.pop_screen()
            composer = screen.query_one(
                "#console-native-composer", ConsoleComposerBar
            )
            for _ in range(100):
                if (
                    screen._console_attach_reconciled
                    and composer.draft_text() == "latest hidden revision"
                ):
                    break
                await pilot.pause(0.05)
            assert screen._console_attach_reconciled
            assert composer.draft_text() == "latest hidden revision"
            assert completions == [(screen, generation)]
            assert runtime.chat_store is store
            assert store.active_session_id == session_id
        finally:
            release.set()
            await pending
