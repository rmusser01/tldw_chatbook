"""Issue #2708: a failed recovery must not strand the mounted controls."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from textual.widgets import Button, Static

from Tests.Chat.test_console_dispatch_recovery import (
    _acceptance,
    _database,
    _insert,
    _NoReplayGateway,
    _restored_store,
    _start,
)
from Tests.private_profile import private_profile_test
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from Tests.UI.test_console_dictation import _mounted_console, _ready_host
from Tests.UI.test_console_dispatch_recovery import _state
from Tests.UI.test_console_prompt_queue import _ui_controller
from Tests.UI.test_console_send_disabled_state import _wait_for_condition
from tldw_chatbook.Chat.console_chat_controller import (
    ConsoleChatController,
    ConsoleSubmitResult,
)
from tldw_chatbook.UI.Console_Modules.dispatch_recovery import (
    ConsoleDispatchRecoveryRegion,
)
from tldw_chatbook.Widgets.Console import ConsoleComposerBar


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["refused", "exception", "cancelled"])
async def test_recovery_rearms_after_unchanged_failed_action(outcome):
    """Exercise completion repaint through the production UI dispatcher."""
    recovery = _state(started=True)
    attempts = []
    pending = []

    async def recover(_session_id):
        attempts.append(True)
        if len(attempts) == 1:
            if outcome == "exception":
                raise RuntimeError("recovery unavailable")
            if outcome == "cancelled":
                raise asyncio.CancelledError
        return ConsoleSubmitResult(False, False, "Try again or discard.")

    ui = _ui_controller(
        SimpleNamespace(
            retry_dispatch_recovery=recover, discard_dispatch_recovery=recover
        ),
        {"notified": [], "sync": []},
    )

    def dispatch(session_id, _assistant_id, action):
        pending.append(
            asyncio.create_task(
                ui.handle_primary_intent(session_id, action=action, expected_revision=0)
            )
        )

    region = ConsoleDispatchRecoveryRegion(
        recovery, session_id="session-1", on_action=dispatch
    )

    class RecoveryApp(ConsolidatedCSSApp):
        def compose(self):
            yield region

    async def sync_ui():
        region.sync_recovery("session-1", recovery)

    ui._sync_ui = sync_ui
    async with RecoveryApp().run_test() as pilot:
        region.query_one("#console-dispatch-recovery-retry_anyway", Button).press()
        await pilot.pause()
        result = (await asyncio.gather(*pending, return_exceptions=True))[0]
        if outcome == "exception":
            assert isinstance(result, RuntimeError)
        elif outcome == "cancelled":
            assert isinstance(result, asyncio.CancelledError)
        region.query_one("#console-dispatch-recovery-discard", Button).press()
        await pilot.pause()
        await asyncio.gather(*pending, return_exceptions=True)
        assert len(attempts) == 2


@pytest.mark.asyncio
async def test_pending_recovery_still_ignores_duplicate_clicks():
    recovery = _state(started=True)
    attempts = []
    region = ConsoleDispatchRecoveryRegion(
        recovery, session_id="session-1", on_action=lambda *args: attempts.append(args)
    )

    class RecoveryApp(ConsolidatedCSSApp):
        def compose(self):
            yield region

    async with RecoveryApp().run_test() as pilot:
        retry = region.query_one("#console-dispatch-recovery-retry_anyway", Button)
        retry.press()
        retry.press()
        await pilot.pause()
        region.sync_recovery("session-1", recovery.with_in_flight(True))
        await pilot.pause()
        region.query_one("#console-dispatch-recovery-discard", Button).press()
        await pilot.pause()
        assert len(attempts) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("check_copy", [False, True])
@pytest.mark.parametrize("first_attempt", ["discard_failure", "retry_cancel"])
@private_profile_test
async def test_restored_recovery_failed_discard_can_be_discarded_again(
    request, tmp_path, check_copy, first_attempt, monkeypatch
):
    """Real SQLite rollback, store release, screen callback, and composer."""
    _app, host = _ready_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        db, conversation_id, repository = _database(tmp_path / "recovery.sqlite")
        request.addfinalizer(db.close)
        checkpoint = _insert(db, repository, _acceptance(conversation_id))
        _start(repository, checkpoint)
        store, session_id = _restored_store(db, conversation_id)
        gateway = _NoReplayGateway(db)
        controller = ConsoleChatController(
            store=store, provider_gateway=gateway, agent_runtime_enabled=False
        )
        runtime = console._console_runtime()
        runtime.set_chat_store(store)
        runtime.set_chat_controller(controller)
        runtime.attach_view(console)
        store.set_session_draft(session_id, "next prompt")
        await console._sync_native_console_chat_ui()
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        console._sync_console_composer_action_state(can_save_chatbook=False)
        await pilot.pause()
        syncs = []
        sync_ui = console._prompt_queue._sync_ui

        async def completed_sync():
            await sync_ui()
            syncs.append(True)

        monkeypatch.setattr(console._prompt_queue, "_sync_ui", completed_sync)
        region = console.query_one(ConsoleDispatchRecoveryRegion)
        if check_copy:
            reason = composer.query_one("#console-send-disabled-reason", Static)
            assert reason.renderable.plain == (
                "Send blocked — resolve response recovery first"
            )
            assert (
                "recovery"
                in str(
                    composer.query_one("#console-send-message", Button).tooltip
                ).lower()
            )
            assert not composer.has_class("console-composer-setup-blocked")
            composer.insert_text("!")
            assert reason.renderable.plain == (
                "Send blocked — resolve response recovery first"
            )
            await pilot.pause()
        connection = db.get_connection()
        if first_attempt == "discard_failure":
            connection.execute(
                "CREATE TRIGGER fail_discard_delete BEFORE DELETE ON "
                "console_dispatch_checkpoints BEGIN SELECT RAISE(ABORT, 'fail'); END"
            )
            connection.commit()
            region.query_one("#console-dispatch-recovery-discard", Button).press()
        else:
            claimed = asyncio.Event()

            async def pending_context(*_args):
                claimed.set()
                await asyncio.Future()

            monkeypatch.setattr(
                controller, "_resolve_dispatch_retry_context", pending_context
            )
            region.query_one("#console-dispatch-recovery-retry_anyway", Button).press()
            await _wait_for_condition(pilot, claimed.is_set)
            assert store.dispatch_recovery_for_session(session_id).in_flight
            worker = next(
                worker
                for worker in console.workers
                if worker.group == "console-dispatch-recovery-action"
            )
            worker.cancel()
        await _wait_for_condition(pilot, lambda: len(syncs) == 1)
        await pilot.pause()
        recovery = store.dispatch_recovery_for_session(session_id)
        assert recovery is not None and not recovery.in_flight
        if first_attempt == "discard_failure":
            connection.execute("DROP TRIGGER fail_discard_delete")
            connection.commit()
        region.query_one("#console-dispatch-recovery-discard", Button).press()
        await _wait_for_condition(pilot, lambda: len(syncs) == 2)
        await pilot.pause()
        assert store.dispatch_recovery_for_session(session_id) is None
        assert store.get_message("assistant-1").status == "discarded"
        assert not composer.query_one("#console-send-message", Button).disabled
        assert (
            composer.query_one("#console-send-disabled-reason", Static).styles.display
            == "none"
        )
