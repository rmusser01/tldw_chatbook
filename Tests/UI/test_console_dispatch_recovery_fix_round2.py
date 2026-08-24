"""Mounted Task 15 round-2 recovery action and runtime-lifetime ratchets."""

from __future__ import annotations

import pytest
from textual.widgets import Button

from Tests.Chat.test_console_durable_turn_fix_round1 import (
    _install_real_effect_failure,
)
from Tests.Chat.test_console_first_send_atomicity import _controller
from Tests.UI.test_console_dictation import _mounted_console, _ready_host
from tldw_chatbook.UI.Console_Modules.dispatch_recovery import (
    ConsoleDispatchRecoveryRegion,
)
from tldw_chatbook.Widgets.Console import ConsoleComposerBar


@pytest.mark.asyncio
async def test_mounted_retry_resumes_interrupted_postcommit_on_app_owned_controller(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _app, host = _ready_host()
    async with host.run_test(size=(100, 34)) as pilot:
        console = await _mounted_console(host, pilot)
        db, store, controller, gateway = _controller(tmp_path)
        runtime = console._console_runtime()
        runtime.set_chat_store(store)
        runtime.set_provider_gateway(gateway)
        runtime.set_chat_controller(controller)
        counts = _install_real_effect_failure(
            controller,
            store,
            "preparation_publication",
            monkeypatch,
        )

        first = await controller.submit_draft(
            "mounted retained body", session_id="session-1"
        )

        assert first.accepted is True
        assert first.provider_started is False
        assert gateway.calls == 0
        runtime.attach_view(console)
        assert runtime.ensure_chat_controller() is controller
        recovery = store.dispatch_recovery_for_session("session-1")
        assert recovery is not None
        assert recovery.runtime_active is False
        assert recovery.recovery_needed is True
        assert (
            controller.prompt_queue_coordinator.dispatch_recovery_blocks_queue(
                "session-1"
            )
            is False
        )

        console._sync_console_composer_action_state(can_save_chatbook=False)
        await pilot.pause()
        region = console.query_one(
            "#console-dispatch-recovery", ConsoleDispatchRecoveryRegion
        )
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        send = composer.query_one("#console-send-message", Button)
        retry = region.query_one("#console-dispatch-recovery-retry_response", Button)
        assert region.display is True
        assert str(retry.label) == "Retry response"
        assert retry.disabled is False
        assert send.disabled is True

        # The actual mounted callback reaches the production Retry action; it
        # must resume the retained postcommit ledger before entering provider.
        retry.press()
        for _ in range(80):
            if (
                gateway.calls == 1
                and store.dispatch_recovery_for_session("session-1") is None
            ):
                break
            await pilot.pause(0.01)

        assert counts == {"attempts": 2, "successes": 1}
        assert gateway.calls == 1
        assert store.dispatch_recovery_for_session("session-1") is None
        assert controller._durable_postcommit_continuations == {}
        assert (
            db.get_connection().execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            == 2
        )
        assert (
            db.get_connection()
            .execute("SELECT COUNT(*) FROM console_dispatch_checkpoints")
            .fetchone()[0]
            == 0
        )
