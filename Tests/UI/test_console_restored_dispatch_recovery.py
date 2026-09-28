"""Restored recovery controls remain usable after a refused action (GH-2708)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
from textual.widgets import Button

from Tests.Chat.test_console_dispatch_recovery import (
    _acceptance,
    _database,
    _insert,
    _NoReplayGateway,
    _restored_store,
    _start,
)
from Tests.UI.test_console_dictation import _mounted_console, _ready_host
from Tests.UI.test_console_dispatch_recovery import _state
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.UI.Console_Modules.dispatch_recovery import (
    ConsoleDispatchRecoveryRegion,
)
from tldw_chatbook.Widgets.Console import ConsoleComposerBar


def test_recovery_polling_does_not_rearm_an_unclaimed_duplicate_intent():
    """Only a new store snapshot may acknowledge the pending local click."""
    state = _state(started=True)
    intents = []
    region = ConsoleDispatchRecoveryRegion(
        state,
        session_id="restored-session",
        on_action=lambda *intent: intents.append(intent),
    )
    discard = Button("Discard", id="console-dispatch-recovery-discard")
    region.on_button_pressed(Button.Pressed(discard))
    region.sync_recovery("restored-session", state)
    region.on_button_pressed(Button.Pressed(discard))
    assert intents == [("restored-session", "assistant-1", "discard")]

    # Store claim/release may happen between paints and return an equal value.
    released = replace(state)
    region.sync_recovery("restored-session", released)
    region.on_button_pressed(Button.Pressed(discard))
    assert intents == [("restored-session", "assistant-1", "discard")] * 2

    region.sync_recovery("restored-session", released.with_in_flight(True))
    region.on_button_pressed(Button.Pressed(discard))
    assert len(intents) == 2


class _UnavailableGateway(_NoReplayGateway):
    def cached_context_window(self, _selection):
        return 4096

    async def resolve_for_send(self, _selection):
        self.resolve_calls += 1
        return SimpleNamespace(ready=False, visible_copy="Test provider unavailable.")


@pytest.mark.asyncio
@pytest.mark.parametrize("first_action", ["retry_anyway", "discard"])
async def test_restored_recovery_can_discard_after_refusal(tmp_path, first_action):
    """A fast failed action must not leave the visible buttons locally latched."""
    db, conversation_id, repository = _database(tmp_path / "restored.sqlite")
    _start(repository, _insert(db, repository, _acceptance(conversation_id)))
    store, session_id = _restored_store(db, conversation_id)
    gateway = _UnavailableGateway(db)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="llama_cpp",
        model="test-model",
        agent_runtime_enabled=False,
    )
    if first_action == "discard":
        with db.transaction() as cursor:
            cursor.execute(
                "CREATE TRIGGER fail_discard_delete BEFORE DELETE ON "
                "console_dispatch_checkpoints BEGIN SELECT RAISE(ABORT, 'fail'); END"
            )

    _app, host = _ready_host()
    try:
        async with host.run_test(size=(100, 34)) as pilot:
            console = await _mounted_console(host, pilot)
            runtime = console._console_runtime()
            runtime.set_chat_store(store)
            runtime.set_provider_gateway(gateway)
            runtime.set_chat_controller(controller)
            runtime.attach_view(console)
            store.set_session_draft(session_id, "next message")
            composer = console.query_one("#console-native-composer", ConsoleComposerBar)
            composer.load_draft("next message")
            console._sync_console_composer_action_state(can_save_chatbook=False)
            await pilot.pause()
            region = console.query_one(
                "#console-dispatch-recovery", ConsoleDispatchRecoveryRegion
            )
            assert region.display is True
            assert composer.query_one("#console-send-message", Button).disabled
            before = store.dispatch_recovery_for_session(session_id)
            assert before is not None
            assert gateway.resolve_calls == 0
            assert gateway.provider_states == []

            region.query_one(
                f"#console-dispatch-recovery-{first_action}", Button
            ).press()
            await pilot.pause()
            await host.workers.wait_for_complete()
            await pilot.pause()
            assert store.dispatch_recovery_for_session(session_id) == before
            assert gateway.provider_states == []
            if first_action == "retry_anyway":
                assert gateway.resolve_calls == 1
            else:
                with db.transaction() as cursor:
                    cursor.execute("DROP TRIGGER fail_discard_delete")

            # The controller has returned the same actionable state. Its next
            # mounted action must still settle the original durable owner.
            discard = region.query_one("#console-dispatch-recovery-discard", Button)
            assert discard.disabled is False
            discard.press()
            await pilot.pause()
            await host.workers.wait_for_complete()
            await pilot.pause()
            assert store.dispatch_recovery_for_session(session_id) is None
            assert db.get_message_by_id("user-1")["content"] == "hello"
            assert db.get_message_by_id("assistant-1")[
                "assistant_generation_state"
            ] == ("discarded")
            assert (
                db.get_connection()
                .execute("SELECT COUNT(*) FROM messages")
                .fetchone()[0]
                == 2
            )
            assert (
                db.get_connection()
                .execute("SELECT COUNT(*) FROM console_dispatch_checkpoints")
                .fetchone()[0]
                == 0
            )
            assert gateway.provider_states == []
            assert region.display is False
            send = composer.query_one("#console-send-message", Button)
            assert not send.disabled, (send.tooltip, composer.draft_text())
    finally:
        db.close_connection()
