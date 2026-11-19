"""Only live temporary acceptance may bypass the controller recovery refusal."""

import asyncio
import threading
from dataclasses import replace

import pytest

from Tests.Chat.test_console_dispatch_queue_recovery import _ephemeral_store
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleDispatchRecoveryKind
from tldw_chatbook.Chat.console_chat_store import ConsoleDispatchSettlementError
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleDispatchCheckpointState,
)


@pytest.mark.bootstrap_profile
@pytest.mark.parametrize(
    "state", ["current", "restored", "failed", "durable", "started"]
)
def test_temporary_accepted_stop_requires_live_exact_recovery(tmp_path, state):
    db, store, session_id = _ephemeral_store(tmp_path)
    controller = ConsoleChatController(
        store=store, provider_gateway=object(), agent_runtime_enabled=False
    )
    recovery = store.dispatch_recovery_for_session(session_id)
    store._message_or_raise(recovery.assistant_message_id).status = "pending"
    recovery = replace(
        recovery, runtime_active=state != "restored", recovery_needed=state == "failed"
    )
    if state == "durable":
        recovery = replace(recovery, kind=ConsoleDispatchRecoveryKind.ACCEPTED)
    if state == "started":
        recovery = replace(
            recovery,
            checkpoint=replace(
                recovery.checkpoint,
                state=ConsoleDispatchCheckpointState.DISPATCH_STARTED,
            ),
        )
    store._dispatch_recoveries_by_session[session_id] = recovery
    try:
        if state == "current":
            result = controller._mark_stream_stopped(
                recovery.assistant_message_id, visible_copy="Response stopped."
            )
            assert result.status == "stopped"
            assert store.dispatch_recovery_for_session(session_id) is None
        else:
            with pytest.raises(ConsoleDispatchSettlementError):
                controller._mark_stream_stopped(
                    recovery.assistant_message_id, visible_copy="Response stopped."
                )
            assert store.dispatch_recovery_for_session(session_id) is recovery
        assert (
            db.get_connection().execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            == 0
        )
        assert (
            db.get_connection()
            .execute("SELECT COUNT(*) FROM console_dispatch_checkpoints")
            .fetchone()[0]
            == 0
        )
    finally:
        db.close_connection()


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
@pytest.mark.parametrize(
    "state",
    [
        "current",
        "restored",
        "failed",
        "in-flight",
        "durable",
        "dispatch-started",
        "foreign-preparation",
        "foreign-assistant",
        "foreign-stream",
        "different-submission",
        "multiple-submissions",
        "reply-owner",
    ],
)
async def test_early_temporary_stop_requires_exact_registered_submit(tmp_path, state):
    from Tests.Chat.test_console_turn_preparation import _preparation_values
    from tldw_chatbook.Chat.console_pre_provider_stop import (
        request_accepted_pre_stream_stop,
    )
    from tldw_chatbook.Chat.console_turn_preparation import (
        ConsoleTurnPreparation,
        ConsoleTurnPreparationState,
    )

    db, store, session_id = _ephemeral_store(tmp_path)
    controller = ConsoleChatController(
        store=store, provider_gateway=object(), agent_runtime_enabled=False
    )
    recovery = store.dispatch_recovery_for_session(session_id)
    assistant_id = recovery.assistant_message_id
    values = _preparation_values(
        ConsoleTurnPreparationState.ACCEPTED, session_id=session_id
    )
    values.update(preparation_id=recovery.preparation_id, ephemeral=True)
    preparation = ConsoleTurnPreparation(**values)
    assert store.begin_preparation(preparation) is preparation
    recovery = replace(recovery, runtime_active=True, recovery_needed=False)
    if state == "restored":
        recovery = replace(recovery, runtime_active=False)
    elif state == "failed":
        recovery = replace(recovery, recovery_needed=True)
    elif state == "in-flight":
        recovery = replace(recovery, in_flight=True)
    elif state == "durable":
        recovery = replace(recovery, kind=ConsoleDispatchRecoveryKind.ACCEPTED)
    elif state == "dispatch-started":
        recovery = replace(
            recovery,
            checkpoint=replace(
                recovery.checkpoint,
                state=ConsoleDispatchCheckpointState.DISPATCH_STARTED,
            ),
        )
    elif state == "foreign-preparation":
        recovery = replace(recovery, preparation_id="different-preparation")
    store._dispatch_recoveries_by_session[session_id] = recovery
    task = asyncio.create_task(asyncio.Event().wait())
    other = asyncio.create_task(asyncio.Event().wait())
    controller._active_submit_tasks[task] = session_id
    controller._active_submit_preparations[task] = (
        "different-preparation"
        if state == "different-submission"
        else preparation.preparation_id
    )
    if state == "multiple-submissions":
        controller._active_submit_tasks[other] = session_id
        controller._active_submit_preparations[other] = preparation.preparation_id
    controller._active_stream_tasks[session_id] = (
        other if state == "foreign-stream" else task
    )
    controller._active_assistant_message_ids[session_id] = (
        "different-assistant" if state == "foreign-assistant" else assistant_id
    )
    if state == "reply-owner":
        controller._active_cancel_events[session_id] = threading.Event()
    controller._deferred_user_stop_markers.add("other-assistant")
    try:
        assert request_accepted_pre_stream_stop(
            controller, session_id, record_user_stop=True
        ) is (state == "current")
        assert task.cancelling() == int(state == "current")
        assert not other.cancelling()
        assert (assistant_id in controller._deferred_user_stop_markers) is (
            state == "current"
        )
        assert "other-assistant" in controller._deferred_user_stop_markers
        assert store.preparation_for_session(session_id) is preparation
        assert store.dispatch_recovery_for_session(session_id) is recovery
        assert (
            db.get_connection().execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            == 0
        )
        assert (
            db.get_connection()
            .execute("SELECT COUNT(*) FROM console_dispatch_checkpoints")
            .fetchone()[0]
            == 0
        )
    finally:
        task.cancel()
        other.cancel()
        await asyncio.gather(task, other, return_exceptions=True)
        db.close_connection()
