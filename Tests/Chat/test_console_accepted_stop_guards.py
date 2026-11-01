"""Only live temporary acceptance may bypass the controller recovery refusal."""

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
