"""A conversation-level Capture change must not leak a database handle.

TASK-32801.4. ``replace_conversation_capture_detail`` reconciles through a
bare ``threading.Thread(name="console-capture-policy-write")``. The
repository's ``replace`` opens a thread-local sqlite connection, and every
connection this class opens is entered in a strong-referenced quiescence
registry -- so a raw thread leaves its handle, and its file descriptor and
WAL reader, alive for the life of the process. Every sibling off-loop write
in this controller already goes through ``operation_owned_connection``;
this one did not.

The count-based assertion mirrors the invariant the send path already pins
(``Tests/UI/test_console_send_refresh_scope.py:129`` and siblings assert
``registered_connection_count() == 0``).
"""

from __future__ import annotations

import pytest

from Tests.Chat.test_console_chat_controller_exchanges import StreamingGateway
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_capture_policy_repository import (
    ConsoleCapturePolicyRepository,
)
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_exchange_capture import CaptureDetail
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.mark.asyncio
async def test_capture_detail_writes_do_not_grow_the_connection_registry(tmp_path):
    chat_db = CharactersRAGDB(
        tmp_path / "capture-policy-handles.sqlite",
        "capture-policy-handles",
    )
    try:
        conversation_id = chat_db.add_conversation({"title": "policy"})
        assert conversation_id is not None
        store = ConsoleChatStore(persistence=ChatPersistenceService(chat_db))
        session = store.ensure_session()
        session.persisted_conversation_id = conversation_id
        controller = ConsoleChatController(
            store=store, provider_gateway=StreamingGateway()
        )
        controller._capture_policy_repository = ConsoleCapturePolicyRepository(chat_db)
        # The offload branch is what leaks; a memory-backed database stays
        # inline and would pass either way.
        assert controller._durable_db_call_offloadable() is True

        baseline = chat_db.registered_connection_count()
        for detail in (CaptureDetail.FULL, CaptureDetail.SAFE, CaptureDetail.FULL):
            revision = controller.capture_policy_snapshot(session.id).policy_revision
            result = await controller.replace_conversation_capture_detail(
                session.id, detail, expected_policy_revision=revision
            )
            assert result.status.value == "applied", result.status

        assert chat_db.registered_connection_count() == baseline, (
            "the capture-policy write thread left its connection registered; "
            f"{chat_db.registered_connection_count() - baseline} handle(s) "
            "leaked across three writes"
        )
        stored = ConsoleCapturePolicyRepository(chat_db).read(conversation_id)
        assert stored.policy is not None
        assert stored.policy.detail is CaptureDetail.FULL
    finally:
        chat_db.close_connection()
