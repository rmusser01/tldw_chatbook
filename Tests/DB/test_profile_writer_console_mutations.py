"""Stock Windows Console edit, policy and recovery repairs use their original transactions."""

import sys

import pytest

from Tests.private_profile import private_profile_test


@pytest.mark.skipif(sys.platform != "win32", reason="Windows profile writer")
@private_profile_test
@pytest.mark.asyncio
async def test_console_edit_policy_and_stranded_dispatch_repair(request, monkeypatch):
    from Tests.Chat.test_console_dispatch_cursor_recovery import _stranded
    from Tests.Chat.test_console_chat_fork_persistence import _active_citation_snapshot
    from tldw_chatbook import config
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.citation_service_factory import build_local_citation_conversation_service
    from tldw_chatbook.Chat.citation_trace_repository import CitationPersistenceUnavailable, CitationTraceRepository
    from tldw_chatbook.Chat.console_context_policy import ConsoleContextPolicyOverrides, ContextCompactionMode
    from tldw_chatbook.Chat.console_context_repository import ConsoleContextRepository, ContextPolicyWriteStatus
    from tldw_chatbook.Chat.console_dispatch_repository import ConsoleDispatchRepository
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(config.get_chachanotes_db_path(), "console-edits")
    try:
        _, citations, _ = build_local_citation_conversation_service(
            db, sidecar_path=config.get_user_data_dir() / "tldw_chatbook_chat_rag_context.json", defer_key=True,
        )
        service = ChatPersistenceService(db, citation_repository=citations)
        chat = db.add_conversation({"title": "Console edits"})
        message = db.add_message({"conversation_id": chat, "sender": "assistant", "content": "Before"})

        def client_mutation(*args, **kwargs):
            raise AssertionError("client entered the mutation")

        with monkeypatch.context() as patch:
            patch.setattr(CitationTraceRepository, "transition_owner_for_message_update", client_mutation)
            patch.setattr(ConsoleContextRepository, "save_policy_if_revision", client_mutation)
            assert service.update_message_content(message_id=message, content="After", image_data=None, image_mime_type=None)
            assert db.get_message_by_id(message)["content"] == "After"
            overrides = ConsoleContextPolicyOverrides(compaction_mode=ContextCompactionMode.OFF)
            result = service.update_conversation_context_policy(conversation_id=chat, overrides=overrides, expected_revision=None)
            assert result.status is ContextPolicyWriteStatus.WRITTEN
            conflict = service.update_conversation_context_policy(conversation_id=chat, overrides=overrides, expected_revision=None)
            assert conflict.status is ContextPolicyWriteStatus.CONFLICT

        _active_citation_snapshot(db)
        original_message = dict(db.get_message_by_id("source-assistant"))
        original_owners = db.get_connection().execute("SELECT * FROM rag_message_trace_owners").fetchall()
        with monkeypatch.context() as patch:
            patch.setattr(CitationTraceRepository, "transition_owner_for_message_update", client_mutation)
            with pytest.raises(CitationPersistenceUnavailable) as refused:
                service.update_message_content(
                    message_id="source-assistant", content="Changed", image_data=None, image_mime_type=None,
                )
        assert refused.value.reason_code == "fingerprint_key_pending"
        assert dict(db.get_message_by_id("source-assistant")) == original_message
        assert db.get_connection().execute("SELECT * FROM rag_message_trace_owners").fetchall() == original_owners

        recovery_chat = db.add_conversation({"title": "Recovery"})
        dispatch = service.console_dispatch_repository
        _, _, _, greeting, checkpoint = _stranded((db, recovery_chat, dispatch))
        with db.transaction() as cursor:
            cursor.execute("UPDATE conversations SET active_leaf_message_id=? WHERE id=?", (greeting, recovery_chat))
        original_pass = ConsoleDispatchRepository._reconcile_pass

        def read_pass_only(owner, conversation_id, *, allow_writes):
            assert not allow_writes, "client entered the repair transaction"
            return original_pass(owner, conversation_id, allow_writes=allow_writes)

        with monkeypatch.context() as patch:
            patch.setattr(ConsoleDispatchRepository, "_reconcile_pass", read_pass_only)
            assert dispatch.reconcile_for_session(recovery_chat).checkpoint == checkpoint
        assert db.get_conversation_active_leaf(recovery_chat) == checkpoint.assistant_message_id
    finally:
        db.close_connection()
