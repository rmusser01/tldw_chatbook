from __future__ import annotations

from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_context_policy import (
    ConsoleContextPolicyOverrides,
    ContextCompactionMode,
)
from tldw_chatbook.Chat.console_context_repository import (
    ConsoleContextRepository,
    ContextPolicyWriteStatus,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def test_context_policy_cas_covers_insert_update_delete_and_conflict(tmp_path) -> None:
    db = CharactersRAGDB(tmp_path / "context-policy-cas.db", "task-22515")
    try:
        service = ChatPersistenceService(db)
        conversation_id = service.create_conversation(conversation_title="CAS")
        repository = ConsoleContextRepository(db)
        ask = ConsoleContextPolicyOverrides(compaction_mode=ContextCompactionMode.ASK)
        off = ConsoleContextPolicyOverrides(compaction_mode=ContextCompactionMode.OFF)

        inserted = repository.save_policy_if_revision(
            conversation_id,
            ask,
            expected_revision=None,
        )
        stale = repository.save_policy_if_revision(
            conversation_id,
            off,
            expected_revision=None,
        )
        updated = repository.save_policy_if_revision(
            conversation_id,
            off,
            expected_revision=inserted.revision,
        )
        stale_delete = repository.save_policy_if_revision(
            conversation_id,
            ConsoleContextPolicyOverrides(),
            expected_revision=inserted.revision,
        )
        deleted = repository.save_policy_if_revision(
            conversation_id,
            ConsoleContextPolicyOverrides(),
            expected_revision=updated.revision,
        )

        assert inserted.status is ContextPolicyWriteStatus.WRITTEN
        assert inserted.revision == 1
        assert stale.status is ContextPolicyWriteStatus.CONFLICT
        assert stale.revision == 1
        assert updated.status is ContextPolicyWriteStatus.WRITTEN
        assert updated.revision == 2
        assert stale_delete.status is ContextPolicyWriteStatus.CONFLICT
        assert stale_delete.revision == 2
        assert deleted.status is ContextPolicyWriteStatus.WRITTEN
        assert deleted.revision is None
        assert repository.load_policy(conversation_id).revision is None
    finally:
        db.close_connection()


def test_context_policy_cas_reports_missing_conversation_without_row(tmp_path) -> None:
    db = CharactersRAGDB(tmp_path / "context-policy-missing.db", "task-22515")
    try:
        repository = ConsoleContextRepository(db)

        result = repository.save_policy_if_revision(
            "missing-conversation",
            ConsoleContextPolicyOverrides(
                compaction_mode=ContextCompactionMode.AUTOMATIC
            ),
            expected_revision=None,
        )

        assert result.status is ContextPolicyWriteStatus.MISSING
        assert result.revision is None
    finally:
        db.close_connection()


def test_chat_persistence_service_exposes_revision_guard_without_breaking_legacy(
    tmp_path,
) -> None:
    db = CharactersRAGDB(tmp_path / "context-policy-service.db", "task-22515")
    try:
        service = ChatPersistenceService(db)
        conversation_id = service.create_conversation(conversation_title="Service")
        first = ConsoleContextPolicyOverrides(
            compaction_mode=ContextCompactionMode.AUTOMATIC
        )
        second = ConsoleContextPolicyOverrides(compaction_mode=ContextCompactionMode.OFF)

        revision = service.update_conversation_context_policy(
            conversation_id=conversation_id,
            overrides=first,
        )
        guarded = service.update_conversation_context_policy(
            conversation_id=conversation_id,
            overrides=second,
            expected_revision=revision,
        )

        assert revision == 1
        assert guarded.status is ContextPolicyWriteStatus.WRITTEN
        assert guarded.revision == 2
    finally:
        db.close_connection()
