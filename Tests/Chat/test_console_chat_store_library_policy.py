import asyncio

from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicyCandidate,
    ConsoleLibraryPolicyDefaults,
    ConsoleLibraryPolicyWriteStatus,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def _defaults(
    *, automatic: bool = True, allowed: bool = True
) -> ConsoleLibraryPolicyDefaults:
    return ConsoleLibraryPolicyDefaults(
        auto_retrieve=(
            ConsoleAutoRetrieve.AUTOMATIC if automatic else ConsoleAutoRetrieve.NEVER
        ),
        assistant_access=(
            ConsoleAssistantLibraryAccess.ALLOWED
            if allowed
            else ConsoleAssistantLibraryAccess.BLOCKED
        ),
    )


def test_new_session_captures_current_defaults_without_following_later_changes():
    store = ConsoleChatStore(library_policy_defaults=_defaults())
    first = store.create_session()

    store.set_library_policy_defaults(_defaults(automatic=False, allowed=False))
    second = store.create_session()

    assert first.library_policy_holder.snapshot.auto_retrieve is ConsoleAutoRetrieve.AUTOMATIC
    assert (
        first.library_policy_holder.snapshot.assistant_access
        is ConsoleAssistantLibraryAccess.ALLOWED
    )
    assert second.library_policy_holder.snapshot.auto_retrieve is ConsoleAutoRetrieve.NEVER
    assert (
        second.library_policy_holder.snapshot.assistant_access
        is ConsoleAssistantLibraryAccess.BLOCKED
    )


def test_first_persistence_inserts_even_unedited_policy_and_publishes_after_commit(
    tmp_path,
):
    db = CharactersRAGDB(tmp_path / "first-policy.db", "policy-test")
    service = ChatPersistenceService(db)
    store = ConsoleChatStore(persistence=service, library_policy_defaults=_defaults())
    session = store.create_session(title="Atomic policy")

    conversation_id = store.persist_session_if_needed(session.id)

    assert conversation_id is not None
    row = service.console_library_policy_repository.read(conversation_id)
    assert row.snapshot.source == "durable"
    assert row.snapshot.policy_revision == 1
    assert row.snapshot.auto_retrieve is ConsoleAutoRetrieve.AUTOMATIC
    assert row.snapshot.assistant_access is ConsoleAssistantLibraryAccess.ALLOWED
    assert session.library_policy_holder.snapshot == row.snapshot


def test_restored_missing_policy_is_fail_closed_and_write_free_until_explicit_save(
    tmp_path,
):
    db = CharactersRAGDB(tmp_path / "missing-policy.db", "policy-test")
    service = ChatPersistenceService(db)
    conversation_id = service.create_conversation(conversation_title="Legacy")
    store = ConsoleChatStore(persistence=service, library_policy_defaults=_defaults())

    session = store.restore_persisted_session(
        title="Legacy",
        workspace_id=None,
        persisted_conversation_id=conversation_id,
        all_nodes=(),
    )

    snapshot = session.library_policy_holder.snapshot
    assert snapshot.source == "missing"
    assert snapshot.auto_retrieve is ConsoleAutoRetrieve.NEVER
    assert snapshot.assistant_access is ConsoleAssistantLibraryAccess.BLOCKED
    assert service.console_library_policy_repository.read(conversation_id).durable_policy is None

    store.stage_session_library_policy(
        session.id,
        ConsoleLibraryPolicyCandidate(
            ConsoleAutoRetrieve.AUTOMATIC,
            ConsoleAssistantLibraryAccess.ALLOWED,
        ),
    )
    result = asyncio.run(store.save_session_library_policy(session.id))

    assert result.status is ConsoleLibraryPolicyWriteStatus.COMMITTED
    assert result.snapshot.policy_revision == 1


def test_committed_save_publishes_to_sibling_holders_and_close_unregisters(tmp_path):
    db = CharactersRAGDB(tmp_path / "policy-publication.db", "policy-test")
    service = ChatPersistenceService(db)
    conversation_id = service.create_conversation(conversation_title="Shared")
    assert (
        service.console_library_policy_repository.insert(
            conversation_id,
            ConsoleLibraryPolicyCandidate(
                ConsoleAutoRetrieve.NEVER,
                ConsoleAssistantLibraryAccess.BLOCKED,
            ),
        ).status
        is ConsoleLibraryPolicyWriteStatus.COMMITTED
    )
    store = ConsoleChatStore(persistence=service, library_policy_defaults=_defaults())
    first = store.restore_persisted_session(
        title="Shared",
        workspace_id=None,
        persisted_conversation_id=conversation_id,
        all_nodes=(),
    )
    second = store.restore_persisted_session(
        title="Shared again",
        workspace_id=None,
        persisted_conversation_id=conversation_id,
        all_nodes=(),
    )
    store.stage_session_library_policy(
        first.id,
        ConsoleLibraryPolicyCandidate(
            ConsoleAutoRetrieve.AUTOMATIC,
            ConsoleAssistantLibraryAccess.ALLOWED,
        ),
    )

    result = asyncio.run(store.save_session_library_policy(first.id))

    assert result.status is ConsoleLibraryPolicyWriteStatus.COMMITTED
    assert second.library_policy_holder.snapshot == result.snapshot
    store.close_session(first.id)
    assert first.id not in store.library_policy_coordinator._holders
    assert second.id in store.library_policy_coordinator._holders
