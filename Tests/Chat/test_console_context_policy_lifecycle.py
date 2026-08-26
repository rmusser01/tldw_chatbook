from __future__ import annotations

from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession, ConsoleChatStore
from tldw_chatbook.Chat.console_context_policy import (
    ConsoleContextPolicyOverrides,
    ContextBudgetMode,
    ContextCompactionMode,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController


def _conversation_count(db: CharactersRAGDB) -> int:
    row = (
        db.get_connection()
        .execute("SELECT COUNT(*) FROM conversations WHERE deleted = 0")
        .fetchone()
    )
    return int(row[0])


def test_empty_tab_policy_stages_without_creating_conversation_then_flushes(
    tmp_path,
) -> None:
    db = CharactersRAGDB(tmp_path / "staged.db", client_id="staged")
    persistence = ChatPersistenceService(db)
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session(title="Unsaved")
    overrides = ConsoleContextPolicyOverrides(
        budget_mode=ContextBudgetMode.CUSTOM,
        custom_budget_tokens=18_000,
        compaction_mode=ContextCompactionMode.AUTOMATIC,
    )
    before = _conversation_count(db)

    updated, persisted = store.set_session_context_policy_overrides(
        session.id, overrides
    )

    assert persisted is True
    assert updated.persisted_conversation_id is None
    assert _conversation_count(db) == before
    assert store.session_context_policy_overrides(session.id) == overrides

    conversation_id = store.persist_session_if_needed(session.id)

    assert conversation_id is not None
    assert _conversation_count(db) == before + 1
    stored = persistence.get_conversation_context_policy(conversation_id)
    assert stored.error is None
    assert stored.overrides == overrides


def test_policy_survives_close_resume_restart_without_cross_conversation_leak(
    tmp_path,
) -> None:
    path = tmp_path / "restart.db"
    first_db = CharactersRAGDB(path, client_id="first")
    first_persistence = ChatPersistenceService(first_db)
    first_store = ConsoleChatStore(persistence=first_persistence)
    customized = first_store.create_session(title="Customized")
    inherited = first_store.create_session(title="Inherited")
    overrides = ConsoleContextPolicyOverrides(
        budget_mode=ContextBudgetMode.CUSTOM,
        custom_budget_tokens=22_000,
        compaction_mode=ContextCompactionMode.OFF,
    )
    first_store.set_session_context_policy_overrides(customized.id, overrides)
    customized_id = first_store.persist_session_if_needed(customized.id)
    inherited_id = first_store.persist_session_if_needed(inherited.id)
    assert customized_id is not None
    assert inherited_id is not None
    first_store.close_session(customized.id)
    first_store.close_session(inherited.id)
    first_db.close_connection()

    reopened_db = CharactersRAGDB(path, client_id="reopened")
    reopened_store = ConsoleChatStore(persistence=ChatPersistenceService(reopened_db))
    restored_customized = reopened_store.restore_persisted_session(
        title="Customized",
        workspace_id=None,
        persisted_conversation_id=customized_id,
        all_nodes=[],
    )
    restored_inherited = reopened_store.restore_persisted_session(
        title="Inherited",
        workspace_id=None,
        persisted_conversation_id=inherited_id,
        all_nodes=[],
    )

    assert restored_customized.context_policy_overrides == overrides
    assert restored_customized.context_policy_error is None
    assert restored_inherited.context_policy_overrides.is_empty
    assert restored_inherited.context_policy_error is None


def test_screen_state_round_trip_preserves_staged_policy() -> None:
    original = ConsoleChatSession(
        title="Unsaved",
        context_policy_overrides=ConsoleContextPolicyOverrides(
            budget_mode=ContextBudgetMode.CUSTOM,
            custom_budget_tokens=10_000,
            compaction_mode=ContextCompactionMode.ASK,
        ),
    )
    controller = ConsoleSessionController.__new__(ConsoleSessionController)

    payload = controller._console_session_to_state(original)
    restored = controller._console_session_from_state(payload)

    assert payload["context_policy_overrides"] == {
        "budget_mode": "custom",
        "custom_budget_tokens": 10_000,
        "compaction_mode": "ask",
    }
    assert restored.context_policy_overrides == original.context_policy_overrides
    assert restored.persisted_conversation_id is None


def test_corrupt_screen_policy_is_bounded_and_does_not_block_restore() -> None:
    controller = ConsoleSessionController.__new__(ConsoleSessionController)
    payload = controller._console_session_to_state(ConsoleChatSession(title="Saved"))
    payload["context_policy_overrides"] = {
        "budget_mode": "custom",
        "custom_budget_tokens": True,
    }

    restored = controller._console_session_from_state(payload)

    assert restored.title == "Saved"
    assert restored.context_policy_overrides.is_empty
    assert restored.context_policy_error == "invalid_screen_context_policy"


def test_temporary_session_normalizes_thinking_history_policy() -> None:
    store = ConsoleChatStore()

    legacy = store.create_session(ephemeral=True)
    included = store.create_session(
        ephemeral=True,
        thinking_history_policy="include",
    )
    invalid = store.create_session(
        ephemeral=True,
        thinking_history_policy="required",
    )

    assert legacy.thinking_history_policy == "auto"
    assert included.thinking_history_policy == "include"
    assert invalid.thinking_history_policy == "auto"


def test_thinking_history_policy_survives_durable_hydration(tmp_path) -> None:
    path = tmp_path / "thinking-policy.db"
    first_db = CharactersRAGDB(path, client_id="first")
    first_store = ConsoleChatStore(
        persistence=ChatPersistenceService(first_db),
    )
    original = first_store.create_session(
        title="Thinking policy",
        thinking_history_policy="exclude",
    )

    conversation_id = first_store.persist_session_if_needed(original.id)

    assert conversation_id is not None
    assert (
        first_db.get_conversation_by_id(conversation_id)["thinking_history_policy"]
        == "exclude"
    )
    first_db.close_connection()

    reopened_db = CharactersRAGDB(path, client_id="reopened")
    reopened_store = ConsoleChatStore(
        persistence=ChatPersistenceService(reopened_db),
    )
    restored = reopened_store.restore_persisted_session(
        title="Thinking policy",
        workspace_id=None,
        persisted_conversation_id=conversation_id,
        all_nodes=[],
    )

    assert restored.thinking_history_policy == "exclude"
    reopened_db.close_connection()


def test_screen_state_round_trip_preserves_thinking_history_policy() -> None:
    controller = ConsoleSessionController.__new__(ConsoleSessionController)
    original = ConsoleChatSession(thinking_history_policy="include")

    payload = controller._console_session_to_state(original)
    restored = controller._console_session_from_state(payload)
    payload["thinking_history_policy"] = "required"
    invalid = controller._console_session_from_state(payload)

    assert restored.thinking_history_policy == "include"
    assert invalid.thinking_history_policy == "auto"
