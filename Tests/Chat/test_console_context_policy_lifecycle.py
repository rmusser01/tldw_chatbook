from __future__ import annotations

from types import SimpleNamespace

import pytest

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


def test_thinking_history_policy_setter_stages_then_persists(tmp_path) -> None:
    db = CharactersRAGDB(tmp_path / "thinking-policy-setter.db", client_id="setter")
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    session = store.create_session(thinking_history_policy="auto")

    staged, staged_ok = store.set_session_thinking_history_policy(session.id, "include")
    assert staged_ok is True
    assert staged.persisted_conversation_id is None
    assert store.session_thinking_history_policy(session.id) == "include"

    conversation_id = store.persist_session_if_needed(session.id)
    assert conversation_id is not None
    updated, persisted = store.set_session_thinking_history_policy(
        session.id, "exclude"
    )
    assert persisted is True
    assert updated.thinking_history_policy == "exclude"
    assert db.get_conversation_by_id(conversation_id)["thinking_history_policy"] == (
        "exclude"
    )


def test_thinking_history_policy_setter_rejects_required() -> None:
    store = ConsoleChatStore()
    session = store.create_session()
    with pytest.raises(ValueError, match="auto, include, or exclude"):
        store.set_session_thinking_history_policy(session.id, "required")


def test_screen_state_round_trip_preserves_thinking_history_policy() -> None:
    controller = ConsoleSessionController.__new__(ConsoleSessionController)
    original = ConsoleChatSession(thinking_history_policy="include")

    payload = controller._console_session_to_state(original)
    restored = controller._console_session_from_state(payload)
    payload["thinking_history_policy"] = "required"
    invalid = controller._console_session_from_state(payload)

    assert restored.thinking_history_policy == "include"
    assert invalid.thinking_history_policy == "auto"


@pytest.mark.asyncio
async def test_new_conversation_copies_resolved_thinking_default_once() -> None:
    controller = ConsoleSessionController.__new__(ConsoleSessionController)
    created = SimpleNamespace(id="new-session")
    controller._capture_console_draft_switch_snapshot = lambda: None
    controller._refresh_console_library_policy_defaults = lambda: None
    controller._active_console_session_settings = lambda: None
    controller._default_console_session_settings = lambda: SimpleNamespace(
        provider="llama_cpp",
        model="model-a",
    )
    controller._ensure_console_chat_controller_fn = lambda: SimpleNamespace(
        new_session=lambda **_kwargs: created
    )
    policy_writes: list[tuple[str, str]] = []
    controller._chat_store_accessor = lambda: SimpleNamespace(
        set_session_thinking_history_policy=lambda session_id, policy: (
            policy_writes.append((session_id, policy))
        )
    )
    controller._provider_readiness_app_config_fn = lambda: {
        "console": {"thinking_history_policy_default": "exclude"}
    }
    controller._invalidate_persisted_rows_cache_fn = lambda: None

    async def _sync() -> None:
        return None

    controller._sync_native_console_chat_ui_fn = _sync
    controller._sync_temporary_chip_fn = lambda: None
    controller._focus_composer_if_needed_fn = lambda **_kwargs: None

    await controller._create_native_console_session_from_active_context()

    assert policy_writes == [("new-session", "exclude")]
