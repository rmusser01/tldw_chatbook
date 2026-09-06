"""Local-only Console project-instruction persistence and lifecycle tests."""

from __future__ import annotations

import inspect
import json

import pytest
from loguru import logger

from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession, ConsoleChatStore
from tldw_chatbook.Chat.console_project_instructions import (
    ProjectInstructionControlState,
    encode_project_context_json,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError


ENABLED_STATE = ProjectInstructionControlState(
    project_instructions_enabled=True,
    working_folder_binding_id="binding-7",
    working_folder_locator_fingerprint="locator-fingerprint",
    project_instruction_notice_key="notice-key",
)
WRITE_WARNING = "project_instruction_state_write_failed: the updated choice may not survive restart."


def _conversation_snapshot(db: CharactersRAGDB, conversation_id: str) -> dict:
    row = (
        db.get_connection()
        .execute(
            "SELECT version, last_modified, metadata, console_project_context_json "
            "FROM conversations WHERE id = ?",
            (conversation_id,),
        )
        .fetchone()
    )
    return dict(row)


def test_local_project_context_set_clear_is_version_and_sync_neutral(tmp_path) -> None:
    db = CharactersRAGDB(tmp_path / "local.db", client_id="local-test")
    conversation_id = db.add_conversation(
        {"title": "local state", "metadata": '{"synced":"unchanged"}'}
    )
    before = _conversation_snapshot(db, conversation_id)
    starting_change_id = db.get_latest_sync_log_change_id()

    db.set_conversation_console_project_context(
        conversation_id, encode_project_context_json(ENABLED_STATE)
    )
    assert db.get_conversation_console_project_context(conversation_id) == (
        encode_project_context_json(ENABLED_STATE)
    )
    after_set = _conversation_snapshot(db, conversation_id)
    assert after_set["version"] == before["version"]
    assert after_set["last_modified"] == before["last_modified"]
    assert after_set["metadata"] == before["metadata"]
    assert (
        db.get_sync_log_entries(
            since_change_id=starting_change_id, entity_type="conversations"
        )
        == []
    )

    db.set_conversation_console_project_context(conversation_id, None)
    assert db.get_conversation_console_project_context(conversation_id) is None
    after_clear = _conversation_snapshot(db, conversation_id)
    assert after_clear["version"] == before["version"]
    assert after_clear["last_modified"] == before["last_modified"]
    assert after_clear["metadata"] == before["metadata"]
    assert (
        db.get_sync_log_entries(
            since_change_id=starting_change_id, entity_type="conversations"
        )
        == []
    )
    db.close_connection()


def test_local_project_context_is_excluded_from_conversation_sync_triggers(
    tmp_path,
) -> None:
    db = CharactersRAGDB(tmp_path / "triggers.db", client_id="trigger-test")
    rows = (
        db.get_connection()
        .execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type = 'trigger' AND name LIKE 'conversations_sync_%'"
        )
        .fetchall()
    )

    assert {row["name"] for row in rows} == {
        "conversations_sync_create",
        "conversations_sync_update",
        "conversations_sync_delete",
        "conversations_sync_undelete",
    }
    assert all("console_project_context_json" not in (row["sql"] or "") for row in rows)
    db.close_connection()


def test_ordinary_mutations_delete_restore_and_restart_preserve_local_state(
    tmp_path,
) -> None:
    db_path = tmp_path / "preserve.db"
    db = CharactersRAGDB(db_path, client_id="preserve-test")
    conversation_id = db.add_conversation({"title": "before"})
    encoded = encode_project_context_json(ENABLED_STATE)
    db.set_conversation_console_project_context(conversation_id, encoded)

    current = db.get_conversation_by_id(conversation_id)
    db.update_conversation(
        conversation_id, {"title": "after"}, expected_version=current["version"]
    )
    assert db.get_conversation_console_project_context(conversation_id) == encoded

    current = db.get_conversation_by_id(conversation_id)
    db.soft_delete_conversation(conversation_id, expected_version=current["version"])
    deleted = db.get_conversation_by_id(conversation_id, include_deleted=True)
    assert deleted["console_project_context_json"] == encoded
    db.restore_conversation(conversation_id, expected_version=deleted["version"])
    assert db.get_conversation_console_project_context(conversation_id) == encoded
    db.close_connection()

    reopened = CharactersRAGDB(db_path, client_id="restart-test")
    assert reopened.get_conversation_console_project_context(conversation_id) == encoded
    synchronized_payloads = [
        row[0]
        for row in reopened.get_connection()
        .execute(
            "SELECT payload FROM sync_log WHERE entity = 'conversations' AND entity_id = ?",
            (conversation_id,),
        )
        .fetchall()
    ]
    assert synchronized_payloads
    assert all(
        "console_project_context_json" not in payload
        for payload in synchronized_payloads
    )
    assert all("binding-7" not in payload for payload in synchronized_payloads)
    reopened.close_connection()


def test_db_accessor_documents_future_inbound_sync_preservation_contract() -> None:
    docstring = (
        inspect.getdoc(CharactersRAGDB.set_conversation_console_project_context) or ""
    )
    normalized = " ".join(docstring.lower().split())

    assert "synchronized-column allowlist" in normalized
    for operation in ("create", "update", "delete", "undelete", "replay", "conflict"):
        assert operation in normalized


def test_persistence_service_delegates_local_project_context_accessors(
    tmp_path,
) -> None:
    db = CharactersRAGDB(tmp_path / "service.db", client_id="service-test")
    conversation_id = db.add_conversation({"title": "service"})
    service = ChatPersistenceService(db)
    encoded = encode_project_context_json(ENABLED_STATE)

    service.set_conversation_console_project_context(
        conversation_id=conversation_id,
        project_context_json=encoded,
    )

    assert (
        service.get_conversation_console_project_context(
            conversation_id=conversation_id
        )
        == encoded
    )
    db.close_connection()


@pytest.mark.parametrize("target", ["missing", "deleted"])
def test_db_project_context_set_rejects_non_active_rows(tmp_path, target) -> None:
    db = CharactersRAGDB(tmp_path / f"{target}.db", client_id="rowcount-test")
    conversation_id = "missing-secret-id"
    if target == "deleted":
        conversation_id = db.add_conversation({"title": "deleted"})
        row = db.get_conversation_by_id(conversation_id)
        db.soft_delete_conversation(conversation_id, expected_version=row["version"])

    with pytest.raises(CharactersRAGDBError, match="active conversation"):
        db.set_conversation_console_project_context(
            conversation_id, encode_project_context_json(ENABLED_STATE)
        )

    db.close_connection()


def test_session_default_is_legacy_disabled_but_store_new_session_opts_in() -> None:
    assert ConsoleChatSession().project_instruction_state == (
        ProjectInstructionControlState.legacy_disabled()
    )

    store = ConsoleChatStore()
    created = store.create_session()
    assert created.project_instruction_state == (
        ProjectInstructionControlState.new_session()
    )


@pytest.mark.parametrize("stored_json", [None, "not-json", '{"version":999}'])
def test_restore_defaults_missing_malformed_and_forward_state_to_legacy_disabled(
    stored_json: str | None,
) -> None:
    class Persistence:
        db = None

        def get_conversation_console_project_context(self, *, conversation_id: str):
            assert conversation_id == "conversation-1"
            return stored_json

    store = ConsoleChatStore(persistence=Persistence())
    session = store.restore_persisted_session(
        title="restored",
        workspace_id=None,
        persisted_conversation_id="conversation-1",
        all_nodes=[],
    )

    assert session.project_instruction_state == (
        ProjectInstructionControlState.legacy_disabled()
    )


def test_restore_decodes_valid_local_project_context() -> None:
    class Persistence:
        db = None

        def get_conversation_console_project_context(self, *, conversation_id: str):
            assert conversation_id == "conversation-1"
            return encode_project_context_json(ENABLED_STATE)

    store = ConsoleChatStore(persistence=Persistence())
    session = store.restore_persisted_session(
        title="restored",
        workspace_id=None,
        persisted_conversation_id="conversation-1",
        all_nodes=[],
    )

    assert session.project_instruction_state == ENABLED_STATE


def test_new_durable_session_persists_its_explicit_opt_in_state(tmp_path) -> None:
    db = CharactersRAGDB(tmp_path / "new-session.db", client_id="session-test")
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    session = store.create_session(title="new")

    conversation_id = store.persist_session_if_needed(session.id)

    assert conversation_id is not None
    assert db.get_conversation_console_project_context(conversation_id) == (
        encode_project_context_json(ProjectInstructionControlState.new_session())
    )
    db.close_connection()


def test_temporary_session_keeps_project_context_only_in_memory() -> None:
    class RecordingPersistence:
        db = None

        def __init__(self) -> None:
            self.writes: list[tuple[str, str | None]] = []

        def set_conversation_console_project_context(
            self, *, conversation_id: str, project_context_json: str | None
        ) -> None:
            self.writes.append((conversation_id, project_context_json))

    persistence = RecordingPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session(ephemeral=True)

    store.set_session_project_instruction_state(session.id, ENABLED_STATE)

    assert session.project_instruction_state == ENABLED_STATE
    assert persistence.writes == []


def test_failed_state_write_keeps_memory_warns_once_and_never_uses_metadata() -> None:
    class FailingPersistence:
        db = None

        def __init__(self) -> None:
            self.metadata_writes = 0

        def get_conversation_console_project_context(self, *, conversation_id: str):
            return None

        def set_conversation_console_project_context(self, **kwargs) -> None:
            raise RuntimeError("secret adapter detail")

        def update_conversation_pinned_prefill(self, **kwargs) -> bool:
            self.metadata_writes += 1
            return True

    persistence = FailingPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.restore_persisted_session(
        title="restored",
        workspace_id=None,
        persisted_conversation_id="conversation-1",
        all_nodes=[],
    )
    warnings: list[str] = []
    sink_id = logger.add(
        lambda message: warnings.append(message.record["message"]), level="WARNING"
    )
    try:
        store.set_session_project_instruction_state(session.id, ENABLED_STATE)
    finally:
        logger.remove(sink_id)

    assert session.project_instruction_state == ENABLED_STATE
    assert warnings == [WRITE_WARNING]
    assert "secret adapter detail" not in warnings[0]
    assert persistence.metadata_writes == 0


def test_missing_real_db_state_write_warns_without_creating_a_row(tmp_path) -> None:
    db_path = tmp_path / "missing.db"
    db = CharactersRAGDB(db_path, client_id="missing-test")
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    session = store.restore_persisted_session(
        title="missing",
        workspace_id=None,
        persisted_conversation_id="missing-secret-id",
        all_nodes=[],
    )
    warnings: list[str] = []
    sink_id = logger.add(
        lambda message: warnings.append(message.record["message"]), level="WARNING"
    )
    try:
        store.set_session_project_instruction_state(session.id, ENABLED_STATE)
    finally:
        logger.remove(sink_id)

    assert session.project_instruction_state == ENABLED_STATE
    assert warnings == [WRITE_WARNING]
    assert "missing-secret-id" not in warnings[0]
    db.close_connection()

    reopened = CharactersRAGDB(db_path, client_id="missing-restart")
    assert reopened.get_conversation_by_id("missing-secret-id") is None
    reopened.close_connection()


def test_deleted_real_db_state_write_warns_and_preserves_state_after_restore(
    tmp_path,
) -> None:
    db_path = tmp_path / "deleted.db"
    db = CharactersRAGDB(db_path, client_id="deleted-test")
    conversation_id = db.add_conversation({"title": "deleted"})
    encoded = encode_project_context_json(ENABLED_STATE)
    db.set_conversation_console_project_context(conversation_id, encoded)
    row = db.get_conversation_by_id(conversation_id)
    db.soft_delete_conversation(conversation_id, expected_version=row["version"])
    deleted = db.get_conversation_by_id(conversation_id, include_deleted=True)

    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    session = store.restore_persisted_session(
        title="deleted",
        workspace_id=None,
        persisted_conversation_id=conversation_id,
        all_nodes=[],
    )
    replacement = ProjectInstructionControlState.new_session()
    warnings: list[str] = []
    sink_id = logger.add(
        lambda message: warnings.append(message.record["message"]), level="WARNING"
    )
    try:
        store.set_session_project_instruction_state(session.id, replacement)
    finally:
        logger.remove(sink_id)

    assert session.project_instruction_state == replacement
    assert warnings == [WRITE_WARNING]
    after_failed_write = db.get_conversation_by_id(
        conversation_id, include_deleted=True
    )
    assert after_failed_write["console_project_context_json"] == encoded
    assert after_failed_write["version"] == deleted["version"]
    assert after_failed_write["last_modified"] == deleted["last_modified"]
    db.restore_conversation(conversation_id, expected_version=deleted["version"])
    db.close_connection()

    reopened = CharactersRAGDB(db_path, client_id="deleted-restart")
    restored = ConsoleChatStore(
        persistence=ChatPersistenceService(reopened)
    ).restore_persisted_session(
        title="restored",
        workspace_id=None,
        persisted_conversation_id=conversation_id,
        all_nodes=[],
    )
    assert restored.project_instruction_state == ENABLED_STATE
    reopened.close_connection()


def test_adapter_without_state_setter_warns_and_keeps_memory_choice() -> None:
    class GetterOnlyPersistence:
        db = None

        def __init__(self) -> None:
            self.metadata_writes = 0

        def get_conversation_console_project_context(self, *, conversation_id: str):
            return None

        def update_conversation_pinned_prefill(self, **kwargs) -> bool:
            self.metadata_writes += 1
            return True

    persistence = GetterOnlyPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.restore_persisted_session(
        title="legacy adapter",
        workspace_id=None,
        persisted_conversation_id="adapter-secret-id",
        all_nodes=[],
    )
    warnings: list[str] = []
    sink_id = logger.add(
        lambda message: warnings.append(message.record["message"]), level="WARNING"
    )
    try:
        store.set_session_project_instruction_state(session.id, ENABLED_STATE)
    finally:
        logger.remove(sink_id)

    assert session.project_instruction_state == ENABLED_STATE
    assert warnings == [WRITE_WARNING]
    assert "adapter-secret-id" not in warnings[0]
    assert persistence.metadata_writes == 0


def test_promotion_writes_project_context_inside_ordinary_durable_promotion(
    tmp_path, monkeypatch
) -> None:
    db = CharactersRAGDB(tmp_path / "promotion.db", client_id="promotion-test")
    service = ChatPersistenceService(db)
    transaction_states: list[bool] = []
    original_write = db.set_conversation_console_project_context

    def recording_write(conversation_id, project_context_json) -> None:
        transaction_states.append(db.get_connection().in_transaction)
        original_write(conversation_id, project_context_json)

    monkeypatch.setattr(db, "set_conversation_console_project_context", recording_write)
    store = ConsoleChatStore(persistence=service)
    session = store.create_session(title="temporary", ephemeral=True)
    store.set_session_project_instruction_state(session.id, ENABLED_STATE)
    store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="hello", persist=True
    )

    try:
        conversation_id = store.promote_ephemeral_session(session.id)

        assert conversation_id is not None
        assert transaction_states == [True]
        assert db.get_conversation_console_project_context(conversation_id) == (
            encode_project_context_json(ENABLED_STATE)
        )
    finally:
        db.close_connection()

    reopened = CharactersRAGDB(tmp_path / "promotion.db", client_id="reopened")
    try:
        assert reopened.get_conversation_console_project_context(conversation_id) == (
            encode_project_context_json(ENABLED_STATE)
        )
    finally:
        reopened.close_connection()


def test_promotion_state_write_failure_rolls_back_the_complete_bundle(
    tmp_path, monkeypatch
) -> None:
    db = CharactersRAGDB(tmp_path / "promotion-failure.db", client_id="promotion-test")
    service = ChatPersistenceService(db)
    store = ConsoleChatStore(persistence=service)
    session = store.create_session(title="temporary", ephemeral=True)
    store.set_session_project_instruction_state(session.id, ENABLED_STATE)
    message = store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="hello", persist=True
    )

    def fail_write(*_args, **_kwargs) -> None:
        raise RuntimeError("do not leak this detail")

    monkeypatch.setattr(db, "set_conversation_console_project_context", fail_write)

    with pytest.raises(RuntimeError, match="do not leak this detail"):
        store.promote_ephemeral_session(session.id)

    assert session.ephemeral is True
    assert session.persisted_conversation_id is None
    assert session.project_instruction_state == ENABLED_STATE
    promoted_message = store.get_message(message.id)
    assert promoted_message.persisted_message_id is None
    assert (
        db.get_connection().execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        == 0
    )
    assert (
        db.get_connection().execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
    )
    db.close_connection()


def test_persisted_json_contains_only_versioned_control_fields(tmp_path) -> None:
    db = CharactersRAGDB(tmp_path / "shape.db", client_id="shape-test")
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    session = store.create_session()
    store.set_session_project_instruction_state(session.id, ENABLED_STATE)
    conversation_id = store.persist_session_if_needed(session.id)

    payload = json.loads(db.get_conversation_console_project_context(conversation_id))
    assert set(payload) == {
        "version",
        "project_instructions_enabled",
        "working_folder_binding_id",
        "working_folder_locator_fingerprint",
        "project_instruction_notice_key",
    }
    db.close_connection()
