import json
import uuid
from pathlib import Path

import pytest

from Tests.ChaChaNotesDB.legacy_conversation_schema import create_legacy_v13_conversations_db
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def client_id():
    return "runtime_parity_client"


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "character_persona_runtime_parity.sqlite"


@pytest.fixture
def db_instance(db_path, client_id):
    current_db_path = Path(db_path)
    for suffix in ["", "-wal", "-shm"]:
        Path(str(current_db_path) + suffix).unlink(missing_ok=True)

    db = CharactersRAGDB(current_db_path, client_id)
    try:
        yield db
    finally:
        db.close_connection()
        for suffix in ["", "-wal", "-shm"]:
            Path(str(current_db_path) + suffix).unlink(missing_ok=True)


def test_character_conversation_stores_canonical_assistant_id(db_instance: CharactersRAGDB):
    character_id = db_instance.add_character_card({"name": "Alice (Runtime Parity)"})
    conversation_id = db_instance.add_conversation(
        {
            "title": "Character Session",
            "assistant_kind": "character",
            "assistant_id": "char.local.alice",
            "character_id": character_id,
            "runtime_backend": "local",
            "discovery_owner": "ccp_character",
            "discovery_entity_id": "char.local.alice",
        }
    )

    conversation = db_instance.get_conversation_by_id(conversation_id)
    assert conversation["assistant_id"] == "char.local.alice"
    assert conversation["runtime_backend"] == "local"
    assert conversation["discovery_owner"] == "ccp_character"
    assert conversation["discovery_entity_id"] == "char.local.alice"


def test_legacy_v13_rows_default_runtime_and_discovery_metadata(db_path, client_id):
    legacy_conversation_id = str(uuid.uuid4())
    create_legacy_v13_conversations_db(
        db_path,
        [
            (
                legacy_conversation_id,
                legacy_conversation_id,
                None,
                None,
                7,
                "Legacy Character",
                None,
                "2026-04-18T02:00:00Z",
                "2026-04-18T02:00:00Z",
                0,
                client_id,
                1,
                "character",
                "7",
                None,
                "global",
                None,
                "in-progress",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        ],
    )

    db = CharactersRAGDB(db_path, client_id)
    try:
        conversation = db.get_conversation_by_id(legacy_conversation_id)
        assert conversation is not None
        assert conversation["runtime_backend"] == "local"
        assert conversation["discovery_owner"] == "general_chat"
        assert conversation["discovery_entity_id"] is None
    finally:
        db.close_connection()


def test_update_conversation_supports_runtime_and_discovery_metadata(db_instance: CharactersRAGDB):
    character_id = db_instance.add_character_card({"name": "Updater"})
    conversation_id = db_instance.add_conversation(
        {
            "title": "Update Runtime",
            "assistant_kind": "character",
            "assistant_id": "char.local.updater",
            "character_id": character_id,
            "runtime_backend": "local",
            "discovery_owner": "ccp_character",
            "discovery_entity_id": "char.local.updater",
        }
    )
    conversation = db_instance.get_conversation_by_id(conversation_id)
    assert conversation is not None

    ok = db_instance.update_conversation(
        conversation_id,
        {
            "runtime_backend": "server",
            "discovery_owner": "general_chat",
            "discovery_entity_id": "canonical.updater",
        },
        expected_version=conversation["version"],
    )
    assert ok is True

    updated = db_instance.get_conversation_by_id(conversation_id)
    assert updated is not None
    assert updated["runtime_backend"] == "server"
    assert updated["discovery_owner"] == "general_chat"
    assert updated["discovery_entity_id"] == "canonical.updater"


def test_undelete_sync_payload_includes_full_conversation_shape_with_runtime_metadata(db_instance: CharactersRAGDB):
    character_id = db_instance.add_character_card({"name": "Undeleter"})
    conversation_id = db_instance.add_conversation(
        {
            "title": "Undelete Payload",
            "assistant_kind": "character",
            "assistant_id": "char.local.undeleter",
            "character_id": character_id,
            "runtime_backend": "server",
            "discovery_owner": "ccp_character",
            "discovery_entity_id": "char.local.undeleter",
        }
    )
    created = db_instance.get_conversation_by_id(conversation_id)
    assert created is not None

    assert db_instance.soft_delete_conversation(conversation_id, expected_version=created["version"]) is True
    deleted_row = db_instance.get_conversation_by_id(conversation_id, include_deleted=True)
    assert deleted_row is not None
    assert deleted_row["deleted"] == 1

    now = db_instance._get_current_utc_timestamp_iso()
    undelete_version = int(deleted_row["version"])
    next_version = undelete_version + 1
    with db_instance.transaction() as conn:
        conn.execute(
            """
            UPDATE conversations
               SET deleted = 0,
                   last_modified = ?,
                   version = ?,
                   client_id = ?
             WHERE id = ?
               AND version = ?
            """,
            (now, next_version, db_instance.client_id, conversation_id, undelete_version),
        )

    payload_row = db_instance.execute_query(
        """
        SELECT payload
          FROM sync_log
         WHERE entity = 'conversations'
           AND entity_id = ?
         ORDER BY rowid DESC
         LIMIT 1
        """,
        (conversation_id,),
    ).fetchone()
    assert payload_row is not None
    payload = json.loads(payload_row["payload"])
    # Ensure this is a full conversation payload, not a thin tombstone.
    for key in (
        "id",
        "root_id",
        "character_id",
        "assistant_kind",
        "assistant_id",
        "scope_type",
        "workspace_id",
        "state",
        "title",
        "created_at",
        "last_modified",
        "deleted",
        "client_id",
        "version",
        "runtime_backend",
        "discovery_owner",
        "discovery_entity_id",
    ):
        assert key in payload
    assert payload["runtime_backend"] == "server"
    assert payload["discovery_owner"] == "ccp_character"
    assert payload["discovery_entity_id"] == "char.local.undeleter"
