import sqlite3
import uuid
from pathlib import Path

import pytest

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


def _create_legacy_v13_db(db_path: Path, client_id: str, rows):
    connection = sqlite3.connect(str(db_path))
    try:
        connection.executescript(
            """
            CREATE TABLE db_schema_version(
              schema_name TEXT PRIMARY KEY NOT NULL,
              version INTEGER NOT NULL
            );

            INSERT INTO db_schema_version(schema_name, version)
            VALUES('rag_char_chat_schema', 13);

            CREATE TABLE conversations(
              id TEXT PRIMARY KEY,
              root_id TEXT NOT NULL,
              forked_from_message_id TEXT,
              parent_conversation_id TEXT,
              character_id INTEGER,
              title TEXT,
              rating INTEGER,
              created_at DATETIME NOT NULL,
              last_modified DATETIME NOT NULL,
              deleted BOOLEAN NOT NULL DEFAULT 0,
              client_id TEXT NOT NULL,
              version INTEGER NOT NULL DEFAULT 1,
              assistant_kind TEXT,
              assistant_id TEXT,
              persona_memory_mode TEXT,
              scope_type TEXT NOT NULL DEFAULT 'global',
              workspace_id TEXT,
              state TEXT NOT NULL DEFAULT 'in-progress',
              topic_label TEXT,
              topic_label_source TEXT,
              topic_last_tagged_at TEXT,
              topic_last_tagged_message_id TEXT,
              cluster_id TEXT,
              source TEXT,
              external_ref TEXT
            );

            CREATE TABLE sync_log(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              entity TEXT NOT NULL,
              entity_id TEXT NOT NULL,
              operation TEXT NOT NULL,
              timestamp DATETIME NOT NULL,
              client_id TEXT NOT NULL,
              version INTEGER NOT NULL,
              payload TEXT
            );
            """
        )
        connection.executemany(
            """
            INSERT INTO conversations(
              id,
              root_id,
              forked_from_message_id,
              parent_conversation_id,
              character_id,
              title,
              rating,
              created_at,
              last_modified,
              deleted,
              client_id,
              version,
              assistant_kind,
              assistant_id,
              persona_memory_mode,
              scope_type,
              workspace_id,
              state,
              topic_label,
              topic_label_source,
              topic_last_tagged_at,
              topic_last_tagged_message_id,
              cluster_id,
              source,
              external_ref
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()
    finally:
        connection.close()


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
    _create_legacy_v13_db(
        db_path,
        client_id,
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
