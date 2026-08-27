"""v50 -> v51 capture-provenance migration contract."""

from __future__ import annotations

import json
import sqlite3
import zlib
from pathlib import Path

from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


SCHEMA_NAME = "rag_char_chat_schema"


def _version(connection: sqlite3.Connection) -> int:
    return int(
        connection.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (SCHEMA_NAME,),
        ).fetchone()[0]
    )


def test_real_v50_fixture_gains_safe_capture_provenance(tmp_path: Path) -> None:
    path = tmp_path / "v50.sqlite"
    with chachanotes_db_at_version(path, 50) as legacy:
        connection = legacy.get_connection()
        conversation_id = "capture-migration-conversation"
        message_id = "capture-migration-message"
        connection.execute(
            "INSERT INTO conversations(id, root_id, title, client_id) "
            "VALUES (?, ?, 'migration', 'v50-seed')",
            (conversation_id, conversation_id),
        )
        connection.execute(
            "INSERT INTO messages(id, conversation_id, sender, role, content, client_id) "
            "VALUES (?, ?, 'user', 'user', 'hi', 'v50-seed')",
            (message_id, conversation_id),
        )
        legacy_blob = zlib.compress(
            json.dumps(
                {
                    "run_tag": "legacy",
                    "seq": 0,
                    "created_at": "t",
                    "provider": "p",
                    "model": "m",
                    "endpoint": None,
                    "request": {},
                    "response": {},
                    "status": "complete",
                    "usage_json": None,
                    "omitted_keys": [],
                }
            ).encode()
        )
        connection.execute(
            "INSERT INTO message_exchanges (message_id, run_tag, seq, status, abandoned, capture_blob, created_at) "
            "VALUES (?, 'legacy', 0, 'complete', 0, ?, 't')",
            (message_id, legacy_blob),
        )
        connection.commit()

    migrated = CharactersRAGDB(path, client_id="upgrade")
    connection = migrated.get_connection()
    assert migrated._CURRENT_SCHEMA_VERSION >= 51
    assert _version(connection) == migrated._CURRENT_SCHEMA_VERSION
    row = connection.execute(
        "SELECT capture_detail FROM message_exchanges WHERE run_tag = 'legacy'"
    ).fetchone()
    assert row[0] == "safe"
    sql = connection.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'console_conversation_capture_policy'"
    ).fetchone()[0]
    assert "CHECK" in sql and "full" in sql and "safe" in sql
    migrated.close_connection()
