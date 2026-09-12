"""V37 -> V38 message_trajectory_metadata sidecar migration and accessors."""

from __future__ import annotations

from pathlib import Path

from Tests.ChaChaNotesDB.historical_bootstrap import (
    chachanotes_db_at_version,
    open_current_chachanotes_from_legacy,
)

from tldw_chatbook.DB.ChaChaNotes_DB import (
    CharactersRAGDB,
    TrajectoryRowWrite,
)

SCHEMA_NAME = "rag_char_chat_schema"

TRAJECTORY_COLUMNS = {
    "message_id",
    "conversation_id",
    "turn_id",
    "seq",
    "event_kind",
    "step_started_at",
    "first_token_at",
    "completed_at",
    "model",
    "provider",
    "payload_json",
}


def _version(connection) -> int:
    row = connection.execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (SCHEMA_NAME,),
    ).fetchone()
    return int(row[0])


def test_migrates_v37_to_v38_and_creates_table(tmp_path: Path) -> None:
    db = CharactersRAGDB(tmp_path / "test.db", client_id="test")
    connection = db.get_connection()
    assert _version(connection) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
    cols = {row["name"] for row in connection.execute(
        "PRAGMA table_info(message_trajectory_metadata)"
    )}
    assert TRAJECTORY_COLUMNS <= cols
    indexes = list(connection.execute(
        "PRAGMA index_list(message_trajectory_metadata)"
    ))
    idx = {row["name"] for row in indexes}
    assert any("conv_seq" in name for name in idx), idx
    # Ledger-ordering guarantee: the (conversation_id, seq) index is UNIQUE.
    conv_seq = next(row for row in indexes if "conv_seq" in row["name"])
    assert conv_seq["unique"] == 1, conv_seq
    # Local-only: no sync triggers may mention the sidecar table.
    triggers = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger'"
        )
    }
    assert not any("trajectory" in t.lower() for t in triggers), triggers


def test_upsert_and_read_roundtrip(tmp_path: Path) -> None:
    db = CharactersRAGDB(tmp_path / "test.db", client_id="test")
    conv = db.add_conversation({"title": "t"})
    msg = db.add_message(
        {"conversation_id": conv, "sender": "user", "content": "hi"}
    )
    db.upsert_trajectory_rows(
        [
            TrajectoryRowWrite(
                message_id=msg,
                conversation_id=conv,
                turn_id=msg,
                seq=None,
                event_kind="user",
                step_started_at=1.0,
                first_token_at=None,
                completed_at=None,
                model=None,
                provider=None,
                payload_json=None,
            )
        ]
    )
    rows = db.get_trajectory_rows(conv)
    assert len(rows) == 1
    assert rows[0].seq == 1
    assert rows[0].event_kind == "user"

    # Multiple tool_calls under one assistant message: distinct seqs.
    db.upsert_trajectory_rows(
        [
            TrajectoryRowWrite(
                message_id=msg,
                conversation_id=conv,
                turn_id=msg,
                seq=None,
                event_kind="tool_call",
                step_started_at=1.0,
                first_token_at=None,
                completed_at=2.0,
                model="m",
                provider="p",
                payload_json='{"n":1}',
            ),
            TrajectoryRowWrite(
                message_id=msg,
                conversation_id=conv,
                turn_id=msg,
                seq=None,
                event_kind="tool_call",
                step_started_at=1.0,
                first_token_at=None,
                completed_at=3.0,
                model="m",
                provider="p",
                payload_json='{"n":2}',
            ),
        ]
    )
    rows = db.get_trajectory_rows(conv)
    assert [r.seq for r in rows] == [1, 2, 3]


def test_explicit_seq_upsert_updates_existing_row(tmp_path: Path) -> None:
    db = CharactersRAGDB(tmp_path / "test.db", client_id="test")
    conv = db.add_conversation({"title": "t"})
    msg = db.add_message(
        {"conversation_id": conv, "sender": "assistant", "content": "hi"}
    )
    db.upsert_trajectory_rows(
        [
            TrajectoryRowWrite(
                message_id=msg,
                conversation_id=conv,
                turn_id=msg,
                seq=1,
                event_kind="assistant",
                step_started_at=1.0,
                first_token_at=1.5,
                completed_at=2.0,
                model="m",
                provider="p",
                payload_json=None,
            )
        ]
    )
    # Same (message_id, event_kind, seq): update in place, not a new row.
    db.upsert_trajectory_rows(
        [
            TrajectoryRowWrite(
                message_id=msg,
                conversation_id=conv,
                turn_id=msg,
                seq=1,
                event_kind="assistant",
                step_started_at=1.0,
                first_token_at=1.4,
                completed_at=2.5,
                model="m2",
                provider="p2",
                payload_json='{"updated":true}',
            )
        ]
    )
    rows = db.get_trajectory_rows(conv)
    assert len(rows) == 1
    assert rows[0].completed_at == 2.5
    assert rows[0].model == "m2"
    assert db.get_next_trajectory_seq(conv) == 2


def test_get_trajectory_rows_includes_soft_deleted_messages(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "test.db", client_id="test")
    conv = db.add_conversation({"title": "t"})
    msg = db.add_message(
        {"conversation_id": conv, "sender": "user", "content": "hi"}
    )
    db.upsert_trajectory_rows(
        [
            TrajectoryRowWrite(
                message_id=msg,
                conversation_id=conv,
                turn_id=msg,
                seq=None,
                event_kind="user",
                step_started_at=1.0,
                first_token_at=None,
                completed_at=None,
                model=None,
                provider=None,
                payload_json=None,
            )
        ]
    )
    db.soft_delete_message(msg, expected_version=1)
    # Sidecar rows survive message soft deletion; the projection filters.
    rows = db.get_trajectory_rows(conv)
    assert len(rows) == 1


def test_v37_database_upgrades(tmp_path: Path) -> None:
    with chachanotes_db_at_version(tmp_path / "seed.db", 37, client_id="seed"):
        pass
    db = open_current_chachanotes_from_legacy(
        tmp_path / "seed.db", client_id="upgraded"
    )
    connection = db.get_connection()
    assert _version(connection) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
    cols = {row["name"] for row in connection.execute(
        "PRAGMA table_info(message_trajectory_metadata)"
    )}
    assert TRAJECTORY_COLUMNS <= cols
