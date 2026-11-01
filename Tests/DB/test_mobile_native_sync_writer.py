"""Native trigger frontiers and transactional import retention."""

import json
from types import SimpleNamespace

import pytest

from tldw_chatbook.Backup_Recovery.mobile_sync_plan import _SyncFrontier
from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.mark.parametrize("deleted", [False, True])
def test_native_sync_writer_preserves_current_and_previous_inside_rollback(
    tmp_path, deleted
):
    from tldw_chatbook.Backup_Recovery.mobile_content_graph import _ContentRow
    from tldw_chatbook.Backup_Recovery.mobile_native_sync import (
        _apply_sync,
        _prepare_sync,
        _verify_sync,
    )

    db = CharactersRAGDB(tmp_path / "chat.db", "native-sync-writer-test")
    try:
        conversation = db.add_conversation({"title": "Native sync proof"})
        message = db.add_message(
            {
                "conversation_id": conversation,
                "sender": "User",
                "role": "user",
                "content": "Original retained body",
            }
        )
        with db.transaction() as cursor:
            columns = tuple(
                row[1] for row in cursor.execute("PRAGMA table_info(messages)")
            )
            values = dict(
                zip(
                    columns,
                    cursor.execute(
                        "SELECT "
                        + ",".join(
                            "CASE WHEN 1 THEN " + column + " END" for column in columns
                        )
                        + " FROM messages WHERE id=?",
                        (message,),
                    ).fetchone(),
                    strict=True,
                )
            )
            native = cursor.execute(
                "SELECT operation,timestamp,client_id,version,payload FROM sync_log WHERE entity='messages' AND entity_id=?",
                (message,),
            ).fetchone()
            # Version two can be inserted natively: its current trigger remains
            # the owner of the current proof; the prior captured proof is retained.
            values["version"] = 2
            values["deleted"] = int(deleted)
            values["id"] = "retained-message"
            values["provider_continuation_json"] = None
            payload = json.loads(native[4])
            payload.update(id=values["id"], version=2, deleted=int(deleted))
            if deleted:
                payload = {
                    key: values[key]
                    for key in (
                        "id",
                        "deleted",
                        "last_modified",
                        "version",
                        "client_id",
                    )
                }
                payload.update(
                    assistant_generation_state=None,
                    base_payload_hash="sha256:" + "a" * 64,
                    legacy_pre_v50_base_reconstruction=False,
                )
            previous = json.loads(native[4])
            previous["id"] = values["id"]
            frontier = _SyncFrontier(
                "messages",
                message,
                values["id"],
                2,
                "delete" if deleted else "create",
                values["last_modified"],
                values["client_id"],
                json.dumps(payload),
                None,
                json.dumps(previous),
                native[0],
                native[1],
                native[2],
                None,
                "replace_native_create_with_body_free_tombstone"
                if deleted
                else "native_insert_current_unique",
                (1,),
                ("messages", (message,)),
            )
            row = _ContentRow("messages", (message,), tuple(values.items()))
            prepared = _prepare_sync(
                SimpleNamespace(rows=(row,), sync_frontiers=(frontier,)),
                _TraceReadBudget(),
            )
        with pytest.raises(RuntimeError, match="rollback"), db.transaction() as cursor:
            fields = ",".join(values)
            cursor.execute(
                "INSERT INTO messages ("
                + fields
                + ") VALUES ("
                + ",".join("?" for _ in values)
                + ")",
                tuple(values.values()),
            )
            _apply_sync(cursor, prepared)
            _verify_sync(cursor, prepared)
            rows = cursor.execute(
                "SELECT version,operation,payload FROM sync_log WHERE entity='messages' AND entity_id=? ORDER BY version",
                (values["id"],),
            ).fetchall()
            assert [(r[0], r[1]) for r in rows] == [
                (1, "create"),
                (2, "delete" if deleted else "create"),
            ]
            assert json.loads(rows[0][2]) == previous
            assert json.loads(rows[1][2]) == payload
            if deleted:
                assert "content" not in json.loads(rows[1][2])
            raise RuntimeError("rollback")
        with db.transaction() as cursor:
            assert (
                cursor.execute(
                    "SELECT count(*) FROM sync_log WHERE entity_id='retained-message'"
                ).fetchone()[0]
                == 0
            )
    finally:
        db.close_connection()
