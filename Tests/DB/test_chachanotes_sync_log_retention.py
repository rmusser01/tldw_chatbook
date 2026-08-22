"""`sync_log` retention: deleted content leaves, reachable proof stays (task-19564).

Why this exists -- the incident, not the rule. `sync_log` stores the COMPLETE
row as JSON, written by 35 triggers, and before this nothing ever deleted a
row. Lane 3 of the 2026-08-21 holistic review soft-deleted a conversation and
read the message body straight back out of `sync_log`; soft-deleting the
MESSAGE left it there too. "Delete" left the user's plaintext in the database
indefinitely, and every edit left a full extra copy behind forever.

The filing recommended retiring the content columns because "both readers have
zero external callers". That is stale, and this module pins the reason:
`read_committed_chat_sync_intent`, `read_committed_chat_delete_intent` and
`list_current_committed_chat_sync_intents` all compare the sync_log payload to
the live `messages` row FIELD BY FIELD, and two of them have live non-test
callers (`ConsoleChatStore.ensure_provider_continuation_durable`, which raises
when the read returns None, and `._reconcile_restored_chat_sync_intents`). So
the log is bounded to the frontier its readers can reach, not emptied of
content -- and `test_retention_does_not_break_the_committed_intent_readers`
below is the guard on that trade.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Sync_Interop.hashing import canonical_payload_hash


@pytest.fixture()
def db(tmp_path: Path) -> CharactersRAGDB:
    database = CharactersRAGDB(tmp_path / "retention.db", "retention_client")
    try:
        yield database
    finally:
        database.close_connection()


def _needle() -> str:
    return "zqx" + uuid.uuid4().hex[:12]


def _sync_log_hits(db: CharactersRAGDB, needle: str) -> list[dict]:
    """Every sync_log row whose payload still carries `needle`, whatever entity."""
    return [
        {
            "entity": row["entity"],
            "entity_id": row["entity_id"],
            "operation": row["operation"],
            "version": row["version"],
        }
        for row in db.execute_query(
            "SELECT entity, entity_id, operation, version FROM sync_log "
            "WHERE payload LIKE ?",
            (f"%{needle}%",),
        ).fetchall()
    ]


def _entries(db: CharactersRAGDB, entity: str, entity_id: str) -> list[tuple]:
    return [
        (row["operation"], row["version"])
        for row in db.execute_query(
            "SELECT operation, version FROM sync_log "
            "WHERE entity = ? AND entity_id = ? ORDER BY change_id",
            (entity, entity_id),
        ).fetchall()
    ]


# ---------------------------------------------------------------------------
# the lane's probe, per entity
# ---------------------------------------------------------------------------
def test_soft_deleting_a_message_removes_its_body_from_sync_log(db: CharactersRAGDB):
    """The lane probe, sharpened onto the entity actually being deleted."""
    needle = _needle()
    conversation_id = db.add_conversation({"title": "probe", "character_id": 1})
    message_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": f"the {needle} body",
        }
    )
    assert _sync_log_hits(db, needle) != []

    db.soft_delete_message(message_id, expected_version=1)

    assert _sync_log_hits(db, needle) == []
    # The tombstone survives -- it is the delete proof and carries no content.
    assert _entries(db, "messages", message_id) == [("delete", 2)]


def test_soft_deleting_a_note_removes_its_body_from_sync_log(db: CharactersRAGDB):
    needle = _needle()
    note_id = db.add_note(title="probe note", content=f"secret {needle}")
    assert _sync_log_hits(db, needle) != []

    db.soft_delete_note(note_id, expected_version=1)

    assert _sync_log_hits(db, needle) == []


def test_soft_deleting_a_character_removes_its_text_from_sync_log(
    db: CharactersRAGDB,
):
    needle = _needle()
    card_id = db.add_character_card(
        {"name": "probe card", "description": f"backstory {needle}"}
    )
    assert _sync_log_hits(db, needle) != []

    db.soft_delete_character_card(card_id, expected_version=1)

    assert _sync_log_hits(db, needle) == []


def test_soft_deleting_a_conversation_removes_its_title_from_sync_log(
    db: CharactersRAGDB,
):
    """A conversation's own content in `sync_log` is its title.

    Its messages are NOT deleted by this operation -- they stay `deleted = 0`
    and come back on restore -- so their frontier row is retained, exactly as
    `messages.content` is. That residue is deliberate and is what
    `test_a_live_message_keeps_only_the_frontier_its_readers_need` bounds.
    """
    needle = _needle()
    conversation_id = db.add_conversation(
        {"title": f"title {needle}", "character_id": 1}
    )
    assert _sync_log_hits(db, needle) != []

    db.soft_delete_conversation(conversation_id, expected_version=1)

    assert _sync_log_hits(db, needle) == []


def test_hard_deleting_a_message_removes_its_body_from_sync_log(db: CharactersRAGDB):
    needle = _needle()
    conversation_id = db.add_conversation({"title": "probe", "character_id": 1})
    message_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": f"hard {needle}",
        }
    )
    with db.transaction() as conn:
        conn.execute("DELETE FROM messages WHERE id = ?", (message_id,))

    assert _sync_log_hits(db, needle) == []
    assert _entries(db, "messages", message_id) == []


def test_hard_deleting_a_conversation_cascades_the_purge_to_its_messages(
    db: CharactersRAGDB,
):
    """The FK cascade fires the child trigger -- verified, not assumed.

    ``messages.conversation_id`` is ``ON DELETE CASCADE``, and SQLite fires the
    child table's AFTER DELETE trigger for a foreign-key action (``PRAGMA
    recursive_triggers`` is 0 here and governs recursion, not this). Without
    that, a hard conversation delete would leave every message body in
    ``sync_log`` with no row left to reach it from.
    """
    needle = _needle()
    conversation_id = db.add_conversation({"title": "cascade", "character_id": 1})
    message_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": f"cascade {needle}",
        }
    )
    with db.transaction() as conn:
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))

    assert db.execute_query(
        "SELECT COUNT(*) FROM messages WHERE id = ?", (message_id,)
    ).fetchone()[0] == 0
    assert _sync_log_hits(db, needle) == []


# ---------------------------------------------------------------------------
# the size story: edit history stops accumulating
# ---------------------------------------------------------------------------
def test_editing_a_message_does_not_accumulate_old_bodies(db: CharactersRAGDB):
    """Before this, every edit left its previous full text in `sync_log`."""
    conversation_id = db.add_conversation({"title": "edits", "character_id": 1})
    needles = [_needle() for _ in range(5)]
    message_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": f"draft {needles[0]}",
        }
    )
    for version, needle in enumerate(needles[1:], start=1):
        db.update_message(
            message_id, {"content": f"draft {needle}"}, expected_version=version
        )

    surviving = [needle for needle in needles if _sync_log_hits(db, needle)]

    # Only the frontier the readers need: the current body and the one below
    # it, never the whole history.
    assert surviving == needles[-2:]
    assert _entries(db, "messages", message_id) == [("update", 4), ("update", 5)]


def test_a_live_message_keeps_only_the_frontier_its_readers_need(
    db: CharactersRAGDB,
):
    conversation_id = db.add_conversation({"title": "frontier", "character_id": 1})
    message_id = db.add_message(
        {"conversation_id": conversation_id, "sender": "user", "content": "v1"}
    )
    db.update_message(message_id, {"content": "v2"}, expected_version=1)
    db.update_message(message_id, {"content": "v3"}, expected_version=2)

    assert _entries(db, "messages", message_id) == [("update", 2), ("update", 3)]


def test_editing_a_note_keeps_only_the_current_version(db: CharactersRAGDB):
    """Non-message entities have no reader at all, so only the frontier stays."""
    note_id = db.add_note(title="n", content="first")
    db.update_note(note_id, {"content": "second"}, expected_version=1)
    db.update_note(note_id, {"content": "third"}, expected_version=2)

    assert _entries(db, "notes", note_id) == [("update", 3)]


# ---------------------------------------------------------------------------
# the trade: retention must not break the readers that keep the content alive
# ---------------------------------------------------------------------------
def test_retention_does_not_break_the_committed_intent_readers(db: CharactersRAGDB):
    """The reason the content columns are retained rather than retired.

    `ensure_provider_continuation_durable` raises when
    `read_committed_chat_sync_intent` returns None, so a retention rule that
    pruned the frontier would turn every continuation checkpoint into a hard
    error while silently disabling Sync v2.
    """
    conversation_id = db.add_conversation({"title": "intents", "character_id": 1})
    message_id = db.add_message(
        {"conversation_id": conversation_id, "sender": "user", "content": "first"}
    )
    db.update_message(message_id, {"content": "second"}, expected_version=1)
    db.update_message(message_id, {"content": "third"}, expected_version=2)
    row = db.execute_query(
        "SELECT role, version FROM messages WHERE id = ?", (message_id,)
    ).fetchone()
    assert row["version"] == 3

    record = db.read_committed_chat_sync_intent(
        message_id=message_id,
        message_version=3,
        payload_hash=canonical_payload_hash({"content": "third", "role": row["role"]}),
    )

    assert record is not None
    assert record.content == "third"
    # The base hash comes from the version below the frontier; that row is
    # exactly what the retention rule keeps for live messages.
    assert record.base_payload_hash == canonical_payload_hash(
        {"content": "second", "role": row["role"]}
    )

    # And a restore-time reconcile still sees the conversation's intents.
    intents = db.list_current_committed_chat_sync_intents(conversation_id)
    assert [intent["message_id"] for intent in intents] == [message_id]
    assert intents[0]["message_version"] == 3


def test_pruning_cannot_flip_the_single_intent_ambiguity_check(db: CharactersRAGDB):
    """Retention must not turn a REJECTED intent into an accepted one.

    `list_current_committed_chat_sync_intents` accepts a message only when
    `1 = (SELECT COUNT(*) FROM sync_log ...)` at the message's CURRENT version
    -- two rows there mean the history is ambiguous and the intent is refused.
    A prune that could delete one of a duplicate pair would silently promote
    that refusal into an acceptance, changing what a sync proof asserts. The
    rule that makes it safe is that pruning only ever removes rows STRICTLY
    BELOW the frontier; this is the test that says so.
    """
    conversation_id = db.add_conversation({"title": "ambiguous", "character_id": 1})
    message_id = db.add_message(
        {"conversation_id": conversation_id, "sender": "user", "content": "v1"}
    )
    db.update_message(message_id, {"content": "v2"}, expected_version=1)
    version = db.execute_query(
        "SELECT version FROM messages WHERE id = ?", (message_id,)
    ).fetchone()["version"]

    # Forge a duplicate at the CURRENT version -- the ambiguous state.
    with db.transaction() as conn:
        duplicate = conn.execute(
            "SELECT entity_id, operation, timestamp, client_id, version, payload "
            "FROM sync_log WHERE entity = 'messages' AND entity_id = ? "
            "AND version = ?",
            (message_id, version),
        ).fetchone()
        conn.execute(
            "INSERT INTO sync_log(entity, entity_id, operation, timestamp, "
            "client_id, version, payload) VALUES ('messages',?,?,?,?,?,?)",
            tuple(duplicate),
        )

    def count_at_current() -> int:
        return db.execute_query(
            "SELECT COUNT(*) FROM sync_log WHERE entity = 'messages' "
            "AND entity_id = ? AND version = ?",
            (message_id, version),
        ).fetchone()[0]

    assert count_at_current() == 2
    assert db.list_current_committed_chat_sync_intents(conversation_id) == []

    db.prune_sync_log()

    assert count_at_current() == 2, "the frontier's row count must be untouched"
    assert db.list_current_committed_chat_sync_intents(conversation_id) == []


def test_a_tombstoned_message_still_proves_its_delete_intent(db: CharactersRAGDB):
    conversation_id = db.add_conversation({"title": "tombstone", "character_id": 1})
    message_id = db.add_message(
        {"conversation_id": conversation_id, "sender": "user", "content": "bye"}
    )
    db.soft_delete_message(message_id, expected_version=1)

    record = db.read_committed_chat_delete_intent(
        message_id=message_id,
        message_version=2,
        payload_hash=canonical_payload_hash({"deleted": True}),
    )

    assert record is not None
    assert record.message_id == message_id


# ---------------------------------------------------------------------------
# existing databases, and the maintenance surface
# ---------------------------------------------------------------------------
def test_prune_sync_log_clears_a_backlog_written_before_the_triggers(
    db: CharactersRAGDB,
):
    """The shape an existing user's database is in when it reaches v45.

    Rows are written straight into `sync_log` here, bypassing the triggers, to
    reproduce a pre-v45 backlog; `prune_sync_log` is the same sweep the
    migration performs once on upgrade.
    """
    needle = _needle()
    conversation_id = db.add_conversation({"title": "backlog", "character_id": 1})
    message_id = db.add_message(
        {"conversation_id": conversation_id, "sender": "user", "content": "v1"}
    )
    db.update_message(message_id, {"content": "v2"}, expected_version=1)
    db.update_message(message_id, {"content": "v3"}, expected_version=2)
    db.update_message(message_id, {"content": "v4"}, expected_version=3)
    assert _entries(db, "messages", message_id) == [("update", 3), ("update", 4)]

    with db.transaction() as conn:
        # Versions 1 and 2 are what a pre-v45 database would still be holding.
        for version in (1, 2):
            conn.execute(
                "INSERT INTO sync_log(entity, entity_id, operation, timestamp, "
                "client_id, version, payload) VALUES ('messages', ?, 'update', "
                "'2020-01-01T00:00:00Z', 'legacy', ?, ?)",
                (
                    message_id,
                    version,
                    json.dumps({"id": message_id, "content": f"old {needle}"}),
                ),
            )
        # An orphan: the entity row is long gone.
        conn.execute(
            "INSERT INTO sync_log(entity, entity_id, operation, timestamp, "
            "client_id, version, payload) VALUES ('notes', 'vanished', 'create', "
            "'2020-01-01T00:00:00Z', 'legacy', 1, ?)",
            (json.dumps({"id": "vanished", "content": f"orphan {needle}"}),),
        )
    assert _sync_log_hits(db, needle) != []

    removed = db.prune_sync_log()

    assert removed == 3
    assert _sync_log_hits(db, needle) == []
    # The genuine frontier rows are untouched.
    assert _entries(db, "messages", message_id) == [("update", 3), ("update", 4)]


def test_prune_sync_log_is_idempotent(db: CharactersRAGDB):
    conversation_id = db.add_conversation({"title": "idem", "character_id": 1})
    db.add_message(
        {"conversation_id": conversation_id, "sender": "user", "content": "hello"}
    )

    assert db.prune_sync_log() == 0
    assert db.prune_sync_log() == 0


def test_delete_sync_log_entries_before_matches_the_media_database_api(
    db: CharactersRAGDB,
):
    """Parity with `Client_Media_DB_v2`, which ChaChaNotes never had."""
    conversation_id = db.add_conversation({"title": "api", "character_id": 1})
    db.add_message(
        {"conversation_id": conversation_id, "sender": "user", "content": "one"}
    )
    watermark = db.get_latest_sync_log_change_id()
    assert watermark > 0
    live_rows = len(db.get_sync_log_entries())
    assert live_rows > 0

    assert db.delete_sync_log_entries_before(watermark) == live_rows
    assert db.get_sync_log_entries() == []

    with pytest.raises(ValueError):
        db.delete_sync_log_entries_before(-1)


def test_delete_sync_log_entries_rejects_non_integers(db: CharactersRAGDB):
    with pytest.raises(ValueError):
        db.delete_sync_log_entries([1, "two"])
    assert db.delete_sync_log_entries([]) == 0
