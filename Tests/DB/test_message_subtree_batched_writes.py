"""A subtree delete and its Undo write their rows in batches, not one by one.

TASK-33628.5. Console Delete tombstones a message and every message beneath
it; Undo puts exactly those rows back. Both ran on the UI loop, and both
issued one statement per row:

* ``restore_message_subtree`` undeleted each row with its own ``UPDATE``.
* ``soft_delete_message_subtree`` tombstones the subtree in one recursive
  ``UPDATE``, but then attached each row's content-free Sync delete proof
  (``base_payload_hash``) to its trigger-written ``sync_log`` intent with its
  own ``UPDATE``.

Measured on a file-backed database (dev 7d155170dc), a 2,999-row subtree
took 0.45-1.1 s to delete and 1.0-1.8 s to restore; the per-row loops were
the largest share of both.

The pins count the statements SQLite actually ran (``set_trace_callback``),
so they fail on the per-row implementation however fast the machine is.
The other cases check that batching kept what the per-row writes
guaranteed: every proof still lands on exactly one intent, a changed row
still refuses the whole Undo, and the FTS index follows both directions.
"""

from __future__ import annotations

import math
import sqlite3

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
)

#: Rows in the subtree; large enough that a per-row loop is unmistakable and
#: that the writes span more than one id batch.
_ROWS = 1_200
#: The id batch ``_bounded_id_batches`` uses at the default variable limit.
_BATCH = 500


def _chain(db: CharactersRAGDB, count: int) -> tuple[str, list[str]]:
    conversation_id = db.add_conversation({"title": "Batched subtree"})
    ids: list[str] = []
    parent = None
    with db.transaction():
        for index in range(count):
            parent = db.add_message(
                {
                    "conversation_id": conversation_id,
                    "sender": "user" if index % 2 == 0 else "assistant",
                    "role": "user" if index % 2 == 0 else "assistant",
                    "content": f"token{index:05d} body",
                    "parent_message_id": parent,
                }
            )
            ids.append(parent)
    return conversation_id, ids


def _top_level(statements: list[str], prefix: str) -> set[str]:
    """Distinct statements the method issued that start with ``prefix``.

    The trace reports bound values inline, and it re-reports the outer
    statement once for every trigger program that statement fires; a set of
    the texts counts each issued statement once (a per-row loop's texts
    differ by their bound id).
    """
    return {s for s in statements if s.lstrip().upper().startswith(prefix)}


def _traced(db: CharactersRAGDB, call):
    conn = db.get_connection()
    statements: list[str] = []
    conn.set_trace_callback(statements.append)
    try:
        result = call()
    finally:
        conn.set_trace_callback(None)
    return result, statements


def _live(db: CharactersRAGDB, ids: list[str]) -> list[bool]:
    with db.transaction() as conn:
        rows = {
            row["id"]: not row["deleted"]
            for row in conn.execute(
                "SELECT id, deleted FROM messages WHERE conversation_id = "
                "(SELECT conversation_id FROM messages WHERE id = ?)",
                (ids[0],),
            ).fetchall()
        }
    return [rows[message_id] for message_id in ids]


def _fts_hits(db: CharactersRAGDB, tokens: list[str]) -> int:
    with db.transaction() as conn:
        return sum(
            conn.execute(
                "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH ?",
                (f'"{token}"',),
            ).fetchone()[0]
            for token in tokens
        )


def test_subtree_delete_attaches_sync_proofs_in_batches():
    """AC#1/#3: no per-row ``UPDATE sync_log`` while tombstoning a subtree."""
    db = CharactersRAGDB(":memory:", "batched-delete")
    _conversation_id, ids = _chain(db, _ROWS)
    first = db.get_message_by_id(ids[1])

    rows, statements = _traced(
        db, lambda: db.soft_delete_message_subtree(ids[1], first["version"])
    )

    assert len(rows) == _ROWS - 1
    attaches = _top_level(statements, "UPDATE SYNC_LOG")
    # The per-row implementation ran one per tombstone (1,199 here).
    assert 0 < len(attaches) <= math.ceil((_ROWS - 1) / 100), len(attaches)
    tombstones = _top_level(statements, "WITH RECURSIVE SUBTREE")
    assert len(tombstones) == 2  # the proof SELECT and the one tombstone UPDATE
    assert _live(db, ids) == [True] + [False] * (_ROWS - 1)
    # Every intent still carries exactly its own row's proof.
    with db.transaction() as conn:
        proofs = conn.execute(
            "SELECT entity_id, version, json_extract(payload, '$.base_payload_hash') "
            "AS proof FROM sync_log WHERE entity = 'messages' "
            "AND operation = 'delete'"
        ).fetchall()
    by_id = {row["entity_id"]: row for row in proofs}
    assert set(by_id) == set(ids[1:])
    assert all(row["proof"] for row in proofs)
    assert len({row["proof"] for row in proofs}) == len(proofs)
    for row in rows:
        assert by_id[row["message_id"]]["version"] == row["version"]


def test_subtree_restore_undeletes_in_batches():
    """AC#1/#3: Undo issues one ``UPDATE messages`` per id batch, not per row."""
    db = CharactersRAGDB(":memory:", "batched-restore")
    _conversation_id, ids = _chain(db, _ROWS)
    first = db.get_message_by_id(ids[1])
    rows = db.soft_delete_message_subtree(ids[1], first["version"])
    tombstones = [(row["message_id"], row["version"]) for row in rows]

    restored, statements = _traced(db, lambda: db.restore_message_subtree(tombstones))

    updates = _top_level(statements, "UPDATE MESSAGES")
    # The per-row implementation ran one per tombstone (1,199 here).
    assert len(updates) == math.ceil(len(tombstones) / _BATCH), len(updates)
    assert _live(db, ids) == [True] * _ROWS
    assert {(row["message_id"], row["version"]) for row in restored} == {
        (message_id, version + 1) for message_id, version in tombstones
    }
    for message_id, version in tombstones:
        assert db.get_message_by_id(message_id)["version"] == version + 1


def test_batched_writes_keep_the_fts_index_in_step():
    """The FTS triggers still fire per row: delete drops, Undo re-adds."""
    db = CharactersRAGDB(":memory:", "batched-fts")
    _conversation_id, ids = _chain(db, 40)
    tokens = [f"token{index:05d}" for index in range(40)]
    assert _fts_hits(db, tokens) == 40
    first = db.get_message_by_id(ids[10])

    rows = db.soft_delete_message_subtree(ids[10], first["version"])
    assert _fts_hits(db, tokens) == 10
    db.restore_message_subtree([(row["message_id"], row["version"]) for row in rows])

    assert _fts_hits(db, tokens) == 40


def test_batched_restore_refuses_a_changed_row_atomically():
    """One drifted tombstone in a later batch still restores nothing."""
    db = CharactersRAGDB(":memory:", "batched-conflict")
    _conversation_id, ids = _chain(db, _ROWS)
    first = db.get_message_by_id(ids[1])
    rows = db.soft_delete_message_subtree(ids[1], first["version"])
    tombstones = [(row["message_id"], row["version"]) for row in rows]
    # The LAST pair is stale: every earlier batch is valid.
    stale = tombstones[:-1] + [(tombstones[-1][0], tombstones[-1][1] - 1)]

    with pytest.raises(ConflictError):
        db.restore_message_subtree(stale)

    assert _live(db, ids) == [True] + [False] * (_ROWS - 1)


def test_batched_proof_attach_still_requires_exactly_one_intent():
    """A tombstone whose delete intent is missing still fails the delete."""
    db = CharactersRAGDB(":memory:", "batched-proof")
    _conversation_id, ids = _chain(db, 6)
    first = db.get_message_by_id(ids[1])
    conn = db.get_connection()
    # Swallow ONE row's trigger-written delete intent.
    conn.execute("CREATE TEMP TABLE swallowed_intents(entity_id TEXT)")
    conn.execute("INSERT INTO temp.swallowed_intents VALUES (?)", (ids[3],))
    conn.execute(
        "CREATE TEMP TRIGGER drop_one_intent AFTER INSERT ON main.sync_log "
        "WHEN NEW.entity = 'messages' AND NEW.operation = 'delete' "
        "AND NEW.entity_id IN (SELECT entity_id FROM temp.swallowed_intents) "
        "BEGIN DELETE FROM sync_log WHERE change_id = NEW.change_id; END"
    )
    try:
        with pytest.raises(CharactersRAGDBError, match="uniquely attached"):
            db.soft_delete_message_subtree(ids[1], first["version"])
    finally:
        conn.execute("DROP TRIGGER IF EXISTS temp.drop_one_intent")
        conn.execute("DROP TABLE IF EXISTS temp.swallowed_intents")

    assert _live(db, ids) == [True] * 6


@pytest.mark.parametrize("extra", [0, 1], ids=["at-limit", "one-past"])
def test_batched_writes_fit_a_tiny_variable_limit(extra):
    """The batched statements also bind their fixed values within the limit."""
    limit = 16
    db = CharactersRAGDB(":memory:", "batched-limit")
    _conversation_id, ids = _chain(db, limit + extra + 1)
    conn = db.get_connection()
    default_limit = conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
    conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, limit)
    try:
        first = db.get_message_by_id(ids[1])
        rows = db.soft_delete_message_subtree(ids[1], first["version"])
        restored = db.restore_message_subtree(
            [(row["message_id"], row["version"]) for row in rows]
        )
    finally:
        conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, default_limit)

    assert len(rows) == len(restored) == limit + extra
    assert _live(db, ids) == [True] * len(ids)
