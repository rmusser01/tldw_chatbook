"""ChaChaNotes v46 -> v47 + the deferred ``messages_fts`` backfill (task-21100).

The v45->v46 migration as first shipped (PR #1974) rebuilt the whole
``messages_fts`` index -- ``'delete-all'`` plus a reinsert of every non-deleted
message -- inside the boot path's single version-bump transaction, which froze
first paint for the duration of an O(total chat text) index rewrite on large
profiles. task-21100 keeps the ``'delete-all'`` in the migration (cheap, and it
still erases the pre-v46 corruption and every tombstoned row immediately) and
delivers the reinsert as a chunked, resumable background backfill
(``CharactersRAGDB.backfill_messages_fts`` driven by
``DB/chachanotes_fts_backfill.py``). The v46->v47 step re-guards the FTS
'delete' halves of ``messages_au``/``messages_ad`` on ``messages_fts_docsize``
membership, because an FTS 'delete' of a not-yet-backfilled row silently
poisons the doclists (and can raise ``database disk image is malformed``
depending on index state: an empty index raises, a partly-filled one absorbs
the poison with no error and a green integrity-check) --
``test_the_v46_shaped_trigger_would_corrupt_during_the_window`` keeps the
raising reproduction, and the ``_fts_data_footprint`` witnesses in the window
test catch the silent form.

Search semantics during the window, chosen deliberately: the index is
empty-but-consistent after the upgrade commits and fills oldest-rowid-first in
the background; message-content search returns progressively more history
until the backfill completes, never errors, and never returns tombstoned rows.

This module carries the repo's EXACT current-schema-version pin. It reached
here from ``test_chachanotes_sync_log_retention_migration.py`` (v46) because
the pin belongs to the NEWEST migration's own file, so a schema bump touches
the file that caused it rather than an unrelated older one (older files assert
``>=`` their own version instead). Updating the number here is a deliberate
schema-review act.

Assertion style follows ``test_fts_soft_delete_index_witness.py``: the FTS
index is queried DIRECTLY (never through a consumer that re-filters on
``deleted``), and expectations are absolute, never before/after snapshots.
"""

from __future__ import annotations

import inspect
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version
from tldw_chatbook.DB.chachanotes_fts_backfill import (
    backfill_chachanotes_messages_fts,
)
from tldw_chatbook.DB.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    SchemaError,
)

SCHEMA_NAME = CharactersRAGDB._SCHEMA_NAME


def _version(connection: sqlite3.Connection) -> int:
    return connection.execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (SCHEMA_NAME,),
    ).fetchone()[0]


def _fts_rowids(db: CharactersRAGDB, needle: str) -> list[int]:
    """Rowids the FTS index itself returns -- no join, no ``deleted`` filter."""
    return [
        row[0]
        for row in db.execute_query(
            "SELECT rowid FROM messages_fts WHERE messages_fts MATCH ? ORDER BY rowid",
            (needle,),
        ).fetchall()
    ]


def _docsize_rowids(db: CharactersRAGDB) -> set[int]:
    """The set of rowids ACTUALLY written into the index (see the
    Subscriptions backfill for why ``%_docsize`` is the only truthful
    answer for an external-content fts5 table)."""
    return {
        row[0]
        for row in db.execute_query(
            "SELECT rowid FROM messages_fts_docsize"
        ).fetchall()
    }


def _live_rowids(db: CharactersRAGDB) -> set[int]:
    return {
        row[0]
        for row in db.execute_query(
            "SELECT rowid FROM messages WHERE deleted = 0"
        ).fetchall()
    }


def _fts_data_footprint(db: CharactersRAGDB) -> tuple[int, int]:
    """(row count, total block bytes) of the index's physical storage.

    The review of this task proved that an unguarded FTS 'delete' of an
    unindexed row against a partly-filled index corrupts SILENTLY: no
    exception, integrity-check(0) green, MATCH results unchanged -- the only
    observable is `messages_fts_data` growing dangling delete-marker blocks
    (measured (3,40)->(4,70) on this schema). So the guard's behavioural
    witness must measure the storage itself.
    """
    row = db.execute_query(
        "SELECT COUNT(*), COALESCE(SUM(LENGTH(block)), 0) FROM messages_fts_data"
    ).fetchone()
    return (row[0], row[1])


def _assert_index_structurally_sound(db: CharactersRAGDB) -> None:
    """FTS5 structural integrity check, NOT the external-content comparison.

    The flag form ``('integrity-check', 0)`` checks the inverted index's own
    consistency without comparing it to the ``messages`` table -- the
    comparison form legitimately fails both during the backfill window (live
    rows not yet indexed) and at steady state (tombstoned rows deliberately
    absent). The flag form needs SQLite >= 3.42; on older runtimes the MATCH
    assertions around each call still carry the behavioural check.
    """
    if sqlite3.sqlite_version_info < (3, 42, 0):
        return
    with db.transaction() as conn:
        conn.execute("INSERT INTO messages_fts(messages_fts, rank) VALUES ('integrity-check', 0)")


def _seed_v45(db_path: Path, bodies: list[str], tombstone: set[int] = frozenset()):
    """Build a genuine v45 DB with one conversation and ``bodies`` messages.

    Returns ``(message_ids, rowids)`` in insertion order. Indices named in
    ``tombstone`` are soft-deleted (their content stays in ``messages`` but
    must never re-enter the index).
    """
    with chachanotes_db_at_version(db_path, 45, client_id="v47-seed") as historical:
        conversation_id = historical.add_conversation(
            {"title": "backfill", "character_id": 1}
        )
        message_ids = [
            historical.add_message(
                {
                    "conversation_id": conversation_id,
                    "sender": "user",
                    "content": body,
                }
            )
            for body in bodies
        ]
        for index in tombstone:
            historical.soft_delete_message(message_ids[index], expected_version=1)
        rowids = [
            historical.execute_query(
                "SELECT rowid FROM messages WHERE id = ?", (message_id,)
            ).fetchone()[0]
            for message_id in message_ids
        ]
    return message_ids, rowids


@pytest.fixture
def db(tmp_path: Path):
    instance = CharactersRAGDB(tmp_path / "chachanotes.db", client_id="v47-test")
    yield instance
    instance.close_connection()


def test_schema_version_is_47(db):
    """The one exact current-version pin (see this module's docstring)."""
    assert _version(db.get_connection()) == 47
    assert CharactersRAGDB._CURRENT_SCHEMA_VERSION == 47


def test_fresh_schema_guards_the_messages_fts_delete_halves_on_membership(db):
    triggers = {
        row[0]: row[1]
        for row in db.execute_query(
            "SELECT name, sql FROM sqlite_master WHERE type = 'trigger' "
            "AND name IN ('messages_au', 'messages_ad')"
        ).fetchall()
    }
    assert set(triggers) == {"messages_au", "messages_ad"}
    for name, sql in triggers.items():
        normalized = " ".join(sql.lower().split())
        assert "messages_fts_docsize" in normalized, f"{name} lacks the membership guard"
    # The v46 leak/corruption guards survive alongside the new condition.
    au = " ".join(triggers["messages_au"].lower().split())
    assert "old.deleted = 0" in au
    assert "new.deleted = 0" in au


def test_migrate_from_v46_to_v47_requires_version_46(db):
    """A fresh DB is already at 47, so re-entering the step must refuse."""
    with pytest.raises(SchemaError, match="requires schema version"):
        db._migrate_from_v46_to_v47(db.get_connection())


def test_upgrade_from_v45_defers_the_messages_fts_reinsert(tmp_path: Path):
    """The wall itself: construction must clear the index, not rewrite it.

    On the pre-task-21100 code this fails because construction reinserts
    every live message inside the version-bump transaction (docsize is full
    immediately). Afterwards, construction leaves an empty-but-consistent
    index, and the background driver -- not the migration -- restores exactly
    the live rows, tombstones excluded.
    """
    db_path = tmp_path / "chachanotes.db"
    bodies = [f"deferneedle{i:03d} body" for i in range(9)]
    _, rowids = _seed_v45(db_path, bodies, tombstone={4})

    migrated = CharactersRAGDB(db_path, client_id="v47-upgrade")
    try:
        assert _version(migrated.get_connection()) == 47
        # The version bump landed WITHOUT the O(total chat text) reinsert.
        assert _docsize_rowids(migrated) == set()
        # Window semantics: search is empty-but-consistent, never an error.
        assert _fts_rowids(migrated, "deferneedle004") == []
        _assert_index_structurally_sound(migrated)

        indexed = backfill_chachanotes_messages_fts(migrated, chunk_size=3)
        assert indexed == 8  # 9 seeded minus the tombstone

        assert _docsize_rowids(migrated) == _live_rowids(migrated)
        assert _fts_rowids(migrated, "deferneedle000") == [rowids[0]]
        # The tombstoned message's content must NOT come back (v46 privacy).
        assert _fts_rowids(migrated, "deferneedle004") == []
        _assert_index_structurally_sound(migrated)
    finally:
        migrated.close_connection()


def test_backfill_resumes_after_interruption_and_matches_the_one_shot_state(
    tmp_path: Path,
):
    """AC #1's resumability half, stop-after-N-chunks form.

    The invariant under test: "not yet indexed" is ``messages_fts_docsize``
    membership -- state in the DATABASE, not in any caller -- so abandoning
    the loop between chunks and reopening the file continues instead of
    restarting or double-indexing, and the end state equals what the one-shot
    v46 rebuild produced: exactly the live rows.
    """
    db_path = tmp_path / "chachanotes.db"
    bodies = [f"resumeneedle{i:03d} body" for i in range(20)]
    _, rowids = _seed_v45(db_path, bodies, tombstone={3, 17})

    first = CharactersRAGDB(db_path, client_id="v47-first-run")
    try:
        indexed_a, cursor = first.backfill_messages_fts(chunk_size=4)
        indexed_b, cursor = first.backfill_messages_fts(chunk_size=4, after_rowid=cursor)
        assert (indexed_a, indexed_b) == (4, 4)
        partial = _docsize_rowids(first)
        assert len(partial) == 8
    finally:
        # Abandon the run between chunks: nothing is finalized, no marker to
        # clean up -- the committed chunks ARE the resume state.
        first.close_connection()

    resumed = CharactersRAGDB(db_path, client_id="v47-resumed")
    try:
        assert backfill_chachanotes_messages_fts(resumed, chunk_size=4) == 10
        assert _docsize_rowids(resumed) == _live_rowids(resumed)
        assert partial <= _docsize_rowids(resumed)
        assert _fts_rowids(resumed, "resumeneedle019") == [rowids[19]]
        assert _fts_rowids(resumed, "resumeneedle003") == []
        assert _fts_rowids(resumed, "resumeneedle017") == []
        _assert_index_structurally_sound(resumed)
        # Idempotent: a run after completion performs no writes.
        assert backfill_chachanotes_messages_fts(resumed) == 0
    finally:
        resumed.close_connection()


def test_backfill_survives_sigkill_mid_run(tmp_path: Path):
    """AC #1's resumability half, real-kill form.

    A subprocess drives the backfill one small chunk at a time and is
    SIGKILLed once at least one chunk has committed. Reopening the file must
    recover (WAL), resume, and converge on the same end state as an
    uninterrupted run -- no brick, no double-indexing, no lost version bump.
    """
    db_path = tmp_path / "chachanotes.db"
    bodies = [f"killneedle{i:03d} body" for i in range(40)]
    _seed_v45(db_path, bodies)

    # The upgrade itself (fast now) happens in THIS process...
    CharactersRAGDB(db_path, client_id="v47-upgrade").close_connection()

    # ...and the kill lands mid-backfill in a child that crawls chunk by chunk.
    child_code = f"""
import sys, time
sys.path.insert(0, {str(Path(__file__).resolve().parents[2])!r})
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
db = CharactersRAGDB({str(db_path)!r}, client_id="v47-killed")
cursor = 0
while True:
    indexed, cursor = db.backfill_messages_fts(chunk_size=2, after_rowid=cursor)
    if indexed == 0:
        break
    print("chunk", flush=True)
    time.sleep(0.15)
"""
    child = subprocess.Popen(
        [sys.executable, "-c", child_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 30.0
        committed = 0
        while time.monotonic() < deadline:
            probe = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            try:
                committed = probe.execute(
                    "SELECT COUNT(*) FROM messages_fts_docsize"
                ).fetchone()[0]
            finally:
                probe.close()
            if committed >= 4:
                break
            if child.poll() is not None:
                break
            time.sleep(0.02)
        assert child.poll() is None, (
            "child finished before it could be killed; enlarge the seed "
            f"(stdout={child.stdout.read()!r} stderr={child.stderr.read()!r})"
        )
        assert committed >= 4, "no chunk committed before the deadline"
        os.kill(child.pid, signal.SIGKILL)
        child.wait(timeout=30)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=30)

    resumed = CharactersRAGDB(db_path, client_id="v47-after-kill")
    try:
        assert _version(resumed.get_connection()) == 47
        remaining = backfill_chachanotes_messages_fts(resumed, chunk_size=7)
        assert 0 < remaining <= 40 - 4
        assert _docsize_rowids(resumed) == _live_rowids(resumed)
        assert len(_fts_rowids(resumed, "killneedle039")) == 1
        _assert_index_structurally_sound(resumed)
    finally:
        resumed.close_connection()


def test_writes_during_the_backfill_window_do_not_corrupt_the_index(
    tmp_path: Path,
):
    """The reason v47 exists: the window must be write-safe.

    Every mutation class a live app can throw at a not-yet-backfilled row --
    content edit, soft delete, hard delete -- must neither error nor corrupt,
    and the backfill must then converge without double-indexing the rows the
    triggers indexed early.
    """
    db_path = tmp_path / "chachanotes.db"
    bodies = [f"windowneedle{i:03d} body" for i in range(6)]
    message_ids, rowids = _seed_v45(db_path, bodies)

    migrated = CharactersRAGDB(db_path, client_id="v47-window")
    try:
        assert _docsize_rowids(migrated) == set()

        # Edit an un-backfilled live row: the guarded delete half skips, the
        # insert half indexes the new content immediately (indexed "early").
        with migrated.transaction() as conn:
            conn.execute(
                "UPDATE messages SET content = 'editedneedle body', "
                "version = version + 1 WHERE id = ?",
                (message_ids[1],),
            )
        assert _fts_rowids(migrated, "editedneedle") == [rowids[1]]
        assert _fts_rowids(migrated, "windowneedle001") == []
        _assert_index_structurally_sound(migrated)

        # Soft-delete an un-backfilled row: nothing to remove, and the
        # backfill must skip it afterwards.
        migrated.soft_delete_message(message_ids[2], expected_version=1)
        _assert_index_structurally_sound(migrated)

        # Hard-delete an un-backfilled row: unguarded, messages_ad SILENTLY
        # poisons the doclists here (and can raise "database disk image is
        # malformed" depending on index state -- against this partly-filled
        # index it is the silent form: no raise, integrity-check(0) stays
        # green, and `messages_fts_data` grows a dangling delete-marker
        # block). MATCH assertions cannot see that, so the witness measures
        # the index's physical storage directly: with the membership guard
        # the delete is a no-op and `messages_fts_data` must not move.
        fts_data_before = _fts_data_footprint(migrated)
        with migrated.transaction() as conn:
            conn.execute("DELETE FROM messages WHERE id = ?", (message_ids[3],))
        assert _fts_data_footprint(migrated) == fts_data_before, (
            "hard-deleting an unindexed row wrote into messages_fts_data -- "
            "the messages_ad membership guard is not holding"
        )
        _assert_index_structurally_sound(migrated)

        # Converge. 6 seeded - 1 soft-deleted - 1 hard-deleted - 1 indexed
        # early by the edit = 3 rows left for the backfill.
        assert backfill_chachanotes_messages_fts(migrated, chunk_size=2) == 3
        assert _docsize_rowids(migrated) == _live_rowids(migrated)
        assert _fts_rowids(migrated, "editedneedle") == [rowids[1]]
        assert _fts_rowids(migrated, "windowneedle002") == []
        assert _fts_rowids(migrated, "windowneedle003") == []
        _assert_index_structurally_sound(migrated)

        # And a hard delete of a BACKFILLED row still leaves the index.
        with migrated.transaction() as conn:
            conn.execute("DELETE FROM messages WHERE id = ?", (message_ids[4],))
        assert _fts_rowids(migrated, "windowneedle004") == []
        _assert_index_structurally_sound(migrated)

        # Hard-delete a TOMBSTONED row (soft-deleted above, so never indexed):
        # the pre-existing latent messages_ad bug. Same physical-storage
        # witness -- unguarded, this either raises or silently appends
        # dangling delete-markers depending on index state; guarded, it must
        # not touch messages_fts_data at all.
        fts_data_before = _fts_data_footprint(migrated)
        with migrated.transaction() as conn:
            conn.execute("DELETE FROM messages WHERE id = ?", (message_ids[2],))
        assert _fts_data_footprint(migrated) == fts_data_before, (
            "hard-deleting a tombstoned row wrote into messages_fts_data -- "
            "the messages_ad membership guard is not holding"
        )
        _assert_index_structurally_sound(migrated)
    finally:
        migrated.close_connection()


def test_the_v46_shaped_trigger_would_corrupt_during_the_window(tmp_path: Path):
    """Keep the raising reproduction that makes v47 mandatory, not defensive.

    With the v46 trigger shape (``WHERE old.deleted = 0`` alone), editing a
    live row that is not in the external-content index corrupts it. Whether
    SQLite RAISES is index-state-dependent: against an empty index (this
    minimal table, and a freshly cleared real one) the UPDATE itself raises
    ``database disk image is malformed``; against a partly-filled index the
    same 'delete' silently poisons the doclists with no error and a green
    integrity-check -- the ``_fts_data_footprint`` witnesses in
    ``test_writes_during_the_backfill_window_do_not_corrupt_the_index`` catch
    that form. If this test ever starts failing because SQLite stops
    objecting, the guard is still required for correctness; re-verify the
    incident evidence.
    """
    probe = sqlite3.connect(tmp_path / "shape.db")
    try:
        probe.executescript(
            """
            CREATE TABLE m(id INTEGER PRIMARY KEY, content TEXT,
                           deleted INT NOT NULL DEFAULT 0);
            CREATE VIRTUAL TABLE m_fts USING fts5(
                content, content='m', content_rowid='id');
            CREATE TRIGGER m_au AFTER UPDATE ON m BEGIN
              INSERT INTO m_fts(m_fts, rowid, content)
              SELECT 'delete', old.id, old.content WHERE old.deleted = 0;
              INSERT INTO m_fts(rowid, content)
              SELECT new.id, new.content WHERE new.deleted = 0;
            END;
            -- Un-backfilled live row: in the table, absent from the index.
            INSERT INTO m(content) VALUES ('hello world');
            """
        )
        with pytest.raises(sqlite3.DatabaseError, match="malformed"):
            probe.execute("UPDATE m SET content = 'hello mars' WHERE id = 1")
    finally:
        probe.close()


def test_v47_leaves_a_complete_index_alone(tmp_path: Path):
    """Databases stamped 46 with a complete index must be unaffected.

    A database whose messages were all indexed at stamp 46 (the population
    the ORIGINAL full-rebuild v46 produced) takes the v47 trigger swap and
    nothing else: no delete-all, no backfill work, search uninterrupted.
    """
    db_path = tmp_path / "chachanotes.db"
    with chachanotes_db_at_version(db_path, 46, client_id="v46-complete") as at46:
        conversation_id = at46.add_conversation(
            {"title": "complete", "character_id": 1}
        )
        for i in range(5):
            at46.add_message(
                {
                    "conversation_id": conversation_id,
                    "sender": "user",
                    "content": f"completeneedle{i:03d} body",
                }
            )
        complete = _docsize_rowids(at46)
        assert len(complete) == 5  # indexed by messages_ai as they landed

    migrated = CharactersRAGDB(db_path, client_id="v47-complete")
    try:
        assert _version(migrated.get_connection()) == 47
        assert _docsize_rowids(migrated) == complete
        assert len(_fts_rowids(migrated, "completeneedle002")) == 1
        # Nothing pending: the driver's first chunk finds nothing.
        assert backfill_chachanotes_messages_fts(migrated) == 0
        assert _docsize_rowids(migrated) == complete
    finally:
        migrated.close_connection()


def test_a_failure_mid_v47_rewinds_the_whole_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Atomic and re-enterable, the task-19553 rule (v46's test, one step on).

    Poisoning the v47 script must rewind EVERYTHING -- including the v46
    step's `delete-all`, so a failed upgrade leaves the old index serving
    search rather than an empty one under an un-bumped stamp.
    """
    db_path = tmp_path / "poisoned.db"
    _, rowids = _seed_v45(db_path, ["poisonneedle body"])

    original = CharactersRAGDB._execute_migration_statements

    def poisoned(self, cursor, script, label):
        if label == "V46→V47":
            script = script + "\nINSERT INTO no_such_table_21100(x) VALUES (1);\n"
        return original(self, cursor, script, label)

    monkeypatch.setattr(CharactersRAGDB, "_execute_migration_statements", poisoned)
    with pytest.raises(SchemaError, match="no_such_table_21100"):
        CharactersRAGDB(db_path, client_id="poisoned")

    connection = sqlite3.connect(str(db_path))
    try:
        assert _version(connection) == 45, "a failing chain must not bump the stamp"
        assert [
            row[0]
            for row in connection.execute(
                "SELECT rowid FROM messages_fts WHERE messages_fts MATCH 'poisonneedle'"
            )
        ] == rowids, "the v46 delete-all must rewind with the failing chain"
    finally:
        connection.close()

    # Re-enterable: with the poison removed the same file migrates fully.
    monkeypatch.undo()
    migrated = CharactersRAGDB(db_path, client_id="poison-removed")
    try:
        assert _version(migrated.get_connection()) == 47
        assert _docsize_rowids(migrated) == set()
        assert backfill_chachanotes_messages_fts(migrated) == 1
        assert _fts_rowids(migrated, "poisonneedle") == rowids
    finally:
        migrated.close_connection()


def test_backfill_chunk_size_must_be_positive(db):
    """A non-positive LIMIT would report completion over a real backlog."""
    with pytest.raises(ValueError, match="chunk_size"):
        db.backfill_messages_fts(chunk_size=0)


# ---------------------------------------------------------------------------
# review fix round: the backfill must not kill concurrent hot writers
# ---------------------------------------------------------------------------
#: Every writer on the `messages` table that runs a read-then-write
#: transaction on a user-facing chat path. A DEFERRED begin that reads before
#: writing can hit SQLite's non-retryable snapshot-upgrade SQLITE_BUSY when a
#: concurrent writer (the per-chunk backfill commits during the whole
#: first-boot window) commits inside the read->write gap -- the writer dies
#: with `database is locked` INSTANTLY, bypassing the 15 s busy timeout
#: (the exact failure TransactionContextManager's own comment documents,
#: with `immediate=True` as its documented cure). These must all reserve the
#: write lock up front. `update_message_feedback` is covered by delegation to
#: `update_message`; the blind single-statement writers
#: (`update_message_usage_local`, `update_message_metadata_local`,
#: `append_message_exchanges_local`) are deliberately NOT here -- with no
#: read before their write there is no snapshot to upgrade, and plain
#: SQLITE_BUSY honors the busy timeout.
HOT_MESSAGE_WRITERS = (
    "add_message",
    "create_assistant_with_continuation",
    "append_message_attachment_with_metadata",
    "swap_message_attachment_with_scalar",
    "update_message",
    "soft_delete_message",
    "soft_delete_message_subtree",
    "create_message_variant",
    "select_message_variant",
)


def test_hot_writer_survives_a_backfill_commit_inside_its_transaction(
    tmp_path: Path,
):
    """The reviewer's deterministic interleave, through the production writer.

    Sequence: `add_message` enters its transaction and takes its read
    snapshot (the conversation SELECT); one backfill chunk then commits from
    another connection; the writer performs its INSERT. On the DEFERRED
    shape this raises `database is locked` instantly (snapshot-upgrade
    SQLITE_BUSY bypasses the busy timeout entirely -- throttling the
    backfill cannot help, ONE commit inside the gap is fatal). With
    `immediate=True` the writer holds the write lock before its first read,
    so the backfill chunk queues on the busy timeout instead and the user's
    message lands.
    """
    db_path = tmp_path / "chachanotes.db"
    _seed_v45(db_path, [f"interleaveneedle{i:03d} body" for i in range(8)])

    writer = CharactersRAGDB(db_path, client_id="v47-writer")
    backfiller = CharactersRAGDB(db_path, client_id="v47-backfiller")
    try:
        assert _docsize_rowids(writer) == set()  # the window is open
        conversation_id = writer.add_conversation(
            {"title": "hot", "character_id": 1}
        )

        outcome: dict[str, str] = {}
        original_execute_query = CharactersRAGDB.execute_query

        def interleaved(self, query, params=None, **kwargs):
            if (
                self is writer
                and "INSERT INTO messages" in query
                and "chunk" not in outcome
            ):
                # The writer is between its read snapshot and its first
                # write: commit one real backfill chunk from a second
                # connection right here. Under the fixed (IMMEDIATE) shape
                # the writer already holds the write lock, so the chunk
                # must time out and queue for later instead of committing.
                backfiller.get_connection().execute("PRAGMA busy_timeout = 200")
                try:
                    indexed, _ = backfiller.backfill_messages_fts(chunk_size=1)
                    outcome["chunk"] = f"committed:{indexed}"
                except (sqlite3.OperationalError, CharactersRAGDBError) as exc:
                    outcome["chunk"] = f"blocked:{exc}"
            return original_execute_query(self, query, params, **kwargs)

        with patch.object(CharactersRAGDB, "execute_query", interleaved):
            message_id = writer.add_message(
                {
                    "conversation_id": conversation_id,
                    "sender": "user",
                    "content": "hotneedle first message after the upgrade",
                }
            )

        assert message_id, "the user's first message after the upgrade must land"
        assert "chunk" in outcome, "the interleave never fired -- test is vacuous"
        # The writer held the write lock, so the chunk queued instead of
        # committing into the gap; nothing is lost -- it lands right after.
        assert outcome["chunk"].startswith("blocked"), outcome["chunk"]
        assert _fts_rowids(writer, "hotneedle") != []
        assert backfill_chachanotes_messages_fts(backfiller) == 8
        assert _docsize_rowids(writer) == _live_rowids(writer)
    finally:
        writer.close_connection()
        backfiller.close_connection()


def test_hot_message_writers_reserve_the_write_lock_up_front():
    """Structural backstop for the interleave test above.

    The behavioural witness exercises `add_message`; this pins the same
    IMMEDIATE shape on every enumerated hot writer so a new or reverted
    DEFERRED read-then-write on the chat path fails here by name. Each of
    these methods has exactly one `self.transaction(` call site (verified by
    AST when the list was built), so the two assertions are precise.
    """
    for name in HOT_MESSAGE_WRITERS:
        source = inspect.getsource(getattr(CharactersRAGDB, name))
        assert "self.transaction(immediate=True)" in source, (
            f"{name} must reserve the write lock up front (see "
            "HOT_MESSAGE_WRITERS)"
        )
        assert "self.transaction()" not in source, (
            f"{name} still opens a DEFERRED transaction"
        )
