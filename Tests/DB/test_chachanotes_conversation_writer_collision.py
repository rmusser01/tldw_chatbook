"""Conversation-path writers under a concurrent backfill chunk commit (task-22501).

TASK-21100 moved the ChaChaNotes hot *message* writers to
``transaction(immediate=True)`` after the first-boot ``messages_fts`` backfill
was shown to kill them with an INSTANT ``database is locked`` (the
snapshot-upgrade SQLITE_BUSY that bypasses the busy handler entirely), and its
review carved out "blind single-statement writers" as safe -- no read before
the write, so no snapshot to upgrade. TASK-22200's adversarial reviewer then
reported the carve-out's most prominent member, ``add_conversation``, as dying
anyway (3/3) while a paced backfill chunk commits.

Mechanism note (verified for this task on real SQLite 3.49 with two
connections and a committing thread): the fatal shape is *a read and then a
write inside ONE DEFERRED transaction*. A blind DEFERRED INSERT (depth-0
``add_conversation``'s shape) always survives -- its only contention is plain
SQLITE_BUSY, which the busy timeout retries; its in-statement FK parent
lookups happen after the write lock is taken. What makes the conversation
CREATE path genuinely exposed is composition: an outer DEFERRED wrapper whose
first statement is a READ (``add_message``'s conversation-existence SELECT in
the fork/copier and import wrappers) silently neutralizes every inner
writer's IMMEDIATE (the manager honours ``immediate`` only at depth 0), and
any backfill chunk committing inside that read->write gap kills the whole
user-facing unit, un-retried. That is the exact nested-composition class
TASK-21100's review fixed for ``ChatPersistenceService`` and missed on these
two sites.

The two probes below follow the repo's established idioms
(``test_chachanotes_v47_messages_fts_backfill.py``'s interleave, and
``test_chachanotes_fts_backfill_pacing.py``'s load probe): a real temp-file
WAL database, TWO ``CharactersRAGDB`` instances (writer + backfiller), the
real paced backfill driver, and the real conversation-path writers, with
in-flight assertions so neither probe can pass vacuously.
"""

from __future__ import annotations

import inspect
import sqlite3
import threading
import time
from pathlib import Path

from tldw_chatbook.DB.chachanotes_fts_backfill import (
    backfill_chachanotes_messages_fts,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError


def _open_with_backfill_window(
    db_path: Path, count: int, *, client_id: str
) -> CharactersRAGDB:
    """A current-schema DB holding ``count`` live messages, index cleared.

    Seeds in one outer transaction, then reproduces the post-v46-upgrade
    state with the migration's own reset (``'delete-all'``) -- the exact
    window the backfill driver exists to close, opened the same way
    ``test_chachanotes_fts_backfill_pacing.py`` opens it.
    """
    db = CharactersRAGDB(db_path, client_id=client_id)
    with db.transaction(immediate=True):
        conversation_id = db.add_conversation(
            {"title": "probe seed", "character_id": 1}
        )
        for i in range(count):
            db.add_message(
                {
                    "conversation_id": conversation_id,
                    "sender": "user",
                    "content": f"probeneedle{i:04d} body",
                }
            )
    with db.transaction(immediate=True) as conn:
        conn.execute("INSERT INTO messages_fts(messages_fts) VALUES ('delete-all')")
    assert _docsize_count(db) == 0  # the window is open
    return db


def _docsize_count(db: CharactersRAGDB) -> int:
    return db.execute_query(
        "SELECT COUNT(*) FROM messages_fts_docsize"
    ).fetchone()[0]


def test_add_conversation_survives_the_concurrent_chunk_commit_load_probe(
    tmp_path: Path,
):
    """AC #1's probe: the real ``add_conversation`` against an in-flight
    paced backfill, ten writes across the window, all of which must land.

    Production shape end to end: the writer and the backfiller are TWO
    ``CharactersRAGDB`` instances over one file, the real paced backfill
    driver on a worker thread (240 rows / chunk 8 / 0.05 s pause = 30 chunk
    commits over >= 1.5 s), and the real ``add_conversation`` (the
    user-facing conversation-create path) ten times from the foreground.
    ``add_message`` (IMMEDIATE) survived this same load 10/10 in
    TASK-22200's review; the DEFERRED begin could never be *proven* safe
    against future body changes, so the fix reserves the lock up front and
    this probe holds it to 10/10. The in-flight assertions keep the probe
    from passing vacuously against a finished backfill.
    """
    db_path = tmp_path / "chachanotes.db"
    db = _open_with_backfill_window(db_path, 240, client_id="t22501-probe")
    backfiller = CharactersRAGDB(db_path, client_id="t22501-backfill")
    try:
        failures: list[BaseException] = []

        def run_backfill() -> None:
            try:
                backfill_chachanotes_messages_fts(
                    backfiller, chunk_size=8, pause_seconds=0.05
                )
            except BaseException as exc:  # pragma: no cover - failure detail
                failures.append(exc)

        backfill_thread = threading.Thread(target=run_backfill, daemon=True)
        backfill_thread.start()

        # Wait for the run to be genuinely in flight (first chunk committed).
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and _docsize_count(db) == 0:
            time.sleep(0.005)
        assert _docsize_count(db) > 0, "backfill never started"

        errors: list[str] = []
        for i in range(10):
            try:
                conv_id = db.add_conversation(
                    {"title": f"probe {i}", "character_id": 1}
                )
                assert conv_id, "a foreground conversation create returned no id"
            except BaseException as exc:  # the exact unretried UI failure
                errors.append(f"write {i}: {type(exc).__name__}: {exc}")
            if i == 4:
                assert backfill_thread.is_alive(), (
                    "backfill finished before the writes -- probe was vacuous"
                )
            time.sleep(0.03)

        assert backfill_thread.is_alive(), (
            "backfill finished before the writes -- probe was vacuous; "
            f"errors so far: {errors}"
        )
        assert errors == [], errors

        backfill_thread.join(timeout=60.0)
        assert not backfill_thread.is_alive()
        assert failures == []
    finally:
        db.close_connection()
        backfiller.close_connection()


def test_conversation_fork_survives_a_chunk_commit_inside_the_read_to_write_gap(
    tmp_path: Path,
):
    """AC #3's regression: the original failure shape, deterministically.

    Drives the REAL user-facing fork path
    (``ChatConversationService.copy_conversation_active_path``: outer
    DEFERRED wrapper, first statement ``add_message``'s conversation SELECT,
    then its INSERT). A trace callback on the writer's connection commits
    one REAL backfill chunk from a second instance at the exact moment the
    wrapper's first INSERT is about to run -- i.e. after the read snapshot
    is pinned, before the write. On the DEFERRED shape the chunk commits
    into the gap and the writer dies with the INSTANT, busy-handler
    bypassing ``database is locked`` (red before the fix). With the outer
    wrapper IMMEDIATE the writer holds the write lock from its BEGIN, so
    the chunk queues on the (shortened) busy timeout instead -- the
    ``blocked`` outcome below is the proof the interleave fired against a
    lock-reserving writer -- and the user's fork lands whole.
    """
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService

    db_path = tmp_path / "chachanotes.db"
    writer = _open_with_backfill_window(db_path, 8, client_id="t22501-fork")
    backfiller = CharactersRAGDB(db_path, client_id="t22501-fork-b")
    try:
        assert _docsize_count(writer) == 0  # the window is open
        # A source conversation with a two-message chain, and a fork target.
        source_id = writer.add_conversation({"title": "fork source"})
        first_id = writer.add_message(
            {
                "conversation_id": source_id,
                "sender": "user",
                "content": "forkneedle first message",
            }
        )
        second_id = writer.add_message(
            {
                "conversation_id": source_id,
                "sender": "assistant",
                "content": "forkneedle second message",
                "parent_message_id": first_id,
            }
        )
        assert second_id
        target_id = writer.add_conversation({"title": "fork target"})
        # The two source messages were indexed by their INSERT triggers (the
        # seeded 8 remain unbackfilled): the window is still genuinely open.
        assert _docsize_count(writer) == 2

        # Bound the backfiller's lock wait so a blocked chunk fails fast
        # instead of serialising behind the writer's whole transaction.
        backfiller.get_connection().execute("PRAGMA busy_timeout = 200")

        outcome: dict[str, str] = {}

        def commit_a_chunk_at_the_insert(statement: str) -> None:
            if outcome or not statement.lstrip().upper().startswith(
                "INSERT INTO MESSAGES"
            ):
                return
            try:
                indexed, _ = backfiller.backfill_messages_fts(chunk_size=1)
                outcome["chunk"] = f"committed:{indexed}"
            except (
                sqlite3.OperationalError,
                CharactersRAGDBError,
            ) as exc:
                outcome["chunk"] = f"blocked:{exc}"

        service = ChatConversationService(writer)
        writer.get_connection().set_trace_callback(commit_a_chunk_at_the_insert)
        try:
            result = service.copy_conversation_active_path(
                source_id, target_id
            )
        finally:
            writer.get_connection().set_trace_callback(None)

        assert "chunk" in outcome, "the interleave never fired -- test is vacuous"
        # The writer held the write lock, so the chunk queued instead of
        # committing into the read->write gap.
        assert outcome["chunk"].startswith("blocked"), outcome["chunk"]
        assert result["copied"] == 2, result
        copied = writer.execute_query(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
            (target_id,),
        ).fetchone()[0]
        assert copied == 2
    finally:
        writer.close_connection()
        backfiller.close_connection()


def test_conversation_path_writers_reserve_the_write_lock_up_front():
    """Structural backstop for the two probes above (the v47
    ``HOT_MESSAGE_WRITERS`` idiom): a reverted DEFERRED begin on any fixed
    conversation-path writer fails here by name, so the race-based probe is
    not the only guard."""
    from tldw_chatbook.Chat.chat_conversation_service import (
        ChatConversationService,
    )

    add_source = inspect.getsource(CharactersRAGDB.add_conversation)
    assert "self.transaction(immediate=True)" in add_source, (
        "add_conversation must reserve the write lock up front (task-22501; "
        "see Tests/DB/test_chachanotes_v47_messages_fts_backfill.py's "
        "HOT_MESSAGE_WRITERS for the standing policy)"
    )
    assert "self.transaction()" not in add_source, (
        "add_conversation still opens a DEFERRED transaction"
    )

    fork_source = inspect.getsource(
        ChatConversationService.copy_conversation_active_path
    )
    assert "self.db.transaction(immediate=True)" in fork_source, (
        "copy_conversation_active_path's outer unit must reserve the write "
        "lock up front: its first statement is add_message's read"
    )
    assert "self.db.transaction()" not in fork_source, (
        "copy_conversation_active_path still opens a DEFERRED outer unit"
    )
