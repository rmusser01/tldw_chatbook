"""PERF-04 (TASK-33263): the semantic mutation guard without a trace callback.

``register_semantic_mutation_guard`` used ``set_trace_callback`` only to spot
BEGIN/COMMIT/ROLLBACK. On Python 3.12 the trace callback receives the
*expanded* SQL, so SQLite rendered every bound parameter -- BLOBs as hex --
for each statement and each trigger step: a 3 MiB image message took ~1.2 s
to insert instead of ~9 ms (2026-09-27 audit). The guard still has to fail
closed when an authorized scope's transaction ends and a new one begins,
which the preservation tests below pin.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tldw_chatbook.DB import base_db
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def _seed_traced_message(db: CharactersRAGDB) -> str:
    """Create a message with a live semantic revision, so its mutations are guarded."""
    conversation_id = db.add_conversation({"title": "guarded"})
    assert conversation_id is not None
    message_id = db.add_message(
        {"conversation_id": conversation_id, "sender": "user", "content": "body"}
    )
    assert message_id is not None
    db.set_message_attachments(
        message_id,
        [
            {
                "position": 1,
                "data": b"attachment",
                "mime_type": "image/png",
                "display_name": "a.png",
            }
        ],
    )
    return message_id


def test_guard_does_not_trace_statements_on_managed_connections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Managed ChaChaNotes connections never route statements through a trace callback.

    Args:
        tmp_path: pytest fixture; holds this test's database file.
        monkeypatch: pytest fixture; wraps ``trace_transaction`` to count calls.
    """
    calls: list[str] = []
    original = base_db._SemanticMutationAuthorization.trace_transaction

    def counting(self: object, statement: str) -> None:
        calls.append(statement[:40])
        original(self, statement)  # type: ignore[arg-type]

    monkeypatch.setattr(
        base_db._SemanticMutationAuthorization, "trace_transaction", counting
    )
    db = CharactersRAGDB(tmp_path / "no-trace.sqlite", "no-trace")
    try:
        conversation_id = db.add_conversation({"title": "image"})
        calls.clear()
        db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "content": "with image",
                "image_data": b"\x89PNG" + b"\x00" * (256 * 1024),
                "image_mime_type": "image/png",
            }
        )
        assert calls == []
    finally:
        db.close_connection()


def test_cached_commit_and_begin_inside_a_scope_fail_closed(tmp_path: Path) -> None:
    """A scope whose transaction was swapped via cached statements cannot mutate.

    A statement served from Python's sqlite3 statement cache is not
    re-prepared, so the authorizer's COMMIT/ROLLBACK denial never sees it.
    The generation check is what refuses the guarded write afterwards.

    Args:
        tmp_path: pytest fixture; holds this test's database file.
    """
    db = CharactersRAGDB(tmp_path / "cached-boundary.sqlite", "cached-boundary")
    try:
        message_id = _seed_traced_message(db)
        conn = db.get_connection()
        authorization = db._semantic_mutation_authorization_for_coordinator(conn)
        # Warm the statement cache with the exact boundary statements.
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("COMMIT")
        conn.execute("BEGIN IMMEDIATE")
        try:
            with authorization._authorize(
                message_id=message_id, operations={"message_update"}
            ):
                conn.execute("COMMIT")
                conn.execute("BEGIN IMMEDIATE")
                with pytest.raises(sqlite3.DatabaseError, match="semantic mutation"):
                    conn.execute(
                        "UPDATE messages SET content = 'escaped' WHERE id = ?",
                        (message_id,),
                    )
                with pytest.raises(
                    RuntimeError, match="semantic_mutation_transaction_changed"
                ):
                    authorization._assert_current_transaction()
        finally:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
        row = conn.execute(
            "SELECT content FROM messages WHERE id = ?", (message_id,)
        ).fetchone()
        assert row[0] == "body"
    finally:
        db.close_connection()


def test_authorized_update_in_the_same_transaction_still_succeeds(
    tmp_path: Path,
) -> None:
    """The guard keeps allowing the authorized mutation in its own transaction.

    Args:
        tmp_path: pytest fixture; holds this test's database file.
    """
    db = CharactersRAGDB(tmp_path / "same-transaction.sqlite", "same-transaction")
    try:
        message_id = _seed_traced_message(db)
        conn = db.get_connection()
        authorization = db._semantic_mutation_authorization_for_coordinator(conn)
        conn.execute("BEGIN IMMEDIATE")
        try:
            with authorization._authorize(
                message_id=message_id, operations={"message_update"}
            ):
                conn.execute(
                    "UPDATE messages SET content = 'authorized' WHERE id = ?",
                    (message_id,),
                )
                authorization._assert_current_transaction()
            conn.execute("COMMIT")
        finally:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
        row = conn.execute(
            "SELECT content FROM messages WHERE id = ?", (message_id,)
        ).fetchone()
        assert row[0] == "authorized"
    finally:
        db.close_connection()


def test_commit_outside_python_method_is_seen_before_the_next_statement(
    tmp_path: Path,
) -> None:
    """A transaction ended by C-level commit is noticed at the next statement.

    Args:
        tmp_path: pytest fixture; holds this test's database file.
    """
    db = CharactersRAGDB(tmp_path / "c-level-commit.sqlite", "c-level-commit")
    try:
        message_id = _seed_traced_message(db)
        conn = db.get_connection()
        authorization = db._semantic_mutation_authorization_for_coordinator(conn)
        conn.execute("BEGIN IMMEDIATE")
        generation = authorization._transaction_generation
        # sqlite3.Connection.__exit__ commits in C, bypassing any Python-level
        # commit() override.
        sqlite3.Connection.__exit__(conn, None, None, None)
        assert not conn.in_transaction
        conn.execute("BEGIN IMMEDIATE")
        try:
            assert authorization._transaction_generation != generation
        finally:
            conn.execute("ROLLBACK")
        assert message_id
    finally:
        db.close_connection()


class _ObservedCursor(sqlite3.Cursor):
    """A caller's own cursor type, as Tests/Workflows passes to observe statements."""


class _DirectCursor(sqlite3.Cursor):
    """Overrides every statement entry point without calling ``super()``."""

    def execute(self, sql, parameters=()):  # noqa: D102
        return sqlite3.Cursor.execute(self, sql, parameters)

    def executemany(self, sql, seq_of_parameters):  # noqa: D102
        return sqlite3.Cursor.executemany(self, sql, seq_of_parameters)

    def executescript(self, sql_script):  # noqa: D102
        return sqlite3.Cursor.executescript(self, sql_script)


@pytest.mark.parametrize("factory", [sqlite3.Cursor, _ObservedCursor, _DirectCursor])
def test_a_cursor_from_any_factory_still_reports_transaction_boundaries(
    tmp_path: Path, factory: type[sqlite3.Cursor]
) -> None:
    """A C-level commit then a BEGIN through a caller-chosen cursor is a new transaction.

    Before the fix, ``cursor(factory)`` returned an untracked cursor: after
    ``with connection:`` committed in C, a BEGIN through it left
    ``in_transaction`` True at every tracked observation, so an authorization
    from the first transaction still held in the second (Qodo, #2894).

    Args:
        tmp_path: pytest fixture; holds this test's database file.
        factory: The cursor type the caller asks for.
    """
    db = CharactersRAGDB(tmp_path / "cursor-factory.sqlite", "cursor-factory")
    try:
        _seed_traced_message(db)
        conn = db.get_connection()
        authorization = db._semantic_mutation_authorization_for_coordinator(conn)
        conn.execute("BEGIN IMMEDIATE")
        generation = authorization._transaction_generation
        sqlite3.Connection.__exit__(conn, None, None, None)
        cursor = conn.cursor(factory)
        assert isinstance(cursor, factory)
        cursor.execute("BEGIN IMMEDIATE")
        try:
            assert authorization._transaction_generation != generation
        finally:
            conn.execute("ROLLBACK")
    finally:
        db.close_connection()


def test_a_non_class_cursor_factory_is_refused() -> None:
    """A factory that cannot be made tracked fails closed rather than untracked."""
    conn = sqlite3.connect(":memory:", factory=base_db._QuiescentSQLiteConnection)
    try:
        with pytest.raises(TypeError):
            conn.cursor(lambda connection: sqlite3.Cursor(connection))
    finally:
        conn.close()


def test_a_script_that_ends_and_restarts_a_transaction_reports_a_boundary() -> None:
    """``in_transaction`` is True before and after, yet the transaction changed."""
    conn = sqlite3.connect(":memory:", factory=base_db._QuiescentSQLiteConnection)
    try:
        conn.execute("CREATE TABLE t(x)")
        conn.execute("BEGIN")
        boundaries: list[bool] = []
        conn.set_transaction_boundary_listener(lambda: boundaries.append(True))
        # executescript commits the pending transaction first; the script
        # then opens a new one.
        conn.executescript("BEGIN; INSERT INTO t VALUES (1);")
        assert conn.in_transaction
        assert boundaries, "a COMMIT+BEGIN inside one script went unreported"
    finally:
        conn.close()


def test_a_script_boundary_advances_the_managed_guard_generation(
    tmp_path: Path,
) -> None:
    """The same script on a managed connection starts a new guard generation.

    Args:
        tmp_path: pytest fixture; holds this test's database file.
    """
    db = CharactersRAGDB(tmp_path / "script-boundary.sqlite", "script-boundary")
    try:
        _seed_traced_message(db)
        conn = db.get_connection()
        authorization = db._semantic_mutation_authorization_for_coordinator(conn)
        conn.execute("BEGIN IMMEDIATE")
        generation = authorization._transaction_generation
        conn.executescript("BEGIN IMMEDIATE; SELECT 1;")
        try:
            assert conn.in_transaction
            assert authorization._transaction_generation != generation
        finally:
            conn.execute("ROLLBACK")
    finally:
        db.close_connection()


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("execute", ("SELECT 1",)),
        ("executemany", ("SELECT ?", [(1,)])),
        ("executescript", ("SELECT 1;",)),
    ],
)
def test_a_cursor_used_after_close_does_not_pin_the_quiescence_registry(
    method: str, args: tuple[object, ...]
) -> None:
    """A stale cursor's failed call releases its use token, so maintenance can drain.

    Args:
        method: The cursor method called after its connection closed.
        args: Arguments for that call.
    """
    conn = sqlite3.connect(":memory:", factory=base_db._QuiescentSQLiteConnection)
    registry = base_db.SQLiteConnectionQuiescenceRegistry()
    conn.attach_quiescence_registry(registry)
    conn.set_transaction_boundary_listener(lambda: None)
    cursor = conn.cursor()
    conn.close()

    with pytest.raises(sqlite3.ProgrammingError):
        getattr(cursor, method)(*args)

    token = registry.begin_quiescence(timeout_seconds=0.2)
    registry.end_quiescence(token)
