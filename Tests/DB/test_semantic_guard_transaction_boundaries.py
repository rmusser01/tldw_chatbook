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

import gc
import os
import sqlite3
import statistics
import time
from pathlib import Path

import pytest

from tldw_chatbook.DB import base_db
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

#: Every test here drives real SQLite connections (cubic, #2894).
pytestmark = pytest.mark.integration


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


class ObservedCursor(sqlite3.Cursor):
    """A caller's own cursor type, as Tests/Workflows passes to observe statements."""


class DirectCursor(sqlite3.Cursor):
    """Overrides every statement entry point without calling ``super()``."""

    def execute(self, sql: str, parameters: object = ()) -> sqlite3.Cursor:
        """Run one statement through the base cursor directly.

        Args:
            sql: Statement text.
            parameters: Bound parameters.

        Returns:
            This cursor.
        """
        return sqlite3.Cursor.execute(self, sql, parameters)

    def executemany(self, sql: str, seq_of_parameters: object) -> sqlite3.Cursor:
        """Run one statement per parameter set through the base cursor directly.

        Args:
            sql: Statement text.
            seq_of_parameters: Parameter sets.

        Returns:
            This cursor.
        """
        return sqlite3.Cursor.executemany(self, sql, seq_of_parameters)

    def executescript(self, sql_script: str) -> sqlite3.Cursor:
        """Run a script through the base cursor directly.

        Args:
            sql_script: Statements separated by semicolons.

        Returns:
            This cursor.
        """
        return sqlite3.Cursor.executescript(self, sql_script)


@pytest.mark.parametrize("factory", [sqlite3.Cursor, ObservedCursor, DirectCursor])
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


def test_tracked_cursor_type_puts_the_tracked_cursor_first() -> None:
    """The composition rule, without a database: tracked methods run first."""
    tracked = base_db._QuiescentSQLiteCursor
    compose = base_db._tracked_cursor_type

    assert compose(sqlite3.Cursor) is tracked
    assert compose(tracked) is tracked
    composed = compose(DirectCursor)
    assert composed.__mro__[1:3] == (tracked, DirectCursor)
    assert composed.execute is tracked.execute
    assert composed.executemany is tracked.executemany
    assert composed.executescript is tracked.executescript
    assert compose(DirectCursor) is composed  # cached, not a new class per call
    with pytest.raises(TypeError):
        compose(object)  # type: ignore[arg-type]

    class TrackedSubclass(tracked):
        """Keeps the tracked statement methods, so it passes through."""

    assert compose(TrackedSubclass) is TrackedSubclass


@pytest.mark.parametrize("method", ["execute", "executemany", "executescript"])
def test_a_tracked_subclass_replacing_a_statement_method_is_refused(method: str) -> None:
    """A tracked subclass that replaces a statement method is refused.

    The tracked cursor cannot be put ahead of its own subclass, so an override
    that could skip boundary observation fails closed (Qodo, #2894).

    Args:
        method: The statement method the subclass replaces.
    """
    replacement = getattr(sqlite3.Cursor, method)
    bypassing = type(
        "Bypassing", (base_db._QuiescentSQLiteCursor,), {method: replacement}
    )
    with pytest.raises(TypeError):
        base_db._tracked_cursor_type(bypassing)


def test_a_composed_cursor_still_runs_the_callers_finalizer() -> None:
    """The tracked cursor's destructor chains to the caller's (Qodo, #2894)."""
    finalized: list[bool] = []

    class FinalizingCursor(sqlite3.Cursor):
        """A caller cursor with its own cleanup."""

        def __del__(self) -> None:
            finalized.append(True)

    conn = sqlite3.connect(":memory:", factory=base_db._QuiescentSQLiteConnection)
    try:
        cursor = conn.cursor(FinalizingCursor)
        cursor.execute("SELECT 1").fetchall()
        del cursor
        gc.collect()
        assert finalized == [True]
    finally:
        conn.close()


class InitStatementCursor(sqlite3.Cursor):
    """Runs a statement from its own initializer."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        """Initialize, then run one statement.

        Args:
            connection: The connection the cursor belongs to.
        """
        super().__init__(connection)
        self.execute("SELECT 1").fetchall()


class DirectInitCursor(sqlite3.Cursor):
    """Initializes the base cursor directly rather than through ``super()``."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        """Initialize the base cursor directly.

        Args:
            connection: The connection the cursor belongs to.
        """
        sqlite3.Cursor.__init__(self, connection)


@pytest.mark.parametrize("factory", [InitStatementCursor, DirectInitCursor])
def test_a_composed_cursor_works_whatever_its_initializer_does(
    factory: type[sqlite3.Cursor],
) -> None:
    """The tracked state exists before the caller's initializer runs (Qodo, #2894).

    Args:
        factory: A caller cursor type with an unusual initializer.
    """
    conn = sqlite3.connect(":memory:", factory=base_db._QuiescentSQLiteConnection)
    try:
        cursor = conn.cursor(factory)
        assert cursor.execute("SELECT 2").fetchall() == [(2,)]
    finally:
        conn.close()


def test_a_savepoint_rollback_that_ran_before_a_failure_is_still_a_boundary(
    tmp_path: Path,
) -> None:
    """Only an authorizer refusal (SQLITE_AUTH) means the statement never ran.

    A caller cursor can execute ROLLBACK TO and then raise; that rollback
    happened, so it must still advance the generation (Qodo, #2894).

    Args:
        tmp_path: pytest fixture; holds this test's database file.
    """

    class RaisesAfterRunning(sqlite3.Cursor):
        """Runs the statement, then fails."""

        def execute(self, sql: str, parameters: object = ()) -> sqlite3.Cursor:
            """Run ``sql``, then raise for a ROLLBACK.

            Args:
                sql: Statement text.
                parameters: Bound parameters.

            Returns:
                This cursor, for any other statement.

            Raises:
                RuntimeError: After a ROLLBACK statement has run.
            """
            result = super().execute(sql, parameters)
            if sql.lstrip().upper().startswith("ROLLBACK"):
                raise RuntimeError("caller failure after the statement ran")
            return result

    db = CharactersRAGDB(tmp_path / "ran-then-failed.sqlite", "ran-then-failed")
    try:
        _seed_traced_message(db)
        conn = db.get_connection()
        authorization = db._semantic_mutation_authorization_for_coordinator(conn)
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute("SAVEPOINT failing_probe")
            generation = authorization._transaction_generation
            with pytest.raises(RuntimeError):
                conn.cursor(RaisesAfterRunning).execute("ROLLBACK TO failing_probe")
            assert conn.in_transaction
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


@pytest.mark.parametrize("end", ["commit", "rollback"])
def test_the_direct_commit_and_rollback_methods_report_a_boundary(end: str) -> None:
    """``connection.commit()``/``rollback()`` notify the listener themselves.

    Args:
        end: The connection method that ends the transaction.
    """
    conn = sqlite3.connect(":memory:", factory=base_db._QuiescentSQLiteConnection)
    try:
        conn.execute("CREATE TABLE t(x)")
        calls: list[None] = []
        conn.set_transaction_boundary_listener(lambda: calls.append(None))
        conn.execute("BEGIN")
        conn.execute("INSERT INTO t VALUES (1)")
        before = len(calls)
        getattr(conn, end)()
        assert not conn.in_transaction
        assert len(calls) == before + 1, f"{end}() did not report the boundary"
    finally:
        conn.close()


@pytest.mark.parametrize("end", ["commit", "rollback"])
def test_the_direct_methods_advance_the_managed_guard_generation(
    tmp_path: Path, end: str
) -> None:
    """Ending a transaction through the connection methods starts a new generation.

    Args:
        tmp_path: pytest fixture; holds this test's database file.
        end: The connection method that ends the transaction.
    """
    db = CharactersRAGDB(tmp_path / f"direct-{end}.sqlite", f"direct-{end}")
    try:
        _seed_traced_message(db)
        conn = db.get_connection()
        authorization = db._semantic_mutation_authorization_for_coordinator(conn)
        conn.execute("BEGIN IMMEDIATE")
        generation = authorization._transaction_generation
        getattr(conn, end)()
        assert authorization._transaction_generation != generation
    finally:
        db.close_connection()


def test_a_rejected_commit_inside_a_scope_keeps_the_authorization(
    tmp_path: Path,
) -> None:
    """A COMMIT the authorizer refuses never ran, so the transaction is unchanged.

    The trace callback only saw statements that executed. Reporting a keyword
    boundary for a refused COMMIT advanced the generation and refused the
    scope's next legitimate write (Qodo and cubic, #2894).

    Args:
        tmp_path: pytest fixture; holds this test's database file.
    """
    db = CharactersRAGDB(tmp_path / "refused-commit.sqlite", "refused-commit")
    try:
        message_id = _seed_traced_message(db)
        conn = db.get_connection()
        authorization = db._semantic_mutation_authorization_for_coordinator(conn)
        conn.execute("BEGIN IMMEDIATE")
        try:
            with authorization._authorize(
                message_id=message_id, operations={"message_update"}
            ):
                generation = authorization._transaction_generation
                # A unique text, so the statement cache cannot skip the
                # authorizer's prepare-time denial.
                with pytest.raises(sqlite3.DatabaseError):
                    conn.execute("COMMIT -- refused inside a mutation scope")
                assert conn.in_transaction
                assert authorization._transaction_generation == generation
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


def test_a_savepoint_rollback_advances_the_managed_guard_generation(
    tmp_path: Path,
) -> None:
    """``ROLLBACK TO`` keeps ``in_transaction`` True, yet ends the guarded work.

    The trace callback advanced the generation on any statement starting with
    BEGIN, COMMIT or ROLLBACK, savepoint rollbacks included; a cached one
    skips the authorizer's prepare-time denial, so the generation is the only
    backstop (Qodo, #2894). RELEASE never advanced it and still does not.

    Args:
        tmp_path: pytest fixture; holds this test's database file.
    """
    db = CharactersRAGDB(tmp_path / "savepoint.sqlite", "savepoint")
    try:
        _seed_traced_message(db)
        conn = db.get_connection()
        authorization = db._semantic_mutation_authorization_for_coordinator(conn)
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute("SAVEPOINT guard_probe")
            generation = authorization._transaction_generation
            conn.execute("ROLLBACK TO guard_probe")
            assert conn.in_transaction
            assert authorization._transaction_generation != generation
            generation = authorization._transaction_generation
            conn.execute("RELEASE guard_probe")
            assert authorization._transaction_generation == generation
        finally:
            conn.execute("ROLLBACK")
    finally:
        db.close_connection()


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


def test_a_3_mib_image_message_inserts_without_the_hex_expansion_cost(
    tmp_path: Path,
) -> None:
    """TASK-33263 AC#3, pinned: a 3 MiB image message no longer costs ~1.2 s.

    The trace callback made SQLite hex-render the bound BLOB for the statement
    and every trigger/FTS step (1,217 ms median, isolated profile); without it
    the insert measured 28.8 ms. The samples are this process's CPU time, not
    wall time: hex rendering is CPU work done here, so it shows in full, while
    a loaded runner's scheduling delays do not (a wall-clock median read
    313 ms with two 8-worker suites running; Qodo, #2894). The pin is 250 ms
    of CPU on the fastest of five inserts.

    Args:
        tmp_path: pytest fixture; holds this test's database file.
    """
    db = CharactersRAGDB(tmp_path / "blob-insert.sqlite", "blob-insert")
    try:
        conversation_id = db.add_conversation({"title": "blob"})
        assert conversation_id is not None
        image = os.urandom(3 * 1024 * 1024)
        samples = []
        for index in range(5):
            started = time.process_time()
            db.add_message(
                {
                    "conversation_id": conversation_id,
                    "sender": "user",
                    "content": f"image {index}",
                    "image_data": image,
                    "image_mime_type": "image/png",
                }
            )
            samples.append(time.process_time() - started)
    finally:
        db.close_connection()

    fastest_ms = min(samples) * 1000
    assert fastest_ms < 250, (
        f"fastest of five 3 MiB image inserts used {fastest_ms:.0f} ms CPU (pin 250 ms; "
        f"median {statistics.median(samples) * 1000:.0f} ms)"
    )
