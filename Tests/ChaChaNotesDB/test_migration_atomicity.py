"""Atomicity and re-enterability of the ChaChaNotes migration chain (task-19553).

Why this exists — the incident, not the rule. The 2026-08-21 holistic review's
Lane 3 ran a live experiment, and this module reproduces it: on a genuine v11
database with ONE of the v11→v12 ``ADD COLUMN`` statements already applied (the
shape an interrupted migration leaves), ``conn.executescript`` COMMITTED the
three ``ALTER``s that ran before the duplicate, then failed — while the schema
version stamp stayed at 11. Every subsequent launch re-entered the step,
re-raised ``duplicate column name`` (on a *different* column the second time),
and ``CharactersRAGDB.__init__`` failed permanently. Conversations, notes and
characters became unreachable with no in-app recovery path.

``executescript`` is the mechanism: it commits whatever transaction is open and
then autocommits each statement individually, so a step driven that way can
neither roll back nor be re-entered. The migration steps now run their scripts
one statement at a time through ``_execute_migration_statements`` inside the
caller's transaction.

What each test pins:

* ``test_interrupted_add_column_step_recovers`` — the audit's exact scenario,
  now reaching the current version, with a schema identical to a clean replay.
* ``test_interrupted_trigger_step_recovers`` — the same for V7→V8, whose bare
  ``CREATE TRIGGER`` statements are the other half-applied shape.
* ``test_failure_mid_step_leaves_no_partial_ddl`` — a statement that fails
  part-way through a step rewinds the run to its entry version with nothing
  applied, and the same file migrates on the next attempt.
* ``test_long_chain_from_v4_*`` — the reachable population is v4–v25
  databases replaying the longest chain, so the long chain is tested with real
  rows in it.
* ``test_no_migration_step_uses_executescript`` — a source-level pin so the
  defect cannot be reintroduced by a new step copying an old one.
"""

from __future__ import annotations

import inspect
import re
import sqlite3
import uuid
from pathlib import Path

import pytest

from tldw_chatbook.Chat.console_library_policy import ConsoleLibraryMigrationSeed
from tldw_chatbook.DB.ChaChaNotes_DB import (
    CharactersRAGDB,
    SchemaError,
    _split_sql_statements,
    _strip_leading_sql_noise,
)
from Tests.ChaChaNotesDB.historical_bootstrap import (
    MINIMUM_BOOTSTRAP_VERSION,
    SCHEMA_NAME,
    chachanotes_db_at_version,
)

CURRENT = CharactersRAGDB._CURRENT_SCHEMA_VERSION
MIGRATION_SEED = ConsoleLibraryMigrationSeed(auto_retrieve_on_send=False)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _version(path: Path) -> int | None:
    """Read the stamped schema version straight off disk."""
    connection = sqlite3.connect(str(path))
    try:
        row = connection.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (SCHEMA_NAME,),
        ).fetchone()
        return row[0] if row else None
    finally:
        connection.close()


def _columns(path: Path, table: str) -> list[str]:
    """Column names of ``table`` in declaration order, read off disk."""
    connection = sqlite3.connect(str(path))
    try:
        return [
            row[0]
            for row in connection.execute(
                "SELECT name FROM pragma_table_info(?)", (table,)
            ).fetchall()
        ]
    finally:
        connection.close()


def _pre_apply_statements(path: Path, script: str, count: int) -> list[str]:
    """Apply and COMMIT the first ``count`` statements of a migration script.

    This is exactly the residue ``executescript`` leaves when it is killed
    part-way through: the statements that already ran are committed, and the
    version stamp — updated by the script's LAST statement — is not.

    Args:
        path: The database file.
        script: A migration script constant.
        count: How many leading statements to apply.

    Returns:
        The statements that were applied (for the failure message).
    """
    applied = _split_sql_statements(script)[:count]
    connection = sqlite3.connect(str(path))
    try:
        for statement in applied:
            connection.execute(statement)
        connection.commit()
    finally:
        connection.close()
    return applied


def _schema_fingerprint(connection: sqlite3.Connection) -> dict[str, object]:
    """Full schema shape: object SQL text plus ORDERED column definitions."""
    master = sorted(
        (row[0], row[1], row[3])
        for row in connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master"
        ).fetchall()
    )
    columns: dict[str, list[tuple]] = {}
    for (table,) in connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall():
        columns[table] = [
            tuple(row)
            for row in connection.execute(f'PRAGMA table_info("{table}")').fetchall()
        ]
    return {"master": master, "columns": columns}


def _fingerprint_of_clean_replay(tmp_path: Path, from_version: int) -> dict:
    """Fingerprint of an UNINTERRUPTED bootstrap-at-N-then-replay database."""
    path = tmp_path / f"clean_replay_v{from_version}.sqlite"
    with chachanotes_db_at_version(path, from_version):
        pass
    db = CharactersRAGDB(
        str(path),
        client_id="clean-replay",
        console_library_migration_seed=MIGRATION_SEED,
    )
    try:
        return _schema_fingerprint(db.get_connection())
    finally:
        db.close_connection()


def _open_current(path: Path, client_id: str) -> CharactersRAGDB:
    return CharactersRAGDB(
        str(path),
        client_id=client_id,
        console_library_migration_seed=MIGRATION_SEED,
    )


# --------------------------------------------------------------------------
# the audit's brick, and its two shapes
# --------------------------------------------------------------------------
class TestInterruptedStepIsRecoverable:
    """A half-applied step must not strand the database."""

    def test_interrupted_add_column_step_recovers(self, tmp_path: Path) -> None:
        """The audit's experiment: v11 + 3 of v11→v12's 4 ``ALTER``s applied.

        Before task-19553 this raised ``duplicate column name`` on every
        launch forever. It must now migrate to the current version, and the
        resulting schema must be indistinguishable from a clean replay.
        """
        path = tmp_path / "interrupted_v11_to_v12.sqlite"
        with chachanotes_db_at_version(path, 11):
            pass
        assert _version(path) == 11

        applied = _pre_apply_statements(
            path, CharactersRAGDB._MIGRATE_V11_TO_V12_SQL, 3
        )
        assert len(applied) == 3, applied
        partial_columns = _columns(path, "messages")
        assert "variant_of" in partial_columns
        assert "total_variants" not in partial_columns, (
            "fixture must leave the step genuinely HALF applied"
        )
        assert _version(path) == 11, "an interrupted step never bumps the stamp"

        db = _open_current(path, "interrupted-add-column")
        try:
            assert _version(path) == CURRENT
            recovered = _schema_fingerprint(db.get_connection())
        finally:
            db.close_connection()

        assert recovered == _fingerprint_of_clean_replay(tmp_path, 11), (
            "a database recovered from a half-applied step must end up with "
            "byte-identical schema SQL and identical column order to one that "
            "was never interrupted"
        )

        # And it keeps opening: the brick failed on the SECOND launch too.
        db = _open_current(path, "interrupted-add-column-again")
        try:
            assert _version(path) == CURRENT
        finally:
            db.close_connection()

    def test_interrupted_trigger_step_recovers(self, tmp_path: Path) -> None:
        """V7→V8 creates triggers with no ``IF NOT EXISTS`` and no ``DROP``.

        A half-applied V7→V8 therefore strands on ``trigger ... already
        exists`` rather than ``duplicate column name`` — the same brick with a
        different error string.
        """
        path = tmp_path / "interrupted_v7_to_v8.sqlite"
        with chachanotes_db_at_version(path, 7):
            pass
        assert _version(path) == 7

        _pre_apply_statements(path, CharactersRAGDB._MIGRATE_V7_TO_V8_SQL, 3)
        connection = sqlite3.connect(str(path))
        try:
            triggers = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'trigger'"
                ).fetchall()
            }
        finally:
            connection.close()
        assert "chat_dictionaries_ai" in triggers, (
            "fixture must leave a bare CREATE TRIGGER already applied"
        )
        assert _version(path) == 7

        db = _open_current(path, "interrupted-trigger")
        try:
            assert _version(path) == CURRENT
            recovered = _schema_fingerprint(db.get_connection())
        finally:
            db.close_connection()

        assert recovered == _fingerprint_of_clean_replay(tmp_path, 7)

    def test_long_chain_from_v4_with_interrupted_step_recovers(
        self, tmp_path: Path
    ) -> None:
        """AC #5: the reachable population is old databases with real rows.

        A v4-era database (the oldest the chain can build) carrying user data,
        interrupted inside V11→V12, must reach the current version with its
        rows intact.
        """
        path = tmp_path / "long_chain_v4.sqlite"
        conversation_id = str(uuid.uuid4())
        note_id = str(uuid.uuid4())
        with chachanotes_db_at_version(path, MINIMUM_BOOTSTRAP_VERSION) as db:
            connection = db.get_connection()
            connection.execute(
                """
                INSERT INTO conversations(
                    id, root_id, title, created_at, last_modified,
                    deleted, client_id, version
                ) VALUES (?, ?, ?, ?, ?, 0, ?, 1)
                """,
                (
                    conversation_id,
                    conversation_id,
                    "Survives the upgrade",
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:00:00Z",
                    "long-chain",
                ),
            )
            connection.execute(
                """
                INSERT INTO notes(
                    id, title, content, created_at, last_modified,
                    deleted, client_id, version
                ) VALUES (?, ?, ?, ?, ?, 0, ?, 1)
                """,
                (
                    note_id,
                    "Old note",
                    "old body",
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:00:00Z",
                    "long-chain",
                ),
            )
            connection.commit()
        assert _version(path) == MINIMUM_BOOTSTRAP_VERSION

        # Walk the chain to v11 the way a real upgrade would, stop there, then
        # leave V11->V12 half applied.
        with chachanotes_db_at_version(path, 11):
            pass
        assert _version(path) == 11
        _pre_apply_statements(path, CharactersRAGDB._MIGRATE_V11_TO_V12_SQL, 2)

        db = _open_current(path, "long-chain-recovered")
        try:
            connection = db.get_connection()
            assert _version(path) == CURRENT
            assert (
                connection.execute(
                    "SELECT title FROM conversations WHERE id = ?", (conversation_id,)
                ).fetchone()["title"]
                == "Survives the upgrade"
            )
            assert (
                connection.execute(
                    "SELECT content FROM notes WHERE id = ?", (note_id,)
                ).fetchone()["content"]
                == "old body"
            )
        finally:
            db.close_connection()


class TestFailingStepRollsBack:
    """A step that cannot finish must leave nothing behind."""

    def test_failure_mid_step_leaves_no_partial_ddl(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Poison statement 7 of V12→V13's 32 and require a full rewind.

        Before task-19553 the six ``ALTER``s ahead of the poison committed
        individually and stayed on disk. The run must now rewind to its ENTRY
        version — 11, because V11→V12 is part of the same transaction — with
        none of the columns from either step present.
        """
        path = tmp_path / "poisoned_v12_to_v13.sqlite"
        with chachanotes_db_at_version(path, 11):
            pass

        statements = _split_sql_statements(CharactersRAGDB._MIGRATE_V12_TO_V13_SQL)
        assert len(statements) > 8, len(statements)
        poisoned = "".join(
            statements[:6]
            + ["\nINSERT INTO no_such_table_19553(x) VALUES (1);\n"]
            + statements[6:]
        )
        monkeypatch.setattr(
            CharactersRAGDB, "_MIGRATE_V12_TO_V13_SQL", poisoned, raising=True
        )

        with pytest.raises(SchemaError, match="no_such_table_19553"):
            _open_current(path, "poisoned")

        assert _version(path) == 11, (
            "a failing step must leave the stamp at the run's entry version"
        )
        message_columns = _columns(path, "messages")
        assert "variant_of" not in message_columns, (
            "V11->V12's DDL must have rolled back with the failing run"
        )
        conversation_columns = _columns(path, "conversations")
        assert "assistant_kind" not in conversation_columns, (
            "the six ALTERs ahead of the poison must NOT survive"
        )

        # Re-enterable: with the poison removed the same file migrates.
        monkeypatch.undo()
        db = _open_current(path, "poison-removed")
        try:
            assert _version(path) == CURRENT
        finally:
            db.close_connection()

    def test_interrupted_base_schema_apply_recovers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The v4 base apply leaves nothing behind when it fails.

        Scope this claim carefully -- it is NOT the brick the migration steps
        had. ``_FULL_SCHEMA_SQL_V4`` is already re-enterable on its own terms:
        42 ``CREATE TRIGGER`` statements but 42 matching
        ``DROP TRIGGER IF EXISTS`` (zero creates without a preceding drop),
        every ``CREATE TABLE``/``VIRTUAL TABLE``/``INDEX`` carrying
        ``IF NOT EXISTS``, and both top-level inserts ``INSERT OR IGNORE``.
        Sweeping all 120 interruption points of the script on the pre-fix
        code, the retry reached the current version 120 times out of 120.

        What changed is the leftover state: pre-fix, the worst interruption
        point left 111 ``sqlite_master`` rows committed in a file the caller
        was told had failed to initialize, and 119 of the 120 points left
        something. This test pins the post-fix number -- zero, at every point
        -- which is why its assertion is about leftovers and not about the
        retry. (The retry is checked at the end too, but it passed before the
        fix as well and is not the property under test.)
        """
        path = tmp_path / "interrupted_base.sqlite"
        statements = _split_sql_statements(CharactersRAGDB._FULL_SCHEMA_SQL_V4)
        assert len(statements) > 60, len(statements)
        poisoned = "".join(
            statements[:50]
            + ["\nINSERT INTO no_such_table_19553(x) VALUES (1);\n"]
            + statements[50:]
        )
        monkeypatch.setattr(
            CharactersRAGDB, "_FULL_SCHEMA_SQL_V4", poisoned, raising=True
        )

        with pytest.raises(SchemaError, match="no_such_table_19553"):
            _open_current(path, "poisoned-base")

        connection = sqlite3.connect(str(path))
        try:
            leftovers = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
                ).fetchall()
            }
        finally:
            connection.close()
        assert not leftovers, (
            f"an interrupted base-schema apply left objects behind: "
            f"{sorted(leftovers)}"
        )

        monkeypatch.undo()
        db = _open_current(path, "base-retry")
        try:
            assert _version(path) == CURRENT
        finally:
            db.close_connection()

    def test_foreign_keys_remain_enforced_on_a_fresh_database(
        self, tmp_path: Path
    ) -> None:
        """The one statement deliberately left outside the transaction.

        ``_FULL_SCHEMA_SQL_V4`` opens with ``PRAGMA foreign_keys = ON``, which
        SQLite silently IGNORES inside a transaction. The guarantee is carried
        by ``_get_thread_connection`` instead, which issues the same pragma on
        every connection — this test is the pin for that hand-off.
        """
        path = tmp_path / "fk_enforced.sqlite"
        db = _open_current(path, "fk-pin")
        try:
            connection = db.get_connection()
            assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    """
                    INSERT INTO messages(
                        id, conversation_id, sender, content, timestamp,
                        client_id, version
                    ) VALUES (?, ?, 'user', 'x', '2026-01-01T00:00:00Z', 'fk', 1)
                    """,
                    (str(uuid.uuid4()), "no-such-conversation"),
                )
        finally:
            db.close_connection()


class TestEntryVersionGuards:
    """Every ported step refuses to run against the wrong version."""

    @pytest.mark.parametrize(
        ("step", "entry_version"),
        [
            ("_migrate_from_v4_to_v5", 4),
            ("_migrate_from_v7_to_v8", 7),
            ("_migrate_from_v11_to_v12", 11),
            ("_migrate_from_v12_to_v13", 12),
            ("_migrate_from_v17_to_v18", 17),
            ("_migrate_from_v28_to_v29", 28),
        ],
    )
    def test_step_rejects_a_database_at_the_wrong_version(
        self, tmp_path: Path, step: str, entry_version: int
    ) -> None:
        path = tmp_path / f"guard_{step}.sqlite"
        with chachanotes_db_at_version(path, entry_version + 1) as db:
            with pytest.raises(SchemaError, match="requires schema version"):
                getattr(db, step)(db.get_connection())

    def test_every_migration_step_has_an_entry_version_guard(self) -> None:
        """No step may run without first checking the version it expects."""
        unguarded = []
        for name, method in _migration_steps():
            source = inspect.getsource(method)
            if (
                "_require_migration_entry_version" not in source
                and "_get_db_version(conn) != " not in source
                and "requires schema version" not in source
            ):
                unguarded.append(name)
        assert not unguarded, (
            "these ChaChaNotes migration steps have no entry-version guard, so "
            "they can be re-entered against an already-advanced database: "
            f"{unguarded}"
        )


def _remaining_sql_after_first_statement(chunk: str) -> str:
    """Return the SQL left over after ``chunk``'s FIRST complete statement.

    The splitter completes a chunk only at a LINE boundary, so two statements
    written on one source line share a chunk — and ``cursor.execute`` rejects
    that. Finding the boundary needs character granularity, which this gets by
    testing ``sqlite3.complete_statement`` at each ``;``: SQLite reports False
    for a ``;`` inside a string, a comment, or a ``BEGIN``/``END`` trigger
    body, so the first True is the real end of statement one.

    Args:
        chunk: One chunk as produced by ``_split_sql_statements``.

    Returns:
        The trailing SQL after the first statement with leading whitespace and
        comments removed, or ``""`` when the chunk holds exactly one.
    """
    for index, char in enumerate(chunk):
        if char != ";":
            continue
        if sqlite3.complete_statement(chunk[: index + 1]):
            return _strip_leading_sql_noise(chunk[index + 1 :])
    return ""


def _code_only(source: str) -> str:
    """Drop whole-line ``#`` comments so prose about the defect is not a hit."""
    return "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )


def _migration_steps() -> list[tuple[str, object]]:
    """Every ``_migrate_from_vN_to_vM`` method on ``CharactersRAGDB``."""
    steps = [
        (name, getattr(CharactersRAGDB, name))
        for name in dir(CharactersRAGDB)
        if re.fullmatch(r"_migrate_from_v\d+_to_v\d+", name)
    ]
    assert len(steps) >= 38, f"expected the full chain, found {len(steps)}"
    return steps


class TestExecutescriptIsGone:
    """Source-level pin: the mechanism itself must not come back."""

    def test_no_migration_step_uses_executescript(self) -> None:
        offenders = [
            name
            for name, method in _migration_steps()
            if ".executescript(" in _code_only(inspect.getsource(method))
        ]
        assert not offenders, (
            "these ChaChaNotes migration steps call executescript, which "
            "COMMITS the caller's transaction and then autocommits each "
            "statement — a failure part-way through leaves committed DDL with "
            "a stale version stamp and permanently bricks the database "
            f"(task-19553): {offenders}. Use "
            "`self._execute_migration_statements(cursor, script, label)` "
            "inside `with self.transaction() as cursor:` instead."
        )

    def test_base_schema_apply_does_not_use_executescript(self) -> None:
        source = _code_only(inspect.getsource(CharactersRAGDB._apply_schema_v4))
        assert ".executescript(" not in source, (
            "_apply_schema_v4 calls executescript, so an interrupted "
            "base-schema apply commits its DDL into a file the caller was "
            "told had failed to initialize — measured at up to 111 "
            "sqlite_master rows, on 119 of the script's 120 interruption "
            "points (task-19553). Unlike the migration steps this is a "
            "leftover-state defect, NOT a brick: the v4 script is internally "
            "re-enterable and retried successfully 120/120 times even before "
            "the fix. It also reintroduces the one commit that would make "
            "'no step commits' conditional again."
        )


class TestStatementSplitting:
    """The runner's two text primitives, pinned directly."""

    def test_trigger_bodies_are_not_split_on_inner_semicolons(self) -> None:
        script = (
            "CREATE TRIGGER t AFTER INSERT ON x BEGIN\n"
            "  INSERT INTO y VALUES (1);\n"
            "  INSERT INTO y VALUES (2);\n"
            "END;\n"
            "CREATE INDEX i ON x(a);\n"
        )
        statements = _split_sql_statements(script)
        assert len(statements) == 2, statements
        assert statements[0].count("INSERT INTO y") == 2

    def test_multi_statement_chunk_detector_actually_fires(self) -> None:
        """Mutation control for the exactly-one-statement pin above.

        Two statements on ONE line share a chunk (the splitter can only break
        at a line boundary) and `cursor.execute` rejects that — so the pin is
        only worth anything if this detector goes red on it.
        """
        one_line = "CREATE INDEX a ON t(x); CREATE INDEX b ON t(y);\n"
        chunks = _split_sql_statements(one_line)
        assert len(chunks) == 1, "the splitter cannot break inside a line"
        assert _remaining_sql_after_first_statement(chunks[0]).startswith(
            "CREATE INDEX b"
        )
        with sqlite3.connect(":memory:") as connection:
            connection.execute("CREATE TABLE t(x, y)")
            with pytest.raises(sqlite3.ProgrammingError):
                connection.execute(chunks[0])

        # ...and stays quiet on the shapes that are legitimately one statement.
        for single in (
            "CREATE INDEX a ON t(x);\n",
            "INSERT INTO t VALUES ('a;b');\n",
            "CREATE TRIGGER tr AFTER INSERT ON t BEGIN\n"
            "  INSERT INTO t VALUES (1, 2);\n"
            "  INSERT INTO t VALUES (3, 4);\n"
            "END;\n",
            "-- lead\nCREATE INDEX a ON t(x); -- trail\n",
        ):
            chunk = _split_sql_statements(single)[0]
            assert _remaining_sql_after_first_statement(chunk) == "", single

    def test_incomplete_trailing_sql_is_rejected(self) -> None:
        with pytest.raises(SchemaError, match="incomplete SQL statement"):
            _split_sql_statements("CREATE TABLE t(a INT)\n")

    @pytest.mark.parametrize(
        ("raw", "expected_head"),
        [
            ("\n-- lead\nALTER TABLE t ADD COLUMN c TEXT;\n", "ALTER"),
            ("/* block */\nCREATE INDEX i ON t(a);\n", "CREATE"),
            ("  \n-- only a comment\n", ""),
        ],
    )
    def test_leading_comments_are_stripped_for_matching_only(
        self, raw: str, expected_head: str
    ) -> None:
        head = _strip_leading_sql_noise(raw)
        assert head.startswith(expected_head)

    def test_every_shipped_migration_script_splits_cleanly(self) -> None:
        """Each embedded script must split into SINGLE runnable statements.

        Splittability alone is not enough. ``sqlite3.complete_statement`` is
        checked per accumulated LINE, so two statements written on one source
        line land in the same chunk — and ``cursor.execute`` refuses a chunk
        holding more than one statement, which would turn a formatting choice
        in a future migration into a failed upgrade. The
        ``sqlite3_stmt``-level count below is what actually closes that trap;
        it holds for all statements shipped today, base script included.
        """
        scripts = [
            name
            for name in dir(CharactersRAGDB)
            if re.fullmatch(r"_MIGRATE_V\d+_TO_V\d+_SQL", name)
        ] + ["_FULL_SCHEMA_SQL_V4"]
        assert len(scripts) > 20, scripts
        checked = 0
        for name in scripts:
            statements = _split_sql_statements(getattr(CharactersRAGDB, name))
            assert statements, name
            for index, statement in enumerate(statements):
                head = _strip_leading_sql_noise(statement)
                assert head, f"{name}[{index}] produced a chunk with no SQL in it"
                remainder = _remaining_sql_after_first_statement(statement)
                assert not remainder, (
                    f"{name}[{index}] holds MORE THAN ONE statement, so "
                    f"cursor.execute would reject it: trailing SQL is "
                    f"{remainder[:120]!r}. Put each statement on its own "
                    f"line(s) — the splitter completes a chunk only at a line "
                    f"boundary."
                )
                checked += 1
        assert checked > 300, f"expected the full corpus, counted {checked}"
