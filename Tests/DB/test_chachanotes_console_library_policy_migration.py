"""V47 -> V48 Console Library policy and dispatch schema migration.

This file held the repo's exact current-schema-version pin while v48 was the
newest step. task-21128 added v48 -> v49, so the pin moved to
``Tests/DB/test_chachanotes_v49_messages_fts_update_scope.py`` -- the pin
belongs to the NEWEST migration's own file, so a schema bump touches the file
that caused it. The end-state assertions here now read
``_CURRENT_SCHEMA_VERSION`` instead of a literal; a version literal stays
correct only at a fixture's SEEDED starting point, never after an upgrade.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tldw_chatbook.Chat.console_library_policy import ConsoleLibraryMigrationSeed
from tldw_chatbook.DB.ChaChaNotes_DB import (
    CharactersRAGDB,
    SchemaError,
    _split_sql_statements,
    _strip_leading_sql_noise,
)
from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version


SCHEMA_NAME = "rag_char_chat_schema"
MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "tldw_chatbook"
    / "DB"
    / "migrations"
    / "chachanotes_v47_to_v48_console_library_policy.sql"
)
MESSAGE_SYNC_TRIGGERS = {
    "messages_sync_create",
    "messages_sync_update",
    "messages_sync_delete",
    "messages_sync_undelete",
}
FALSE_SEED = ConsoleLibraryMigrationSeed(auto_retrieve_on_send=False)
TRUE_SEED = ConsoleLibraryMigrationSeed(auto_retrieve_on_send=True)


def _version(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (SCHEMA_NAME,),
    ).fetchone()
    assert row is not None
    return int(row[0])


def _objects(connection: sqlite3.Connection) -> dict[str, tuple[str, str, str | None]]:
    return {
        row[1]: (row[0], row[2], row[3])
        for row in connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%'"
        )
    }


def _schema_snapshot(path: Path) -> tuple[int, tuple[tuple[object, ...], ...]]:
    with sqlite3.connect(path) as connection:
        schema_cookie = int(connection.execute("PRAGMA schema_version").fetchone()[0])
        objects = tuple(
            connection.execute(
                "SELECT type, name, tbl_name, sql FROM sqlite_master "
                "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
            ).fetchall()
        )
    return schema_cookie, objects


def _rollback_state(
    path: Path,
) -> tuple[int, tuple[int, tuple[tuple[object, ...], ...]], tuple[tuple[object, ...], ...]]:
    """Return version, complete schema, and policy rows for rollback assertions."""
    with sqlite3.connect(path) as connection:
        version = _version(connection)
        policy_table_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            ("console_conversation_library_policy",),
        ).fetchone()
        policy_rows = (
            tuple(
                connection.execute(
                    "SELECT * FROM console_conversation_library_policy "
                    "ORDER BY conversation_id"
                ).fetchall()
            )
            if policy_table_exists is not None
            else ()
        )
    return version, _schema_snapshot(path), policy_rows


def _insert_conversation(
    connection: sqlite3.Connection,
    conversation_id: str,
    *,
    deleted: int = 0,
) -> None:
    connection.execute(
        """
        INSERT INTO conversations(id, root_id, title, deleted, client_id, version)
        VALUES (?, ?, ?, ?, 'migration-fixture', 1)
        """,
        (conversation_id, conversation_id, conversation_id, deleted),
    )


def _build_v47(path: Path, *, with_conversations: bool = True) -> None:
    with chachanotes_db_at_version(path, 47) as db:
        if with_conversations:
            connection = db.get_connection()
            _insert_conversation(connection, "active-conversation")
            _insert_conversation(connection, "deleted-conversation", deleted=1)
            connection.commit()


@pytest.fixture(scope="module")
def v47_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("console-library-v47") / "template.sqlite"
    _build_v47(path)
    return path


def _copy_template(template: Path, destination: Path) -> None:
    shutil.copy2(template, destination)


def _column_contract(
    connection: sqlite3.Connection, table: str
) -> list[tuple[str, str, int, str | None, int]]:
    return [
        (row[1], row[2], int(row[3]), row[4], int(row[5]))
        for row in connection.execute(f'PRAGMA table_info("{table}")')
    ]


def _foreign_key_contract(
    connection: sqlite3.Connection, table: str
) -> set[tuple[str, str, str, str, str]]:
    return {
        (row[3], row[2], row[4], row[5], row[6])
        for row in connection.execute(f'PRAGMA foreign_key_list("{table}")')
    }


def test_real_v47_fixture_gains_exact_v48_local_schema_and_seed_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "schema.sqlite"
    _build_v47(path)
    with sqlite3.connect(path) as before:
        names = set(_objects(before))
        message_columns = {row[1] for row in before.execute("PRAGMA table_info(messages)")}
        assert "console_conversation_library_policy" not in names
        assert "console_dispatch_checkpoints" not in names
        assert "idx_console_dispatch_checkpoint_conversation" not in names
        assert "assistant_generation_state" not in message_columns

    db = CharactersRAGDB(
        path,
        client_id="upgrade",
        console_library_migration_seed=TRUE_SEED,
    )
    connection = db.get_connection()

    assert _version(connection) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
    assert _column_contract(connection, "console_conversation_library_policy") == [
        ("conversation_id", "TEXT", 0, None, 1),
        ("schema_version", "INTEGER", 1, "1", 0),
        ("auto_retrieve_on_send", "INTEGER", 1, "0", 0),
        ("assistant_library_access", "INTEGER", 1, "0", 0),
        ("policy_revision", "INTEGER", 1, "1", 0),
        ("updated_at", "DATETIME", 1, "CURRENT_TIMESTAMP", 0),
    ]
    assert _column_contract(connection, "console_dispatch_checkpoints") == [
        ("assistant_message_id", "TEXT", 0, None, 1),
        ("user_message_id", "TEXT", 1, None, 0),
        ("conversation_id", "TEXT", 1, None, 0),
        ("schema_version", "INTEGER", 1, "1", 0),
        ("preparation_id", "TEXT", 1, None, 0),
        ("attempt_id", "TEXT", 1, None, 0),
        ("state", "TEXT", 1, None, 0),
        ("checkpoint_revision", "INTEGER", 1, "1", 0),
        ("user_message_version", "INTEGER", 1, None, 0),
        ("assistant_message_version", "INTEGER", 1, None, 0),
        ("origin", "TEXT", 1, None, 0),
        ("queue_entry_id", "TEXT", 0, None, 0),
        ("frozen_authority_json", "TEXT", 1, None, 0),
        ("resolved_destination_json", "TEXT", 1, None, 0),
        ("reconstructability_json", "TEXT", 1, None, 0),
        ("created_at", "DATETIME", 1, "CURRENT_TIMESTAMP", 0),
        ("updated_at", "DATETIME", 1, "CURRENT_TIMESTAMP", 0),
    ]
    message_state = {
        row[1]: (row[2], int(row[3]), row[4], int(row[5]))
        for row in connection.execute("PRAGMA table_info(messages)")
    }["assistant_generation_state"]
    assert message_state == ("TEXT", 0, None, 0)

    assert _foreign_key_contract(
        connection, "console_conversation_library_policy"
    ) == {("conversation_id", "conversations", "id", "CASCADE", "CASCADE")}
    assert _foreign_key_contract(connection, "console_dispatch_checkpoints") == {
        ("assistant_message_id", "messages", "id", "NO ACTION", "CASCADE"),
        ("user_message_id", "messages", "id", "NO ACTION", "CASCADE"),
        ("conversation_id", "conversations", "id", "NO ACTION", "CASCADE"),
    }
    index = _objects(connection)["idx_console_dispatch_checkpoint_conversation"]
    assert index[0:2] == ("index", "console_dispatch_checkpoints")
    assert [
        row[2]
        for row in connection.execute(
            "PRAGMA index_info('idx_console_dispatch_checkpoint_conversation')"
        )
    ] == ["conversation_id"]

    policy_sql = _objects(connection)["console_conversation_library_policy"][2] or ""
    checkpoint_sql = _objects(connection)["console_dispatch_checkpoints"][2] or ""
    messages_sql = _objects(connection)["messages"][2] or ""
    compact_policy = " ".join(policy_sql.split())
    compact_checkpoint = " ".join(checkpoint_sql.split())
    compact_messages = " ".join(messages_sql.split())
    for clause in (
        "CHECK(schema_version > 0)",
        "CHECK(auto_retrieve_on_send IN (0, 1))",
        "CHECK(assistant_library_access IN (0, 1))",
        "CHECK(policy_revision > 0)",
    ):
        assert clause in compact_policy
    for clause in (
        "CHECK(state IN ('accepted', 'dispatch_started'))",
        "CHECK(origin IN ('manual', 'queued'))",
        "CHECK(schema_version > 0)",
        "CHECK(checkpoint_revision > 0)",
        "CHECK(user_message_version > 0)",
        "CHECK(assistant_message_version > 0)",
    ):
        assert clause in compact_checkpoint
    for state in (
        "'accepted'",
        "'dispatch_started'",
        "'continuation_active'",
        "'complete'",
        "'stopped'",
        "'failed'",
        "'discarded'",
    ):
        assert state in compact_messages
    assert "assistant_generation_state IS NULL" in compact_messages

    local_trigger_tables = {
        row[0]
        for row in connection.execute(
            "SELECT tbl_name FROM sqlite_master WHERE type = 'trigger'"
        )
    }
    assert "console_conversation_library_policy" not in local_trigger_tables
    assert "console_dispatch_checkpoints" not in local_trigger_tables
    assert [
        tuple(row)
        for row in connection.execute(
            "SELECT conversation_id, auto_retrieve_on_send, "
            "assistant_library_access FROM console_conversation_library_policy "
            "ORDER BY conversation_id"
        )
    ] == [
        ("active-conversation", 1, 1),
        ("deleted-conversation", 1, 1),
    ]

    _insert_conversation(connection, "after-migration")
    connection.commit()
    assert connection.execute(
        "SELECT 1 FROM console_conversation_library_policy "
        "WHERE conversation_id = 'after-migration'"
    ).fetchone() is None
    db.close_connection()


def test_v48_constraints_reject_out_of_contract_values(tmp_path: Path) -> None:
    db = CharactersRAGDB(tmp_path / "constraints.sqlite", client_id="constraints")
    connection = db.get_connection()
    _insert_conversation(connection, "constraint-conversation")
    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
        connection.execute(
            "INSERT INTO console_conversation_library_policy"
            "(conversation_id, auto_retrieve_on_send) VALUES (?, 2)",
            ("constraint-conversation",),
        )
    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
        connection.execute(
            """
            INSERT INTO messages(id, conversation_id, sender, content, client_id,
                                 assistant_generation_state)
            VALUES ('invalid-state', 'constraint-conversation', 'assistant', '',
                    'constraints', 'not-a-state')
            """
        )
    db.close_connection()


def test_all_final_message_sync_triggers_serialize_state_and_update_watches_it(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "triggers.sqlite", client_id="triggers")
    connection = db.get_connection()
    _insert_conversation(connection, "trigger-conversation")
    connection.execute("DELETE FROM sync_log")

    connection.execute(
        """
        INSERT INTO messages(id, conversation_id, sender, content, client_id,
                             assistant_generation_state)
        VALUES ('assistant-message', 'trigger-conversation', 'assistant', '',
                'triggers', NULL)
        """
    )
    create_payload = json.loads(
        connection.execute(
            "SELECT payload FROM sync_log WHERE entity_id = 'assistant-message' "
            "ORDER BY change_id DESC LIMIT 1"
        ).fetchone()[0]
    )
    assert "assistant_generation_state" in create_payload
    assert create_payload["assistant_generation_state"] is None

    connection.execute("DELETE FROM sync_log")
    connection.execute(
        "UPDATE messages SET assistant_generation_state = 'accepted' "
        "WHERE id = 'assistant-message'"
    )
    update = connection.execute(
        "SELECT operation, payload FROM sync_log WHERE entity_id = 'assistant-message'"
    ).fetchone()
    assert update[0] == "update"
    assert json.loads(update[1])["assistant_generation_state"] == "accepted"

    connection.execute("DELETE FROM sync_log")
    connection.execute(
        "UPDATE messages SET deleted = 1 WHERE id = 'assistant-message'"
    )
    deleted = connection.execute(
        "SELECT operation, payload FROM sync_log WHERE entity_id = 'assistant-message'"
    ).fetchone()
    assert deleted[0] == "delete"
    assert json.loads(deleted[1])["assistant_generation_state"] == "accepted"

    trigger_sql = {
        row[0]: row[1]
        for row in connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'trigger'"
        )
        if row[0] in MESSAGE_SYNC_TRIGGERS
    }
    assert set(trigger_sql) == MESSAGE_SYNC_TRIGGERS
    assert all("assistant_generation_state" in sql for sql in trigger_sql.values())
    assert (
        "OLD.assistant_generation_state IS NOT NEW.assistant_generation_state"
        in trigger_sql["messages_sync_update"]
    )
    db.close_connection()


class _TraceOpeningDB(CharactersRAGDB):
    """Capture the real SQL trace before the initializer's first version read."""

    first_version_read_trace: tuple[str, ...]
    first_version_read_in_transaction: bool

    def get_connection(self) -> sqlite3.Connection:
        connection = super().get_connection()
        if not hasattr(self._local, "schema_trace"):
            self._local.schema_trace = []
            connection.set_trace_callback(self._local.schema_trace.append)
        return connection

    def _get_db_version(self, conn: sqlite3.Connection) -> int:
        if not hasattr(self, "first_version_read_trace"):
            self.first_version_read_trace = tuple(self._local.schema_trace)
            self.first_version_read_in_transaction = conn.in_transaction
        return super()._get_db_version(conn)


@pytest.mark.parametrize("starting_shape", ["fresh", "current", "v47", "older"])
def test_initializer_begins_immediate_before_its_first_version_read(
    tmp_path: Path,
    starting_shape: str,
) -> None:
    path = tmp_path / f"{starting_shape}.sqlite"
    if starting_shape == "current":
        current = CharactersRAGDB(path, client_id="current-fixture")
        current.close_connection()
    elif starting_shape == "v47":
        _build_v47(path, with_conversations=False)
    elif starting_shape == "older":
        with chachanotes_db_at_version(path, 43):
            pass

    seed = TRUE_SEED if starting_shape in {"v47", "older"} else None
    db = _TraceOpeningDB(
        path,
        client_id=f"trace-{starting_shape}",
        console_library_migration_seed=seed,
    )

    assert db.first_version_read_in_transaction is True
    assert db.first_version_read_trace == ("BEGIN IMMEDIATE",)
    db.close_connection()


def test_v47_invalid_seed_leaves_schema_unchanged(
    tmp_path: Path,
    v47_template: Path,
) -> None:
    """A wrong-typed seed is a caller defect and still stops before the DDL.

    The ``None`` half of this parametrization inverted in task-21441 -- see
    ``test_v47_absent_seed_migrates_with_retrieval_off`` below. Absent is a
    legitimate state with a defined default; malformed is not.
    """
    path = tmp_path / "invalid-seed.sqlite"
    _copy_template(v47_template, path)
    before = _schema_snapshot(path)

    with pytest.raises(SchemaError, match="must be a ConsoleLibraryMigrationSeed"):
        CharactersRAGDB(
            path,
            client_id="invalid-seed",
            console_library_migration_seed=object(),  # type: ignore[arg-type]
        )

    with sqlite3.connect(path) as connection:
        assert _version(connection) == 47
    assert _schema_snapshot(path) == before


def test_v47_absent_seed_migrates_with_retrieval_off(
    tmp_path: Path,
    v47_template: Path,
) -> None:
    """No seed is not a failure: the step defaults, and the default is safe.

    This case used to raise ``SchemaError`` and is the whole reason
    ``CharactersRAGDB`` could not migrate itself (task-21441). The result must
    be indistinguishable from an explicit ``auto_retrieve_on_send=False``,
    including for the soft-deleted conversation the template carries.
    """
    absent = tmp_path / "absent-seed.sqlite"
    explicit = tmp_path / "explicit-false-seed.sqlite"
    _copy_template(v47_template, absent)
    _copy_template(v47_template, explicit)

    CharactersRAGDB(absent, client_id="absent-seed").close_connection()
    CharactersRAGDB(
        explicit,
        client_id="explicit-seed",
        console_library_migration_seed=FALSE_SEED,
    ).close_connection()

    for path in (absent, explicit):
        with sqlite3.connect(path) as connection:
            assert _version(connection) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
            assert sorted(
                connection.execute(
                    "SELECT conversation_id, auto_retrieve_on_send,"
                    " assistant_library_access"
                    " FROM console_conversation_library_policy"
                ).fetchall()
            ) == [
                ("active-conversation", 0, 1),
                ("deleted-conversation", 0, 1),
            ]


@pytest.mark.parametrize("failure_index", range(12))
def test_failure_after_each_v48_ddl_statement_rolls_back_everything(
    tmp_path: Path,
    v47_template: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_index: int,
) -> None:
    path = tmp_path / f"failure-{failure_index}.sqlite"
    _copy_template(v47_template, path)
    before = _schema_snapshot(path)
    statements = _split_sql_statements(MIGRATION_PATH.read_text(encoding="utf-8"))
    assert len(statements) == 12
    original_execute = CharactersRAGDB._execute_migration_statements

    def execute_with_failure(
        self: CharactersRAGDB,
        cursor: sqlite3.Cursor,
        script: str,
        label: str,
    ) -> None:
        if label != "V47→V48":
            original_execute(self, cursor, script, label)
            return
        for index, statement in enumerate(_split_sql_statements(script)):
            head = _strip_leading_sql_noise(statement)
            if not head:
                continue
            cursor.execute(statement)
            if index == failure_index:
                raise sqlite3.OperationalError(f"injected-v48-{failure_index}")

    monkeypatch.setattr(
        CharactersRAGDB,
        "_execute_migration_statements",
        execute_with_failure,
    )
    with pytest.raises(SchemaError, match=f"injected-v48-{failure_index}"):
        CharactersRAGDB(
            path,
            client_id="rollback",
            console_library_migration_seed=TRUE_SEED,
        )

    assert _schema_snapshot(path) == before
    with sqlite3.connect(path) as connection:
        assert _version(connection) == 47
        assert "assistant_generation_state" not in {
            row[1] for row in connection.execute("PRAGMA table_info(messages)")
        }


def test_failure_after_policy_seed_insert_rolls_back_schema_rows_and_version(
    tmp_path: Path,
    v47_template: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "failure-after-policy-seed.sqlite"
    _copy_template(v47_template, path)
    before = _rollback_state(path)
    assert before[0] == 47
    assert before[2] == ()
    original_seed = getattr(
        CharactersRAGDB,
        "_seed_console_library_policy_rows",
        None,
    )

    def fail_after_seed(
        self: CharactersRAGDB,
        cursor: sqlite3.Cursor,
        auto_retrieve_on_send: int,
    ) -> None:
        if original_seed is not None:
            original_seed(self, cursor, auto_retrieve_on_send)
        raise sqlite3.OperationalError("injected-after-policy-seed")

    monkeypatch.setattr(
        CharactersRAGDB,
        "_seed_console_library_policy_rows",
        fail_after_seed,
        raising=False,
    )
    with pytest.raises(SchemaError, match="injected-after-policy-seed"):
        CharactersRAGDB(
            path,
            client_id="rollback-after-seed",
            console_library_migration_seed=TRUE_SEED,
        )

    assert _rollback_state(path) == before


def test_failure_after_guarded_version_update_rolls_back_schema_rows_and_version(
    tmp_path: Path,
    v47_template: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "failure-after-version-update.sqlite"
    _copy_template(v47_template, path)
    before = _rollback_state(path)
    assert before[0] == 47
    assert before[2] == ()
    original_update = getattr(
        CharactersRAGDB,
        "_update_console_library_policy_schema_version",
        None,
    )

    def fail_after_version_update(
        self: CharactersRAGDB,
        cursor: sqlite3.Cursor,
    ) -> None:
        if original_update is not None:
            original_update(self, cursor)
        raise sqlite3.OperationalError("injected-after-v48-version-update")

    monkeypatch.setattr(
        CharactersRAGDB,
        "_update_console_library_policy_schema_version",
        fail_after_version_update,
        raising=False,
    )
    with pytest.raises(SchemaError, match="injected-after-v48-version-update"):
        CharactersRAGDB(
            path,
            client_id="rollback-after-version",
            console_library_migration_seed=TRUE_SEED,
        )

    assert _rollback_state(path) == before


def test_failed_migration_retries_with_a_different_seed(
    tmp_path: Path,
    v47_template: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "retry.sqlite"
    _copy_template(v47_template, path)
    original_execute = CharactersRAGDB._execute_migration_statements

    def fail_after_ddl(
        self: CharactersRAGDB,
        cursor: sqlite3.Cursor,
        script: str,
        label: str,
    ) -> None:
        original_execute(self, cursor, script, label)
        if label == "V47→V48":
            raise sqlite3.OperationalError("retry-seed-failure")

    monkeypatch.setattr(
        CharactersRAGDB, "_execute_migration_statements", fail_after_ddl
    )
    with pytest.raises(SchemaError, match="retry-seed-failure"):
        CharactersRAGDB(
            path,
            client_id="failed-true",
            console_library_migration_seed=TRUE_SEED,
        )

    monkeypatch.setattr(
        CharactersRAGDB, "_execute_migration_statements", original_execute
    )
    db = CharactersRAGDB(
        path,
        client_id="retry-false",
        console_library_migration_seed=FALSE_SEED,
    )
    values = {
        row[0]
        for row in db.get_connection().execute(
            "SELECT auto_retrieve_on_send FROM console_conversation_library_policy"
        )
    }
    assert values == {0}
    db.close_connection()


def test_two_concurrent_openers_converge_on_one_complete_seed(
    tmp_path: Path,
    v47_template: Path,
) -> None:
    path = tmp_path / "concurrent.sqlite"
    _copy_template(v47_template, path)
    ready = threading.Barrier(2)
    calls: list[bool] = []
    calls_lock = threading.Lock()

    class RacingDB(CharactersRAGDB):
        def get_connection(self) -> sqlite3.Connection:
            connection = super().get_connection()
            if not hasattr(self._local, "ready_for_race"):
                self._local.ready_for_race = True
                ready.wait(timeout=10)
            return connection

        def _migrate_from_v47_to_v48(self, conn: sqlite3.Connection) -> None:
            seed = self.console_library_migration_seed
            assert isinstance(seed, ConsoleLibraryMigrationSeed)
            with calls_lock:
                calls.append(seed.auto_retrieve_on_send)
            super()._migrate_from_v47_to_v48(conn)

    def open_with(seed: ConsoleLibraryMigrationSeed) -> tuple[int, tuple[int, ...]]:
        db = RacingDB(
            path,
            client_id=f"racer-{int(seed.auto_retrieve_on_send)}",
            console_library_migration_seed=seed,
        )
        connection = db.get_connection()
        result = (
            _version(connection),
            tuple(
                row[0]
                for row in connection.execute(
                    "SELECT auto_retrieve_on_send "
                    "FROM console_conversation_library_policy "
                    "ORDER BY conversation_id"
                )
            ),
        )
        db.close_connection()
        return result

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(open_with, seed) for seed in (TRUE_SEED, FALSE_SEED)]
        results = [future.result(timeout=20) for future in futures]

    assert len(calls) == 1
    winner = int(calls[0])
    current = CharactersRAGDB._CURRENT_SCHEMA_VERSION
    assert results == [
        (current, (winner, winner)),
        (current, (winner, winner)),
    ]
