"""v41 -> v42 local-only Console project-context migration contracts."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    SchemaError,
)


SCHEMA_NAME = "rag_char_chat_schema"
COLUMN_NAME = "console_project_context_json"


def _version(connection) -> int:
    row = connection.execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (SCHEMA_NAME,),
    ).fetchone()
    return int(row[0])


def _conversation_columns(connection) -> dict[str, object]:
    return {
        str(row[1]): row
        for row in connection.execute("PRAGMA table_info(conversations)").fetchall()
    }


def _minimal_v41_partial_database(
    column_definition: str,
    *,
    leading_column: str | None = None,
    table_constraint: str | None = None,
) -> tuple[CharactersRAGDB, sqlite3.Connection]:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    leading_column_sql = f", {leading_column}" if leading_column else ""
    constraint_sql = f", {table_constraint}" if table_constraint else ""
    connection.executescript(
        f"""
        CREATE TABLE db_schema_version(
          schema_name TEXT PRIMARY KEY,
          version INTEGER NOT NULL
        );
        INSERT INTO db_schema_version(schema_name, version)
        VALUES ('{SCHEMA_NAME}', 41);
        CREATE TABLE local_project_context_parent(id TEXT PRIMARY KEY);
        CREATE TABLE conversations(
          id TEXT,
          rating INTEGER{leading_column_sql},
          {COLUMN_NAME} {column_definition}{constraint_sql}
        );
        """
    )
    db = object.__new__(CharactersRAGDB)
    db.db_path_str = ":memory:"
    return db, connection


def _seed_v41_database(path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Create the actual v41 shape even after the canonical schema advances."""
    with monkeypatch.context() as v41_patch:
        v41_patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 41)
        db = CharactersRAGDB(path, client_id="migration-seed")
        conversation_id = db.add_conversation({"title": "preserve me"})
        connection = db.get_connection()
        columns = _conversation_columns(connection)
        if COLUMN_NAME in columns:
            connection.execute(f"ALTER TABLE conversations DROP COLUMN {COLUMN_NAME}")
            connection.commit()
        assert _version(connection) == 41
        assert COLUMN_NAME not in _conversation_columns(connection)
        db.close_connection()
    return str(conversation_id)


def test_v41_to_v42_adds_nullable_local_column(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "chachanotes.db"
    conversation_id = _seed_v41_database(db_path, monkeypatch)

    db = CharactersRAGDB(db_path, client_id="migration-test")
    connection = db.get_connection()
    columns = _conversation_columns(connection)

    assert _version(connection) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
    assert columns[COLUMN_NAME][3] == 0
    row = connection.execute(
        "SELECT title, console_project_context_json FROM conversations WHERE id = ?",
        (conversation_id,),
    ).fetchone()
    assert tuple(row) == ("preserve me", None)
    db.close_connection()


def test_v41_to_v42_recovers_column_present_version_still_41(
    tmp_path, monkeypatch
) -> None:
    db_path = tmp_path / "chachanotes.db"
    conversation_id = _seed_v41_database(db_path, monkeypatch)

    with monkeypatch.context() as v41_patch:
        v41_patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 41)
        db = CharactersRAGDB(db_path, client_id="partial-migration")
        connection = db.get_connection()
        connection.execute(
            "ALTER TABLE conversations ADD COLUMN console_project_context_json TEXT"
        )
        connection.execute(
            "UPDATE conversations SET console_project_context_json = ? WHERE id = ?",
            ('{"version":1}', conversation_id),
        )
        connection.commit()
        assert _version(connection) == 41
        db.close_connection()

    db = CharactersRAGDB(db_path, client_id="migration-test")
    connection = db.get_connection()
    assert _version(connection) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
    assert (
        connection.execute(
            "SELECT console_project_context_json FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()[0]
        == '{"version":1}'
    )
    db.close_connection()


def test_v41_to_v42_rejects_wrong_start_version(tmp_path) -> None:
    db = CharactersRAGDB(tmp_path / "fresh.db", client_id="version-test")
    connection = db.get_connection()
    starting_version = _version(connection)

    with pytest.raises(SchemaError, match="requires schema version 41"):
        with db.transaction():
            db._migrate_from_v41_to_v42(connection)

    assert _version(connection) == starting_version
    db.close_connection()


@pytest.mark.parametrize("declared_default", ["DEFAULT 'hostile'", "DEFAULT NULL"])
def test_v41_to_v42_rejects_any_declared_column_default(
    declared_default: str,
) -> None:
    db, connection = _minimal_v41_partial_database(f"TEXT {declared_default}")
    before = _conversation_columns(connection)[COLUMN_NAME]
    assert before[4] is not None

    connection.execute("BEGIN")
    with pytest.raises(SchemaError, match="incompatible shape"):
        db._migrate_from_v41_to_v42(connection)
    connection.rollback()

    assert _version(connection) == 41
    assert _conversation_columns(connection)[COLUMN_NAME] == before
    connection.close()


def test_v41_to_v42_rejects_primary_key_column_shape() -> None:
    db, connection = _minimal_v41_partial_database("TEXT PRIMARY KEY")
    before = _conversation_columns(connection)[COLUMN_NAME]
    assert before[5] != 0

    connection.execute("BEGIN")
    with pytest.raises(SchemaError, match="incompatible shape"):
        db._migrate_from_v41_to_v42(connection)
    connection.rollback()

    assert _version(connection) == 41
    assert _conversation_columns(connection)[COLUMN_NAME] == before
    connection.close()


@pytest.mark.parametrize(
    "column_definition",
    [
        "TEXT UNIQUE",
        "TEXT CHECK (console_project_context_json IS NULL)",
        "TEXT REFERENCES local_project_context_parent(id)",
    ],
)
def test_v41_to_v42_rejects_additional_column_constraints(
    column_definition: str,
) -> None:
    db, connection = _minimal_v41_partial_database(column_definition)

    connection.execute("BEGIN")
    with pytest.raises(SchemaError, match="incompatible shape"):
        db._migrate_from_v41_to_v42(connection)
    connection.rollback()

    assert _version(connection) == 41
    connection.close()


@pytest.mark.parametrize(
    "index_expression",
    ["console_project_context_json", "lower(console_project_context_json)"],
)
def test_v41_to_v42_rejects_separate_unique_index_on_local_column(
    index_expression: str,
) -> None:
    db, connection = _minimal_v41_partial_database("TEXT")
    connection.execute(
        "CREATE UNIQUE INDEX hostile_project_context_unique "
        f"ON conversations({index_expression})"
    )
    connection.commit()

    connection.execute("BEGIN")
    with pytest.raises(SchemaError, match="incompatible shape"):
        db._migrate_from_v41_to_v42(connection)
    connection.rollback()

    assert _version(connection) == 41
    assert connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'index' "
        "AND name = 'hostile_project_context_unique'"
    ).fetchone()
    connection.close()


@pytest.mark.parametrize(
    "table_constraint",
    [
        "CHECK(console_project_context_json IS NULL)",
        "CONSTRAINT project_context_guard CHECK(console_project_context_json IS NULL)",
        "CHECK(coalesce(length(trim(console_project_context_json)), "
        "(1 + (2 * 3))) >= 0)",
    ],
)
def test_v41_to_v42_rejects_table_checks_referencing_local_column(
    table_constraint: str,
) -> None:
    db, connection = _minimal_v41_partial_database(
        "TEXT", table_constraint=table_constraint
    )

    connection.execute("BEGIN")
    with pytest.raises(SchemaError, match="incompatible shape"):
        db._migrate_from_v41_to_v42(connection)
    connection.rollback()

    assert _version(connection) == 41
    connection.close()


@pytest.mark.parametrize(
    "unrelated_check",
    [
        "CHECK(rating BETWEEN 1 AND 5)",
        "CHECK('console_project_context_json' <> '')",
        "CHECK(rating > 0 /* console_project_context_json */)",
        "CHECK(rating > 0 -- console_project_context_json\n)",
    ],
)
def test_v41_to_v42_accepts_unrelated_checks_and_target_text_in_non_code(
    unrelated_check: str,
) -> None:
    db, connection = _minimal_v41_partial_database(
        "TEXT", table_constraint=unrelated_check
    )

    connection.execute("BEGIN")
    db._migrate_from_v41_to_v42(connection)
    connection.commit()

    assert _version(connection) == 42
    connection.close()


@pytest.mark.parametrize(
    "leading_column",
    [
        '"quoted--identifier" INTEGER',
        '"quoted/*identifier*/" INTEGER',
        '"quoted""--identifier" INTEGER',
        "`quoted``/*identifier*/` INTEGER",
    ],
)
def test_v41_to_v42_rejects_hostile_check_after_quoted_comment_tokens(
    leading_column: str,
) -> None:
    db, connection = _minimal_v41_partial_database(
        "TEXT",
        leading_column=leading_column,
        table_constraint="CHECK(console_project_context_json IS NULL)",
    )

    connection.execute("BEGIN")
    with pytest.raises(SchemaError, match="incompatible shape"):
        db._migrate_from_v41_to_v42(connection)
    connection.rollback()

    assert _version(connection) == 41
    connection.close()


@pytest.mark.parametrize(
    ("leading_column", "unrelated_check"),
    [
        (
            '"prefix_console_project_context_json_suffix" INTEGER',
            'CHECK("prefix_console_project_context_json_suffix" IS NULL)',
        ),
        (
            "`prefix_console_project_context_json_suffix` INTEGER",
            "CHECK(`prefix_console_project_context_json_suffix` IS NULL)",
        ),
        (
            "[prefix_console_project_context_json_suffix] INTEGER",
            "CHECK([prefix_console_project_context_json_suffix] IS NULL)",
        ),
        (
            "prefix_console_project_context_json_suffix INTEGER",
            "CHECK(prefix_console_project_context_json_suffix IS NULL)",
        ),
    ],
)
def test_v41_to_v42_accepts_non_exact_quoted_and_bare_identifiers(
    leading_column: str,
    unrelated_check: str,
) -> None:
    db, connection = _minimal_v41_partial_database(
        "TEXT",
        leading_column=leading_column,
        table_constraint=unrelated_check,
    )

    connection.execute("BEGIN")
    db._migrate_from_v41_to_v42(connection)
    connection.commit()

    assert _version(connection) == 42
    connection.close()


@pytest.mark.parametrize(
    "hostile_check",
    [
        'CHECK("console_project_context_json" IS NULL)',
        "CHECK(`console_project_context_json` IS NULL)",
        "CHECK([console_project_context_json] IS NULL)",
    ],
)
def test_v41_to_v42_rejects_exact_quoted_target_in_check(
    hostile_check: str,
) -> None:
    db, connection = _minimal_v41_partial_database(
        "TEXT", table_constraint=hostile_check
    )

    connection.execute("BEGIN")
    with pytest.raises(SchemaError, match="incompatible shape"):
        db._migrate_from_v41_to_v42(connection)
    connection.rollback()

    assert _version(connection) == 41
    connection.close()


def test_v41_to_v42_accepts_unicode_confusable_identifier_in_unrelated_check() -> None:
    db, connection = _minimal_v41_partial_database(
        "TEXT",
        leading_column="conſole_project_context_json INTEGER",
        table_constraint="CHECK(conſole_project_context_json IS NULL)",
    )

    connection.execute("BEGIN")
    db._migrate_from_v41_to_v42(connection)
    connection.commit()

    assert _version(connection) == 42
    connection.close()


def test_v41_to_v42_rejects_ascii_case_variant_exact_target_in_check() -> None:
    db, connection = _minimal_v41_partial_database(
        "TEXT",
        table_constraint="CHECK(CONSOLE_PROJECT_CONTEXT_JSON IS NULL)",
    )

    connection.execute("BEGIN")
    with pytest.raises(SchemaError, match="incompatible shape"):
        db._migrate_from_v41_to_v42(connection)
    connection.rollback()

    assert _version(connection) == 41
    connection.close()


def test_v41_to_v42_error_rolls_back_and_leaves_version_41(
    tmp_path, monkeypatch
) -> None:
    db_path = tmp_path / "broken.db"
    _seed_v41_database(db_path, monkeypatch)

    with monkeypatch.context() as v41_patch:
        v41_patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 41)
        db = CharactersRAGDB(db_path, client_id="broken-migration")
        connection = db.get_connection()
        connection.execute(
            """
            CREATE TRIGGER block_v42_version_update
            BEFORE UPDATE OF version ON db_schema_version
            WHEN OLD.schema_name = 'rag_char_chat_schema'
              AND OLD.version = 41
              AND NEW.version = 42
            BEGIN
                SELECT RAISE(ABORT, 'blocked version update');
            END
            """
        )
        connection.commit()
        db.close_connection()

    with pytest.raises(CharactersRAGDBError, match="Migration from V41 to V42 failed"):
        CharactersRAGDB(db_path, client_id="failed-migration")

    with sqlite3.connect(db_path) as connection:
        assert _version(connection) == 41
        assert COLUMN_NAME not in _conversation_columns(connection)


def test_fresh_schema_contains_console_project_context_column(tmp_path) -> None:
    db = CharactersRAGDB(tmp_path / "fresh.db", client_id="fresh-test")
    connection = db.get_connection()
    columns = _conversation_columns(connection)

    # M8: deliberately `>= 42`, not `== 42` -- this file only owns the
    # v41->v42 console_project_context migration, and a LATER, unrelated
    # migration (e.g. task-18300's own v42->v43 message_exchanges table)
    # legitimately bumps `_CURRENT_SCHEMA_VERSION` further without this
    # file needing to know or care. The exact current-version pin lives in
    # exactly one place -- as of task-19554 that is
    # `Tests/DB/test_chachanotes_sync_conflict_preservation_migration.py`'s
    # `test_schema_version_is_44` (the pin moves to the newest migration's
    # own file on every bump, rather than staying on an older one from
    # which it can only drift) -- this assertion only
    # needs to confirm a fresh schema landed AT OR PAST this migration's
    # own version, not the overall latest.
    assert CharactersRAGDB._CURRENT_SCHEMA_VERSION >= 42
    assert _version(connection) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
    assert columns[COLUMN_NAME][2].upper() == "TEXT"
    assert columns[COLUMN_NAME][3] == 0
    db.close_connection()
