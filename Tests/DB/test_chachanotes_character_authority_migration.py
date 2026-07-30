import inspect
import json
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    InputError,
)


SCHEMA_NAME = "rag_char_chat_schema"
SERVER_AUTHORITY = f"server-user-v1:{'a' * 64}"


def _version(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (SCHEMA_NAME,),
    ).fetchone()
    assert row is not None
    return int(row[0])


def _conversation_columns(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(conversations)").fetchall()
    }


def _seed_v27_database(
    path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[set[str], str]:
    """Create a real v27 database without calling v28 conversation CRUD."""

    with monkeypatch.context() as v27_patch:
        v27_patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 27)
        db = CharactersRAGDB(path, client_id="migration-seed")
        connection = db.get_connection()
        before_columns = _conversation_columns(connection)
        local_authority_id = connection.execute(
            """
            SELECT local_authority_id
            FROM rag_identity_context
            WHERE context_name = 'default'
            """
        ).fetchone()[0]
        rows = (
            ("local-proven", "local", "character", "1", 1),
            ("local-noncanonical", "local", "character", "01", 1),
            ("server-legacy", "server", "character", "server/opaque:A-7", 1),
            ("persona-legacy", "local", "persona", "persona-7", 1),
            ("generic-legacy", "local", "generic", "console", 1),
        )
        with db.transaction() as cursor:
            for conversation_id, source, kind, assistant_id, character_id in rows:
                cursor.execute(
                    """
                    INSERT INTO conversations(
                        id,
                        root_id,
                        character_id,
                        assistant_kind,
                        assistant_id,
                        runtime_backend,
                        title,
                        created_at,
                        last_modified,
                        deleted,
                        client_id,
                        version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP,
                              CURRENT_TIMESTAMP, 0, 'migration-seed', 1)
                    """,
                    (
                        conversation_id,
                        conversation_id,
                        character_id,
                        kind,
                        assistant_id,
                        source,
                        conversation_id,
                    ),
                )
            cursor.execute("DELETE FROM sync_log")
        db.close_connection()
    return before_columns, str(local_authority_id)


def _identity_only_v28_database(
    path: Path,
    rows: tuple[tuple[str, object], ...],
) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            f"""
            CREATE TABLE db_schema_version(
                schema_name TEXT PRIMARY KEY NOT NULL,
                version INTEGER NOT NULL
            );
            INSERT INTO db_schema_version VALUES ('{SCHEMA_NAME}', 28);
            CREATE TABLE rag_identity_context(
                context_name TEXT,
                local_authority_id
            );
            """
        )
        connection.executemany(
            """
            INSERT INTO rag_identity_context(context_name, local_authority_id)
            VALUES (?, ?)
            """,
            rows,
        )


def test_fresh_database_reaches_v28_and_local_authority_survives_reopen(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fresh-v28.sqlite"
    db = CharactersRAGDB(path, client_id="authority-test")

    assert db._CURRENT_SCHEMA_VERSION == 28
    assert _version(db.get_connection()) == 28
    authority_id = db.get_local_authority_id()
    assert 1 <= len(authority_id.encode("utf-8")) <= 256
    db.close_connection()

    reopened = CharactersRAGDB(path, client_id="authority-test")
    assert reopened.get_local_authority_id() == authority_id
    reopened.close_connection()


def test_local_authority_accessor_uses_shared_transaction_seam(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = CharactersRAGDB(
        tmp_path / "authority-transaction.sqlite",
        client_id="authority-transaction-test",
    )
    real_transaction = db.transaction
    transaction_calls = 0

    @contextmanager
    def recording_transaction():
        nonlocal transaction_calls
        transaction_calls += 1
        with real_transaction() as cursor:
            yield cursor

    monkeypatch.setattr(db, "transaction", recording_transaction)
    try:
        assert db.get_local_authority_id()
        assert transaction_calls == 1
    finally:
        db.close_connection()


def test_local_authority_accessor_documents_public_contract() -> None:
    doc = inspect.getdoc(CharactersRAGDB.get_local_authority_id)

    assert doc is not None
    assert "Returns:" in doc
    assert "Raises:" in doc


@pytest.mark.parametrize(
    "rows",
    [
        (),
        (("default", ""),),
        (("default", " authority-with-whitespace "),),
        (("default", "x" * 257),),
        (("default", "authority-one"), ("default", "authority-two")),
    ],
)
def test_local_authority_accessor_fails_closed_for_unavailable_or_ambiguous_state(
    tmp_path: Path,
    rows: tuple[tuple[str, object], ...],
) -> None:
    path = tmp_path / "invalid-identity.sqlite"
    _identity_only_v28_database(path, rows)
    db = CharactersRAGDB(path, client_id="authority-test")

    with pytest.raises(
        CharactersRAGDBError,
        match=r"^Local authority identity is unavailable or invalid\.$",
    ) as exc_info:
        db.get_local_authority_id()

    assert len(str(exc_info.value).encode("utf-8")) <= 128


def test_v27_migration_adds_only_nullable_authority_and_backfills_proven_local_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "v27-to-v28.sqlite"
    before_columns, expected_authority = _seed_v27_database(path, monkeypatch)

    db = CharactersRAGDB(path, client_id="migration-test")
    connection = db.get_connection()

    assert _version(connection) == 28
    after_columns = _conversation_columns(connection)
    assert after_columns - before_columns == {"assistant_authority_id"}
    authority_column = next(
        row
        for row in connection.execute("PRAGMA table_info(conversations)").fetchall()
        if row[1] == "assistant_authority_id"
    )
    assert authority_column[2] == "TEXT"
    assert authority_column[3] == 0

    actual = dict(
        connection.execute(
            """
            SELECT id, assistant_authority_id
            FROM conversations
            ORDER BY id
            """
        ).fetchall()
    )
    assert actual == {
        "generic-legacy": None,
        "local-noncanonical": None,
        "local-proven": expected_authority,
        "persona-legacy": None,
        "server-legacy": None,
    }
    assert connection.execute("SELECT COUNT(*) FROM sync_log").fetchone()[0] == 0


def test_v27_migration_rolls_back_column_backfill_and_version_on_late_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "v27-rollback.sqlite"
    before_columns, expected_authority = _seed_v27_database(path, monkeypatch)
    original = getattr(
        CharactersRAGDB,
        "_update_character_authority_schema_version",
        None,
    )

    def fail_after_version_update(self, cursor):
        if original is not None:
            original(self, cursor)
        raise sqlite3.OperationalError("forced character authority failure")

    with monkeypatch.context() as failure_patch:
        failure_patch.setattr(
            CharactersRAGDB,
            "_update_character_authority_schema_version",
            fail_after_version_update,
            raising=False,
        )
        with pytest.raises(Exception, match="forced character authority failure"):
            CharactersRAGDB(path, client_id="migration-test")

    with sqlite3.connect(path) as connection:
        assert _version(connection) == 27
        assert _conversation_columns(connection) == before_columns

    migrated = CharactersRAGDB(path, client_id="migration-test")
    row = migrated.get_connection().execute(
        """
        SELECT assistant_authority_id
        FROM conversations
        WHERE id = 'local-proven'
        """
    ).fetchone()
    assert row["assistant_authority_id"] == expected_authority


def test_migration_sql_adds_no_table_index_trigger_or_transaction_owner() -> None:
    sql_path = (
        Path(__file__).parents[2]
        / "tldw_chatbook"
        / "DB"
        / "migrations"
        / "chachanotes_v27_to_v28_character_authority.sql"
    )
    executable = "\n".join(
        line
        for line in sql_path.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("--")
    ).lower()
    normalized_sql = re.sub(r"\s+", " ", executable)

    assert normalized_sql.count("alter table conversations add column") == 1
    assert "assistant_authority_id" in executable
    assert "create table" not in executable
    assert "create index" not in executable
    assert "create trigger" not in executable
    assert "db_schema_version" not in executable
    assert "begin" not in executable
    assert "commit" not in executable


def test_authority_column_enforces_nullable_bounded_nonempty_text(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "authority-check.sqlite", client_id="check")
    conversation_id = db.add_conversation({"title": "Generic"})
    connection = db.get_connection()

    connection.execute(
        """
        UPDATE conversations
        SET assistant_authority_id = NULL
        WHERE id = ?
        """,
        (conversation_id,),
    )
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """
            UPDATE conversations
            SET assistant_authority_id = ''
            WHERE id = ?
            """,
            (conversation_id,),
        )
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """
            UPDATE conversations
            SET assistant_authority_id = ?
            WHERE id = ?
            """,
            ("x" * 257, conversation_id),
        )
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """
            UPDATE conversations
            SET assistant_authority_id = ?
            WHERE id = ?
            """,
            (sqlite3.Binary(b"authority"), conversation_id),
        )


def test_local_character_create_read_and_list_infer_same_database_authority(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "local-roundtrip.sqlite", client_id="crud")
    authority_id = db.get_local_authority_id()

    conversation_id = db.add_conversation(
        {
            "title": "Local Character",
            "character_id": "1",
            "assistant_kind": " CHARACTER ",
            "runtime_backend": " LOCAL ",
        }
    )

    row = db.get_conversation_by_id(conversation_id)
    assert row is not None
    assert row["runtime_backend"] == "local"
    assert row["assistant_kind"] == "character"
    assert row["character_id"] == 1
    assert row["assistant_id"] == "1"
    assert row["assistant_authority_id"] == authority_id
    listed = {
        item["id"]: item for item in db.list_all_active_conversations()
    }[conversation_id]
    assert listed["assistant_authority_id"] == authority_id
    searched = db.search_conversations_page("Local Character")[0][0]
    assert searched["assistant_authority_id"] == authority_id


def test_local_character_explicit_null_stays_unassignable_and_title_update_preserves_it(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "local-null.sqlite", client_id="crud")
    conversation_id = db.add_conversation(
        {
            "title": "Imported Local Character",
            "character_id": 1,
            "assistant_kind": "character",
            "assistant_authority_id": None,
        }
    )
    created = db.get_conversation_by_id(conversation_id)
    assert created["assistant_authority_id"] is None

    assert db.update_conversation(
        conversation_id,
        {"title": "Still Unproven"},
        expected_version=created["version"],
    )
    updated = db.get_conversation_by_id(conversation_id)
    assert updated["assistant_authority_id"] is None


def test_local_character_omitted_authority_survives_noop_runtime_update(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "local-null-runtime.sqlite", client_id="crud")
    conversation_id = db.add_conversation(
        {
            "character_id": 1,
            "assistant_kind": "character",
            "assistant_authority_id": None,
        }
    )
    created = db.get_conversation_by_id(conversation_id)

    assert db.update_conversation(
        conversation_id,
        {"runtime_backend": "local"},
        expected_version=created["version"],
    )
    updated = db.get_conversation_by_id(conversation_id)
    assert updated["assistant_authority_id"] is None


def test_local_character_omitted_authority_survives_redundant_identity_update(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "local-null-identity.sqlite", client_id="crud")
    conversation_id = db.add_conversation(
        {
            "character_id": 1,
            "assistant_kind": "character",
            "assistant_authority_id": None,
        }
    )
    created = db.get_conversation_by_id(conversation_id)

    assert db.update_conversation(
        conversation_id,
        {
            "runtime_backend": " LOCAL ",
            "assistant_kind": " CHARACTER ",
            "character_id": "1",
            "assistant_id": " 1 ",
        },
        expected_version=created["version"],
    )
    updated = db.get_conversation_by_id(conversation_id)
    assert updated["runtime_backend"] == "local"
    assert updated["assistant_kind"] == "character"
    assert updated["character_id"] == 1
    assert updated["assistant_id"] == "1"
    assert updated["assistant_authority_id"] is None


@pytest.mark.parametrize(
    ("source_kind", "source_id"),
    [("generic", "console"), ("persona", "persona-1")],
)
def test_noncharacter_to_local_character_transition_infers_omitted_authority(
    tmp_path: Path,
    source_kind: str,
    source_id: str,
) -> None:
    db = CharactersRAGDB(
        tmp_path / f"{source_kind}-to-local.sqlite",
        client_id="crud",
    )
    conversation_id = db.add_conversation(
        {
            "assistant_kind": source_kind,
            "assistant_id": source_id,
        }
    )
    source = db.get_conversation_by_id(conversation_id)

    assert db.update_conversation(
        conversation_id,
        {
            "runtime_backend": "local",
            "assistant_kind": "character",
            "character_id": 1,
        },
        expected_version=source["version"],
    )
    local = db.get_conversation_by_id(conversation_id)
    assert local["assistant_id"] == "1"
    assert local["assistant_authority_id"] == db.get_local_authority_id()


def test_local_character_identity_change_infers_omitted_authority(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "local-character-change.sqlite", client_id="crud")
    first_character_id = db.add_character_card({"name": "Imported Character"})
    second_character_id = db.add_character_card({"name": "Replacement Character"})
    conversation_id = db.add_conversation(
        {
            "character_id": first_character_id,
            "assistant_kind": "character",
            "assistant_authority_id": None,
        }
    )
    imported = db.get_conversation_by_id(conversation_id)

    assert db.update_conversation(
        conversation_id,
        {"character_id": second_character_id},
        expected_version=imported["version"],
    )
    changed = db.get_conversation_by_id(conversation_id)
    assert changed["character_id"] == second_character_id
    assert changed["assistant_id"] == str(second_character_id)
    assert changed["assistant_authority_id"] == db.get_local_authority_id()


def test_local_character_rejects_noncanonical_identity_or_wrong_authority(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "local-validation.sqlite", client_id="crud")

    with pytest.raises(InputError, match="canonical decimal"):
        db.add_conversation(
            {
                "character_id": 1,
                "assistant_kind": "character",
                "assistant_id": "01",
            }
        )
    with pytest.raises(InputError, match="positive"):
        db.add_conversation(
            {
                "character_id": 0,
                "assistant_kind": "character",
            }
        )
    with pytest.raises(InputError, match="local authority"):
        db.add_conversation(
            {
                "character_id": 1,
                "assistant_kind": "character",
                "assistant_authority_id": "authority-from-another-database",
            }
        )


def test_server_character_accepts_opaque_scoped_or_null_identity_and_clears_character_id(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "server-roundtrip.sqlite", client_id="crud")
    scoped_id = db.add_conversation(
        {
            "title": "Scoped Server",
            "runtime_backend": " SERVER ",
            "assistant_kind": "character",
            "assistant_id": " server/opaque:A-7 ",
            "assistant_authority_id": SERVER_AUTHORITY,
            "character_id": 1,
        }
    )
    null_id = db.add_conversation(
        {
            "title": "Unscoped Server",
            "runtime_backend": "server",
            "assistant_kind": "character",
            "assistant_id": "server-character-9",
            "assistant_authority_id": None,
        }
    )

    scoped = db.get_conversation_by_id(scoped_id)
    assert scoped["runtime_backend"] == "server"
    assert scoped["assistant_id"] == "server/opaque:A-7"
    assert scoped["character_id"] is None
    assert scoped["assistant_authority_id"] == SERVER_AUTHORITY
    assert db.get_conversation_by_id(null_id)["assistant_authority_id"] is None


def test_server_character_rejects_empty_or_overlong_identity_text(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "server-validation.sqlite", client_id="crud")

    with pytest.raises(InputError, match="assistant_id"):
        db.add_conversation(
            {
                "runtime_backend": "server",
                "assistant_kind": "character",
                "assistant_id": " ",
            }
        )
    with pytest.raises(InputError, match="256 UTF-8 bytes"):
        db.add_conversation(
            {
                "runtime_backend": "server",
                "assistant_kind": "character",
                "assistant_id": "x" * 257,
            }
        )
    with pytest.raises(InputError, match="256 UTF-8 bytes"):
        db.add_conversation(
            {
                "runtime_backend": "server",
                "assistant_kind": "character",
                "assistant_id": "opaque",
                "assistant_authority_id": "x" * 257,
            }
        )


@pytest.mark.parametrize("assistant_kind", [None, "generic", "persona"])
def test_persona_and_generic_create_reject_character_authority(
    tmp_path: Path,
    assistant_kind: str | None,
) -> None:
    db = CharactersRAGDB(
        tmp_path / f"authority-free-{assistant_kind}.sqlite",
        client_id="crud",
    )
    data = {
        "assistant_kind": assistant_kind,
        "assistant_authority_id": SERVER_AUTHORITY,
    }
    if assistant_kind in {"generic", "persona"}:
        data["assistant_id"] = f"{assistant_kind}-assistant"

    with pytest.raises(InputError, match="cannot carry character authority"):
        db.add_conversation(data)


@pytest.mark.parametrize(
    ("target_kind", "target_id"),
    [("persona", "persona-1"), ("generic", "console")],
)
def test_identity_and_source_conversions_clear_or_infer_authority_jointly(
    tmp_path: Path,
    target_kind: str,
    target_id: str,
) -> None:
    db = CharactersRAGDB(tmp_path / "conversions.sqlite", client_id="crud")
    local_authority = db.get_local_authority_id()
    conversation_id = db.add_conversation(
        {
            "title": "Source Conversion",
            "character_id": 1,
            "assistant_kind": "character",
        }
    )
    local = db.get_conversation_by_id(conversation_id)

    assert db.update_conversation(
        conversation_id,
        {"runtime_backend": "server"},
        expected_version=local["version"],
    )
    unscoped_server = db.get_conversation_by_id(conversation_id)
    assert unscoped_server["runtime_backend"] == "server"
    assert unscoped_server["assistant_id"] == "1"
    assert unscoped_server["character_id"] is None
    assert unscoped_server["assistant_authority_id"] is None

    assert db.update_conversation(
        conversation_id,
        {
            "runtime_backend": "local",
            "character_id": 1,
        },
        expected_version=unscoped_server["version"],
    )
    local_again = db.get_conversation_by_id(conversation_id)
    assert local_again["assistant_id"] == "1"
    assert local_again["assistant_authority_id"] == local_authority

    assert db.update_conversation(
        conversation_id,
        {
            "assistant_kind": target_kind,
            "assistant_id": target_id,
        },
        expected_version=local_again["version"],
    )
    authority_free = db.get_conversation_by_id(conversation_id)
    assert authority_free["assistant_kind"] == target_kind
    assert authority_free["character_id"] is None
    assert authority_free["assistant_authority_id"] is None


def test_explicit_local_null_update_is_preserved_and_wrong_authority_is_rejected(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "authority-update.sqlite", client_id="crud")
    conversation_id = db.add_conversation(
        {"character_id": 1, "assistant_kind": "character"}
    )
    created = db.get_conversation_by_id(conversation_id)

    assert db.update_conversation(
        conversation_id,
        {"assistant_authority_id": None},
        expected_version=created["version"],
    )
    unproven = db.get_conversation_by_id(conversation_id)
    assert unproven["assistant_authority_id"] is None

    with pytest.raises(InputError, match="local authority"):
        db.update_conversation(
            conversation_id,
            {"assistant_authority_id": "wrong-authority"},
            expected_version=unproven["version"],
        )
    assert db.get_conversation_by_id(conversation_id)["version"] == unproven["version"]


def test_unrelated_update_does_not_repair_degraded_legacy_identity(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "legacy-readable.sqlite", client_id="crud")
    conversation_id = db.add_conversation({"title": "Legacy"})
    with db.transaction() as cursor:
        cursor.execute(
            """
            UPDATE conversations
            SET runtime_backend = 'local',
                assistant_kind = 'character',
                assistant_id = 'legacy-display-name',
                character_id = 1,
                assistant_authority_id = NULL
            WHERE id = ?
            """,
            (conversation_id,),
        )
    legacy = db.get_conversation_by_id(conversation_id)

    assert db.update_conversation(
        conversation_id,
        {"title": "Legacy Renamed"},
        expected_version=legacy["version"],
    )
    renamed = db.get_conversation_by_id(conversation_id)
    assert renamed["assistant_id"] == "legacy-display-name"
    assert renamed["character_id"] == 1
    assert renamed["assistant_authority_id"] is None


def test_sync_trigger_payloads_never_carry_assistant_authority(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "sync-exclusion.sqlite", client_id="crud")
    conversation_id = db.add_conversation(
        {"character_id": 1, "assistant_kind": "character"}
    )
    created = db.get_conversation_by_id(conversation_id)
    db.update_conversation(
        conversation_id,
        {"assistant_authority_id": None},
        expected_version=created["version"],
    )

    connection = db.get_connection()
    trigger_sql = connection.execute(
        """
        SELECT group_concat(sql, ' ')
        FROM sqlite_master
        WHERE type = 'trigger'
          AND name LIKE 'conversations_sync_%'
        """
    ).fetchone()[0]
    assert "assistant_authority_id" not in trigger_sql
    payloads = [
        json.loads(row[0])
        for row in connection.execute(
            """
            SELECT payload
            FROM sync_log
            WHERE entity = 'conversations'
              AND entity_id = ?
            """,
            (conversation_id,),
        ).fetchall()
    ]
    assert payloads
    assert all("assistant_authority_id" not in payload for payload in payloads)
