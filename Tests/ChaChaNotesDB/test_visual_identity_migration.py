"""ChaChaNotes V38 -> V39 Visual Identity schema migration coverage."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, SchemaError


VISUAL_IDENTITY_TABLES = {
    "visual_identity_packs",
    "visual_identity_pack_versions",
    "visual_identity_assets",
    "visual_identity_bindings",
}

EXPECTED_COLUMNS = {
    "visual_identity_packs": {
        "id": ("INTEGER", 0, None, 1),
        "owner_user_id": ("INTEGER", 1, None, 0),
        "title": ("TEXT", 1, None, 0),
        "description": ("TEXT", 1, "''", 0),
        "status": ("TEXT", 1, "'active'", 0),
        "active_version_id": ("INTEGER", 0, None, 0),
        "default_expression_key": ("TEXT", 1, "'neutral'", 0),
        "source_kind": ("TEXT", 1, "'manual'", 0),
        "source_context_json": ("TEXT", 1, "'{}'", 0),
        "created_at": ("TEXT", 1, "CURRENT_TIMESTAMP", 0),
        "updated_at": ("TEXT", 1, "CURRENT_TIMESTAMP", 0),
        "version": ("INTEGER", 1, "1", 0),
    },
    "visual_identity_pack_versions": {
        "id": ("INTEGER", 0, None, 1),
        "pack_id": ("INTEGER", 1, None, 0),
        "owner_user_id": ("INTEGER", 1, None, 0),
        "version_number": ("INTEGER", 1, None, 0),
        "default_expression_key": ("TEXT", 1, "'neutral'", 0),
        "manifest_json": ("TEXT", 1, None, 0),
        "created_at": ("TEXT", 1, "CURRENT_TIMESTAMP", 0),
    },
    "visual_identity_assets": {
        "id": ("INTEGER", 0, None, 1),
        "owner_user_id": ("INTEGER", 1, None, 0),
        "pack_id": ("INTEGER", 0, None, 0),
        "pack_version_id": ("INTEGER", 1, None, 0),
        "expression_key": ("TEXT", 1, None, 0),
        "original_expression_key": ("TEXT", 1, "''", 0),
        "display_label": ("TEXT", 1, "''", 0),
        "source_filename": ("TEXT", 1, None, 0),
        "storage_relpath": ("TEXT", 1, None, 0),
        "content_type": ("TEXT", 1, None, 0),
        "bytes": ("INTEGER", 1, None, 0),
        "sha256": ("TEXT", 1, None, 0),
        "width": ("INTEGER", 1, None, 0),
        "height": ("INTEGER", 1, None, 0),
        "source_context_json": ("TEXT", 1, "'{}'", 0),
        "is_animated": ("INTEGER", 1, "0", 0),
        "frame_count": ("INTEGER", 0, None, 0),
        "duration_ms": ("INTEGER", 0, None, 0),
        "preview_relpath": ("TEXT", 0, None, 0),
        "deleted": ("INTEGER", 1, "0", 0),
        "created_at": ("TEXT", 1, "CURRENT_TIMESTAMP", 0),
        "updated_at": ("TEXT", 1, "CURRENT_TIMESTAMP", 0),
    },
    "visual_identity_bindings": {
        "id": ("INTEGER", 0, None, 1),
        "owner_user_id": ("INTEGER", 1, None, 0),
        "actor_kind": ("TEXT", 1, None, 0),
        "actor_id": ("TEXT", 1, None, 0),
        "pack_id": ("INTEGER", 1, None, 0),
        "active_version_id": ("INTEGER", 1, None, 0),
        "status": ("TEXT", 1, "'active'", 0),
        "created_at": ("TEXT", 1, "CURRENT_TIMESTAMP", 0),
        "updated_at": ("TEXT", 1, "CURRENT_TIMESTAMP", 0),
        "version": ("INTEGER", 1, "1", 0),
    },
}

EXPECTED_FOREIGN_KEYS = {
    "visual_identity_packs": {
        ("active_version_id", "visual_identity_pack_versions", "id")
    },
    "visual_identity_pack_versions": {("pack_id", "visual_identity_packs", "id")},
    "visual_identity_assets": {
        ("pack_id", "visual_identity_packs", "id"),
        ("pack_version_id", "visual_identity_pack_versions", "id"),
    },
    "visual_identity_bindings": {
        ("pack_id", "visual_identity_packs", "id"),
        ("active_version_id", "visual_identity_pack_versions", "id"),
    },
}

EXPECTED_INDEXES = {
    "idx_visual_identity_packs_owner_status": (
        "visual_identity_packs",
        False,
        ("owner_user_id", "status"),
        False,
    ),
    "idx_visual_identity_assets_pack_expression": (
        "visual_identity_assets",
        False,
        ("pack_id", "pack_version_id", "expression_key", "deleted"),
        False,
    ),
    "idx_visual_identity_bindings_actor_active": (
        "visual_identity_bindings",
        True,
        ("owner_user_id", "actor_kind", "actor_id"),
        True,
    ),
}


def _version(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (CharactersRAGDB._SCHEMA_NAME,),
    ).fetchone()
    assert row is not None
    return int(row[0])


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }


def _table_sql(connection: sqlite3.Connection, table: str) -> str:
    row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone()
    assert row is not None
    return str(row[0])


def _seed_v38_database(
    path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[set[str], str]:
    with monkeypatch.context() as v38:
        v38.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 38)
        seeded = CharactersRAGDB(path, client_id="v38-seed")
        connection = seeded.get_connection()
        assert _version(connection) == 38
        tables = _tables(connection)
        assert not (tables & VISUAL_IDENTITY_TABLES)
        expression_table = _table_sql(connection, "character_expression_images")
        seeded.close_connection()
    return tables, expression_table


def _assert_schema_contract(connection: sqlite3.Connection) -> None:
    visual_tables = {
        name for name in _tables(connection) if name.startswith("visual_identity_")
    }
    assert visual_tables == VISUAL_IDENTITY_TABLES
    assert _version(connection) == CharactersRAGDB._CURRENT_SCHEMA_VERSION

    for table, expected in EXPECTED_COLUMNS.items():
        actual = {
            str(row[1]): (str(row[2]), int(row[3]), row[4], int(row[5]))
            for row in connection.execute(f"PRAGMA table_info('{table}')")
        }
        assert actual == expected

        foreign_keys = {
            (str(row[3]), str(row[2]), str(row[4]))
            for row in connection.execute(f"PRAGMA foreign_key_list('{table}')")
        }
        assert foreign_keys == EXPECTED_FOREIGN_KEYS[table]

    assert "draft_id" not in EXPECTED_COLUMNS["visual_identity_assets"]

    actual_indexes = {}
    for table in VISUAL_IDENTITY_TABLES:
        for row in connection.execute(f"PRAGMA index_list('{table}')"):
            name = str(row[1])
            if name.startswith("sqlite_autoindex_"):
                continue
            columns = tuple(
                str(column[2])
                for column in connection.execute(f"PRAGMA index_info('{name}')")
            )
            actual_indexes[name] = (table, bool(row[2]), columns, bool(row[4]))
    assert actual_indexes == EXPECTED_INDEXES

    normalized_sql = {
        table: "".join(_table_sql(connection, table).split()).lower()
        for table in VISUAL_IDENTITY_TABLES
    }
    assert (
        "check(statusin('active','archived','deleted'))"
        in normalized_sql["visual_identity_packs"]
    )
    assert (
        "unique(pack_id,version_number)"
        in normalized_sql["visual_identity_pack_versions"]
    )
    for check in (
        "check(bytes>0)",
        "check(width>0)",
        "check(height>0)",
        "check(is_animatedin(0,1))",
        "check(deletedin(0,1))",
    ):
        assert check in normalized_sql["visual_identity_assets"]
    assert (
        "check(actor_kindin('character','persona'))"
        in normalized_sql["visual_identity_bindings"]
    )
    assert (
        "check(statusin('active','deleted'))"
        in normalized_sql["visual_identity_bindings"]
    )


def _assert_required_version_and_active_binding_uniqueness(
    connection: sqlite3.Connection,
) -> None:
    pack_id = connection.execute(
        "INSERT INTO visual_identity_packs(owner_user_id, title) VALUES (0, 'Pack')"
    ).lastrowid
    version_id = connection.execute(
        """
        INSERT INTO visual_identity_pack_versions(
            pack_id, owner_user_id, version_number, manifest_json
        ) VALUES (?, 0, 1, '{}')
        """,
        (pack_id,),
    ).lastrowid
    connection.execute(
        "UPDATE visual_identity_packs SET active_version_id = ? WHERE id = ?",
        (version_id, pack_id),
    )

    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """
            INSERT INTO visual_identity_assets(
                owner_user_id, pack_id, pack_version_id, expression_key,
                source_filename, storage_relpath, content_type, bytes, sha256,
                width, height
            ) VALUES (0, ?, NULL, 'neutral', 'neutral.webp',
                      'characters/test/neutral.webp', 'image/webp', 1, 'abc', 1, 1)
            """,
            (pack_id,),
        )

    binding = (0, "character", "42", pack_id, version_id)
    connection.execute(
        """
        INSERT INTO visual_identity_bindings(
            owner_user_id, actor_kind, actor_id, pack_id, active_version_id
        ) VALUES (?, ?, ?, ?, ?)
        """,
        binding,
    )
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """
            INSERT INTO visual_identity_bindings(
                owner_user_id, actor_kind, actor_id, pack_id, active_version_id
            ) VALUES (?, ?, ?, ?, ?)
            """,
            binding,
        )

    deleted_binding = (*binding, "deleted")
    connection.executemany(
        """
        INSERT INTO visual_identity_bindings(
            owner_user_id, actor_kind, actor_id, pack_id, active_version_id, status
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (deleted_binding, deleted_binding),
    )
    deleted_count = connection.execute(
        """
        SELECT COUNT(*)
          FROM visual_identity_bindings
         WHERE owner_user_id = 0
           AND actor_kind = 'character'
           AND actor_id = '42'
           AND status = 'deleted'
        """
    ).fetchone()[0]
    assert deleted_count == 2


@pytest.mark.parametrize("construction", ["upgrade_v38", "fresh"])
def test_visual_identity_schema_is_installed_by_migration_and_fresh_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    construction: str,
) -> None:
    path = tmp_path / f"{construction}.db"
    tables_before: set[str] | None = None
    expression_table_before: str | None = None

    if construction == "upgrade_v38":
        tables_before, expression_table_before = _seed_v38_database(path, monkeypatch)

    db = CharactersRAGDB(path, client_id="v39-open")
    try:
        connection = db.get_connection()
        _assert_schema_contract(connection)
        _assert_required_version_and_active_binding_uniqueness(connection)

        if tables_before is not None:
            # Superset, not equality: upgrading from v38 runs EVERY later
            # migration, so tables added after v39 (e.g. v40's
            # transcript_annotations) legitimately appear in the delta.
            assert VISUAL_IDENTITY_TABLES <= _tables(connection) - tables_before
            assert (
                _table_sql(connection, "character_expression_images")
                == expression_table_before
            )
    finally:
        db.close_connection()


def test_v38_to_v39_failure_rolls_back_all_visual_identity_tables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "rollback.db"
    _seed_v38_database(path, monkeypatch)
    migration_path = (
        Path(__file__).parents[2]
        / "tldw_chatbook"
        / "DB"
        / "migrations"
        / "chachanotes_v38_to_v39_visual_identity.sql"
    )
    original_read_text = Path.read_text

    def read_text_with_invalid_v39_sql(path_to_read: Path, *args, **kwargs) -> str:
        source = original_read_text(path_to_read, *args, **kwargs)
        if path_to_read != migration_path:
            return source
        next_table = "\n\nCREATE TABLE visual_identity_pack_versions"
        assert next_table in source
        return source.replace(next_table, f"\n\nINVALID SQL;{next_table}", 1)

    monkeypatch.setattr(Path, "read_text", read_text_with_invalid_v39_sql)

    with pytest.raises(SchemaError, match=r"V38.*V39"):
        CharactersRAGDB(path, client_id="failed-v39-open")

    with sqlite3.connect(path) as connection:
        version = connection.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).fetchone()[0]
        tables = _tables(connection)

    assert version == 38
    assert not (VISUAL_IDENTITY_TABLES & tables)
