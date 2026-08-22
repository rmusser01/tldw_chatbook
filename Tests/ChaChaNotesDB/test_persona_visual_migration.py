"""ChaChaNotes V40 -> V41 Persona Visual schema migration coverage."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, SchemaError


PERSONA_VISUAL_TABLES = {
    "persona_visual_packs",
    "persona_visual_pack_versions",
    "persona_visual_assets",
    "persona_visual_bindings",
}

EXPECTED_COLUMNS = {
    "persona_visual_packs": (
        "id",
        "title",
        "description",
        "status",
        "active_version_id",
        "source_kind",
        "source_context_json",
        "created_at",
        "updated_at",
        "version",
    ),
    "persona_visual_pack_versions": (
        "id",
        "pack_id",
        "version_number",
        "renderer_type",
        "manifest_version",
        "manifest_json",
        "manifest_sha256",
        "storage_relpath",
        "created_at",
    ),
    "persona_visual_assets": (
        "id",
        "pack_id",
        "pack_version_id",
        "asset_key",
        "role",
        "storage_relpath",
        "mime_type",
        "bytes",
        "sha256",
        "width",
        "height",
        "frame_count",
        "duration_ms",
        "created_at",
    ),
    "persona_visual_bindings": (
        "id",
        "persona_id",
        "persona_revision",
        "pack_id",
        "active_version_id",
        "status",
        "created_at",
        "updated_at",
        "version",
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


def _seed_v40(path: Path) -> tuple[dict[str, str], tuple[object, ...]]:
    with chachanotes_db_at_version(path, 40, client_id="persona-visual-v40") as db:
        connection = db.get_connection()
        assert _version(connection) == 40
        assert not (_tables(connection) & PERSONA_VISUAL_TABLES)
        shared_schema = {
            table: _table_sql(connection, table)
            for table in (
                "visual_identity_packs",
                "visual_identity_pack_versions",
                "visual_identity_assets",
                "visual_identity_bindings",
            )
        }
        pack_id = connection.execute(
            """
            INSERT INTO visual_identity_packs(owner_user_id, title)
            VALUES (0, 'Existing shared visual pack')
            """
        ).lastrowid
        connection.commit()
        shared_row = tuple(
            connection.execute(
                "SELECT * FROM visual_identity_packs WHERE id = ?", (pack_id,)
            ).fetchone()
        )
    return shared_schema, shared_row


def _insert_pack_version(connection: sqlite3.Connection, title: str) -> tuple[int, int]:
    pack_id = int(
        connection.execute(
            "INSERT INTO persona_visual_packs(title) VALUES (?)", (title,)
        ).lastrowid
    )
    version_id = int(
        connection.execute(
            """
            INSERT INTO persona_visual_pack_versions(
                pack_id, version_number, renderer_type, manifest_version,
                manifest_json, manifest_sha256, storage_relpath
            ) VALUES (?, 1, 'sprite_frames', 1, '{}', ?, ?)
            """,
            (pack_id, "a" * 64, f"persona_visual/{pack_id}/v1/manifest.json"),
        ).lastrowid
    )
    connection.execute(
        "UPDATE persona_visual_packs SET active_version_id = ? WHERE id = ?",
        (version_id, pack_id),
    )
    return pack_id, version_id


def test_real_v40_upgrade_installs_separate_persona_visual_schema(
    tmp_path: Path,
) -> None:
    """Applying just the v40->v41 step installs the persona-visual schema.

    task-19568: this used to open the path as a plain, unpatched
    ``CharactersRAGDB`` and gate the whole body behind
    ``assert CharactersRAGDB._CURRENT_SCHEMA_VERSION == 41`` as its FIRST
    statement -- so once a later migration bumped the global version past
    41 (persona-visual is no longer the newest migration), that assertion
    failed immediately and the migration exercise, the table census, the
    five ``EXPECTED_COLUMNS`` checks, the index assertions, and the
    foreign-key matrix below it never ran at all. Using
    ``chachanotes_db_at_version(path, 41, ...)`` instead exercises the
    v40->v41 step in isolation -- the production chain still stops exactly
    at 41 regardless of how high ``_CURRENT_SCHEMA_VERSION`` has since
    climbed, so the test asserts the behaviour of *its own* migration, not
    the global current version, and there is no upfront version equality
    to short-circuit the rest of the body.
    """
    path = tmp_path / "persona-visual-v40.db"
    persona_json = tmp_path / "personas.json"
    persona_bytes = b'{"personas":[{"id":"local-1","revision":7}]}'
    persona_json.write_bytes(persona_bytes)
    shared_schema, shared_row = _seed_v40(path)

    with chachanotes_db_at_version(path, 41, client_id="persona-visual-v41") as db:
        connection = db.get_connection()
        assert _version(connection) == 41
        assert PERSONA_VISUAL_TABLES <= _tables(connection)
        assert persona_json.read_bytes() == persona_bytes

        for table, expected_columns in EXPECTED_COLUMNS.items():
            columns = tuple(
                str(row[1])
                for row in connection.execute(f"PRAGMA table_info('{table}')")
            )
            assert columns == expected_columns

        indexes = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
        assert "idx_persona_visual_bindings_persona_active" in indexes
        assert "idx_persona_visual_assets_version_key" in indexes

        expected_foreign_keys = {
            "persona_visual_packs": {
                ("id", "pack_id", "persona_visual_pack_versions"),
                ("active_version_id", "id", "persona_visual_pack_versions"),
            },
            "persona_visual_pack_versions": {("pack_id", "id", "persona_visual_packs")},
            "persona_visual_assets": {
                ("pack_id", "id", "persona_visual_packs"),
                ("pack_id", "pack_id", "persona_visual_pack_versions"),
                ("pack_version_id", "id", "persona_visual_pack_versions"),
            },
            "persona_visual_bindings": {
                ("pack_id", "id", "persona_visual_packs"),
                ("pack_id", "pack_id", "persona_visual_pack_versions"),
                ("active_version_id", "id", "persona_visual_pack_versions"),
            },
        }
        for table, expected in expected_foreign_keys.items():
            actual = {
                (str(row[3]), str(row[4]), str(row[2]))
                for row in connection.execute(f"PRAGMA foreign_key_list('{table}')")
            }
            assert actual == expected

        for table, sql in shared_schema.items():
            assert _table_sql(connection, table) == sql
        assert (
            tuple(
                connection.execute(
                    "SELECT * FROM visual_identity_packs WHERE title = ?",
                    ("Existing shared visual pack",),
                ).fetchone()
            )
            == shared_row
        )


def test_persona_visual_schema_enforces_immutable_graph_relationships(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "constraints.db", client_id="constraints")
    try:
        connection = db.get_connection()
        first_pack, first_version = _insert_pack_version(connection, "First")
        second_pack, second_version = _insert_pack_version(connection, "Second")

        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO persona_visual_pack_versions(
                    pack_id, version_number, renderer_type, manifest_version,
                    manifest_json, manifest_sha256, storage_relpath
                ) VALUES (?, 1, 'sprite_frames', 1, '{}', ?, 'duplicate')
                """,
                (first_pack, "b" * 64),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE persona_visual_packs SET active_version_id = ? WHERE id = ?",
                (second_version, first_pack),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO persona_visual_assets(
                    pack_id, pack_version_id, asset_key, role, storage_relpath,
                    mime_type, bytes, sha256, width, height
                ) VALUES (?, ?, 'idle', 'sprite', 'asset.png',
                          'image/png', 1, ?, 1, 1)
                """,
                (first_pack, second_version, "c" * 64),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO persona_visual_bindings(
                    persona_id, persona_revision, pack_id, active_version_id
                ) VALUES ('persona-cross-pack', 1, ?, ?)
                """,
                (first_pack, second_version),
            )

        binding = ("persona-1", 7, first_pack, first_version)
        connection.execute(
            """
            INSERT INTO persona_visual_bindings(
                persona_id, persona_revision, pack_id, active_version_id
            ) VALUES (?, ?, ?, ?)
            """,
            binding,
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO persona_visual_bindings(
                    persona_id, persona_revision, pack_id, active_version_id
                ) VALUES (?, ?, ?, ?)
                """,
                binding,
            )
        connection.executemany(
            """
            INSERT INTO persona_visual_bindings(
                persona_id, persona_revision, pack_id, active_version_id, status
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                (*binding, "archived"),
                (*binding, "deleted"),
            ),
        )
    finally:
        db.close_connection()


def test_persona_visual_schema_rejects_nested_windows_drive_storage_paths(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "storage-paths.db", client_id="storage-paths")
    try:
        connection = db.get_connection()
        pack_id, version_id = _insert_pack_version(connection, "Storage paths")

        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO persona_visual_pack_versions(
                    pack_id, version_number, renderer_type, manifest_version,
                    manifest_json, manifest_sha256, storage_relpath
                ) VALUES (?, 2, 'sprite_frames', 1, '{}', ?, ?)
                """,
                (pack_id, "b" * 64, "persona_visual/C:/manifest.json"),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO persona_visual_assets(
                    pack_id, pack_version_id, asset_key, role, storage_relpath,
                    mime_type, bytes, sha256, width, height
                ) VALUES (?, ?, 'idle', 'frame', ?, 'image/png', 1, ?, 1, 1)
                """,
                (
                    pack_id,
                    version_id,
                    "persona_visual/C:/idle.png",
                    "c" * 64,
                ),
            )
    finally:
        db.close_connection()


def test_v40_to_v41_failure_rolls_back_all_persona_visual_tables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "rollback.db"
    _seed_v40(path)
    migration_path = (
        Path(__file__).parents[2]
        / "tldw_chatbook"
        / "DB"
        / "migrations"
        / "chachanotes_v40_to_v41_persona_visual.sql"
    )
    original_read_text = Path.read_text

    def read_text_with_invalid_v41_sql(path_to_read: Path, *args, **kwargs) -> str:
        source = original_read_text(path_to_read, *args, **kwargs)
        if path_to_read != migration_path:
            return source
        marker = "\n\nCREATE TABLE persona_visual_pack_versions"
        assert marker in source
        return source.replace(marker, f"\n\nINVALID SQL;{marker}", 1)

    monkeypatch.setattr(Path, "read_text", read_text_with_invalid_v41_sql)

    with pytest.raises(SchemaError, match=r"V40.*V41"):
        CharactersRAGDB(path, client_id="failed-v41-open")

    with sqlite3.connect(path) as connection:
        assert _version(connection) == 40
        assert not (_tables(connection) & PERSONA_VISUAL_TABLES)
