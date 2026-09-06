"""Focused v66 -> v68 persistence tests for durable Canvas revisions."""

from __future__ import annotations

import sqlite3
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import pytest

from Tests.ChaChaNotesDB.historical_bootstrap import (
    SCHEMA_NAME,
    chachanotes_db_at_version,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, SchemaError

MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[2] / "tldw_chatbook" / "DB" / "migrations"
)
CANVAS_TABLES = {
    "canvas_conversation_hints",
    "canvas_documents",
    "canvas_revisions",
}
CANVAS_INDEXES = {
    "idx_canvas_documents_conversation",
    "idx_canvas_revisions_canvas_sequence",
    "idx_canvas_revisions_origin_message",
    "idx_canvas_revisions_parent",
    "uq_canvas_documents_id_conversation",
    "uq_canvas_revisions_id_canvas",
}
CANVAS_TRIGGERS = {
    "canvas_documents_ownership_immutable",
    "canvas_origin_message_owner_guard",
    "canvas_revisions_no_delete",
    "canvas_revisions_no_update",
    "canvas_revisions_origin_owner_guard",
    "canvas_revisions_parent_guard",
}


def _owner(db: CharactersRAGDB) -> tuple[str, str]:
    conversation_id = db.add_conversation({"title": "Canvas archive owner"})
    assert conversation_id is not None
    message_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "origin",
        }
    )
    assert message_id is not None
    return str(conversation_id), str(message_id)


def _version(path: Path) -> int:
    connection = sqlite3.connect(path)
    try:
        row = connection.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (SCHEMA_NAME,),
        ).fetchone()
        assert row is not None
        return int(row[0])
    finally:
        connection.close()


def _canvas_schema(connection: sqlite3.Connection) -> dict[tuple[str, str], str]:
    rows = connection.execute(
        "SELECT type, name, sql FROM sqlite_master "
        "WHERE name LIKE 'canvas_%' OR name LIKE 'idx_canvas_%' "
        "OR name LIKE 'uq_canvas_%' ORDER BY type, name"
    ).fetchall()
    return {(str(row[0]), str(row[1])): str(row[2]) for row in rows}


def _create_legacy_canvas_only_database(path: Path, *, version: int) -> str:
    """Reproduce the unmerged Canvas branch's ambiguous v66/v67 schema."""

    with chachanotes_db_at_version(path, 65) as legacy:
        conversation_id, message_id = _owner(legacy)
        creation_sql = (
            MIGRATIONS_DIR / "chachanotes_v66_to_v67_canvas_revisions.sql"
        ).read_text(encoding="utf-8")
        with legacy.transaction(immediate=True) as cursor:
            legacy._execute_migration_statements(
                cursor,
                creation_sql,
                "legacy Canvas V65→V66",
            )
            cursor.execute(
                "UPDATE db_schema_version SET version = 66 "
                "WHERE schema_name = ? AND version = 65",
                (SCHEMA_NAME,),
            )

        source = "<main>pre-release Canvas row</main>"
        canvas_id = str(uuid4())
        revision_id = str(uuid4())
        with legacy.transaction(immediate=True) as cursor:
            cursor.execute(
                "INSERT INTO canvas_documents "
                "(id, conversation_id, created_at, deleted, deleted_at) "
                "VALUES (?, ?, '2026-09-05T11:00:00+00:00', 0, NULL)",
                (canvas_id, conversation_id),
            )
            cursor.execute(
                "INSERT INTO canvas_revisions "
                "(id, canvas_id, parent_revision_id, sequence, title, "
                "runtime_profile, html, content_sha256, html_bytes, actor_kind, "
                "origin_message_id, origin_turn_id, created_at, deleted_at) "
                "VALUES (?, ?, NULL, 1, 'Pre-release', 'canvas-v1', ?, ?, ?, "
                "'assistant', ?, 'legacy-turn', "
                "'2026-09-05T11:00:00+00:00', NULL)",
                (
                    revision_id,
                    canvas_id,
                    source,
                    sha256(source.encode("utf-8")).hexdigest(),
                    len(source.encode("utf-8")),
                    message_id,
                ),
            )

        if version == 67:
            profile_sql = (
                MIGRATIONS_DIR / "chachanotes_v67_to_v68_canvas_runtime_profiles.sql"
            ).read_text(encoding="utf-8")
            with legacy.transaction(immediate=True) as cursor:
                legacy._execute_migration_statements(
                    cursor,
                    profile_sql,
                    "legacy Canvas V66→V67",
                )
                cursor.execute(
                    "UPDATE db_schema_version SET version = 67 "
                    "WHERE schema_name = ? AND version = 66",
                    (SCHEMA_NAME,),
                )

    return revision_id


@pytest.mark.parametrize("legacy_version", [66, 67])
def test_legacy_canvas_only_schema_is_refused_without_mutation(
    tmp_path: Path,
    legacy_version: int,
) -> None:
    """Ambiguous pre-release numbering must not be guessed or relabeled."""

    path = tmp_path / f"legacy-canvas-v{legacy_version}.sqlite"
    revision_id = _create_legacy_canvas_only_database(path, version=legacy_version)
    before = sqlite3.connect(path)
    try:
        expected_schema = _canvas_schema(before)
        expected_row = before.execute(
            "SELECT html, runtime_profile FROM canvas_revisions WHERE id = ?",
            (revision_id,),
        ).fetchone()
        assert expected_row is not None
        assert (
            before.execute(
                "SELECT COUNT(*) FROM sqlite_master "
                "WHERE name = 'character_conversation_search_state'"
            ).fetchone()[0]
            == 0
        )
    finally:
        before.close()

    with pytest.raises(
        SchemaError,
        match="incompatible Canvas migration predecessor schema",
    ):
        CharactersRAGDB(path, client_id=f"legacy-canvas-v{legacy_version}")

    after = sqlite3.connect(path)
    try:
        assert _version(path) == legacy_version
        assert _canvas_schema(after) == expected_schema
        assert (
            after.execute(
                "SELECT html, runtime_profile FROM canvas_revisions WHERE id = ?",
                (revision_id,),
            ).fetchone()
            == expected_row
        )
        assert (
            after.execute(
                "SELECT COUNT(*) FROM sqlite_master "
                "WHERE name = 'character_conversation_search_state'"
            ).fetchone()[0]
            == 0
        )
    finally:
        after.close()


def test_genuine_v66_database_migrates_to_v68_with_complete_canvas_schema(
    tmp_path: Path,
) -> None:
    """A missing v66 dispatch or schema object leaves this real upgrade red."""

    path = tmp_path / "genuine-v66.sqlite"
    with chachanotes_db_at_version(path, 66) as historical:
        assert _canvas_schema(historical.get_connection()) == {}

    migrated = CharactersRAGDB(path, client_id="canvas-v68-migrated")
    try:
        connection = migrated.get_connection()
        assert _version(path) == 68
        objects = _canvas_schema(connection)
        assert {
            name for (object_type, name) in objects if object_type == "table"
        } == CANVAS_TABLES
        assert {
            name for (object_type, name) in objects if object_type == "index"
        } == CANVAS_INDEXES
        assert {
            name for (object_type, name) in objects if object_type == "trigger"
        } == CANVAS_TRIGGERS
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        migrated.close_connection()


def test_fresh_v68_schema_matches_migrated_v66_and_reopen_is_idempotent(
    tmp_path: Path,
) -> None:
    """Fresh-only DDL or replay-only DDL cannot silently diverge."""

    fresh_path = tmp_path / "fresh.sqlite"
    migrated_path = tmp_path / "migrated.sqlite"
    with chachanotes_db_at_version(migrated_path, 66):
        pass

    fresh = CharactersRAGDB(fresh_path, client_id="canvas-v68-fresh")
    migrated = CharactersRAGDB(migrated_path, client_id="canvas-v68-replay")
    try:
        fresh_schema = _canvas_schema(fresh.get_connection())
        migrated_schema = _canvas_schema(migrated.get_connection())
        assert fresh._CURRENT_SCHEMA_VERSION == 68
        assert fresh_schema == migrated_schema
        assert fresh_schema
    finally:
        fresh.close_connection()
        migrated.close_connection()

    reopened = CharactersRAGDB(fresh_path, client_id="canvas-v68-reopen")
    try:
        assert _version(fresh_path) == 68
        assert _canvas_schema(reopened.get_connection()) == fresh_schema
        assert (
            reopened.get_connection().execute("PRAGMA foreign_key_check").fetchall()
            == []
        )
    finally:
        reopened.close_connection()


def test_v66_migration_rolls_back_all_ddl_and_version_then_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failure after the real DDL must leave a genuine v66 database retryable."""

    path = tmp_path / "rollback.sqlite"
    with chachanotes_db_at_version(path, 66):
        pass

    original = CharactersRAGDB._execute_migration_statements

    def fail_after_canvas_ddl(self, cursor, script, label):
        original(self, cursor, script, label)
        if label == "V66→V67":
            raise sqlite3.OperationalError("injected canvas migration failure")

    monkeypatch.setattr(
        CharactersRAGDB,
        "_execute_migration_statements",
        fail_after_canvas_ddl,
    )
    with pytest.raises(SchemaError):
        CharactersRAGDB(path, client_id="canvas-v68-fail")

    connection = sqlite3.connect(path)
    try:
        assert _version(path) == 66
        assert _canvas_schema(connection) == {}
    finally:
        connection.close()

    monkeypatch.setattr(
        CharactersRAGDB,
        "_execute_migration_statements",
        original,
    )
    retried = CharactersRAGDB(path, client_id="canvas-v68-retry")
    try:
        assert _version(path) == 68
        assert _canvas_schema(retried.get_connection())
    finally:
        retried.close_connection()


def test_v67_migration_preserves_canvas_rows_and_accepts_inert_runtime_profiles(
    tmp_path: Path,
) -> None:
    """Keeping the v67 equality CHECK would reject an archived future profile."""

    path = tmp_path / "genuine-v67.sqlite"
    with chachanotes_db_at_version(path, 67) as historical:
        conversation_id, message_id = _owner(historical)
        source = "<main>preserved ✨</main>"
        canvas_id = str(uuid4())
        revision_id = str(uuid4())
        digest = sha256(source.encode("utf-8")).hexdigest()
        with historical.transaction(immediate=True) as cursor:
            cursor.execute(
                "INSERT INTO canvas_documents "
                "(id, conversation_id, created_at, deleted, deleted_at) "
                "VALUES (?, ?, ?, 0, NULL)",
                (canvas_id, conversation_id, "2026-09-04T12:00:00+00:00"),
            )
            cursor.execute(
                "INSERT INTO canvas_revisions "
                "(id, canvas_id, parent_revision_id, sequence, title, "
                "runtime_profile, html, content_sha256, html_bytes, actor_kind, "
                "origin_message_id, origin_turn_id, created_at, deleted_at) "
                "VALUES (?, ?, NULL, 1, ?, 'canvas-v1', ?, ?, ?, 'assistant', "
                "?, ?, ?, NULL)",
                (
                    revision_id,
                    canvas_id,
                    "Preserved",
                    source,
                    digest,
                    len(source.encode("utf-8")),
                    message_id,
                    "turn-preserved",
                    "2026-09-04T12:00:00+00:00",
                ),
            )

    migrated = CharactersRAGDB(path, client_id="canvas-v68-migrated")
    try:
        connection = migrated.get_connection()
        assert _version(path) == 68
        preserved = connection.execute(
            "SELECT html, runtime_profile FROM canvas_revisions WHERE id = ?",
            (revision_id,),
        ).fetchone()
        assert preserved is not None
        assert tuple(preserved) == (source, "canvas-v1")

        future_source = "<main>stored, never executed</main>"
        with migrated.transaction(immediate=True) as cursor:
            cursor.execute(
                "INSERT INTO canvas_revisions "
                "(id, canvas_id, parent_revision_id, sequence, title, "
                "runtime_profile, html, content_sha256, html_bytes, actor_kind, "
                "origin_message_id, origin_turn_id, created_at, deleted_at) "
                "VALUES (?, ?, ?, 2, ?, ?, ?, ?, ?, 'user_import', ?, ?, ?, NULL)",
                (
                    str(uuid4()),
                    canvas_id,
                    revision_id,
                    "Future",
                    "canvas-v9",
                    future_source,
                    sha256(future_source.encode("utf-8")).hexdigest(),
                    len(future_source.encode("utf-8")),
                    message_id,
                    "turn-future",
                    "2026-09-04T12:01:00+00:00",
                ),
            )
        assert (
            connection.execute(
                "SELECT runtime_profile FROM canvas_revisions WHERE sequence = 2"
            ).fetchone()[0]
            == "canvas-v9"
        )
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        migrated.close_connection()


def test_v67_to_v68_migration_rolls_back_and_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure after table rebuild leaves the genuine v67 graph untouched."""

    path = tmp_path / "v67-rollback.sqlite"
    with chachanotes_db_at_version(path, 67) as historical:
        conversation_id, message_id = _owner(historical)
        source = "<main>rollback</main>"
        canvas_id = str(uuid4())
        revision_id = str(uuid4())
        with historical.transaction(immediate=True) as cursor:
            cursor.execute(
                "INSERT INTO canvas_documents "
                "(id, conversation_id, created_at, deleted, deleted_at) "
                "VALUES (?, ?, ?, 0, NULL)",
                (canvas_id, conversation_id, "2026-09-04T13:00:00+00:00"),
            )
            cursor.execute(
                "INSERT INTO canvas_revisions "
                "(id, canvas_id, parent_revision_id, sequence, title, "
                "runtime_profile, html, content_sha256, html_bytes, actor_kind, "
                "origin_message_id, origin_turn_id, created_at, deleted_at) "
                "VALUES (?, ?, NULL, 1, 'Rollback', 'canvas-v1', ?, ?, ?, "
                "'assistant', ?, 'turn-rollback', ?, NULL)",
                (
                    revision_id,
                    canvas_id,
                    source,
                    sha256(source.encode()).hexdigest(),
                    len(source.encode()),
                    message_id,
                    "2026-09-04T13:00:00+00:00",
                ),
            )

    original = CharactersRAGDB._execute_migration_statements

    def fail_after_rebuild(self, cursor, script, label):
        original(self, cursor, script, label)
        if label == "V67→V68":
            raise sqlite3.OperationalError("injected v68 failure")

    monkeypatch.setattr(
        CharactersRAGDB, "_execute_migration_statements", fail_after_rebuild
    )
    with pytest.raises(SchemaError):
        CharactersRAGDB(path, client_id="canvas-v68-fail")
    connection = sqlite3.connect(path)
    try:
        assert _version(path) == 67
        assert connection.execute(
            "SELECT html, runtime_profile FROM canvas_revisions WHERE id = ?",
            (revision_id,),
        ).fetchone() == (source, "canvas-v1")
    finally:
        connection.close()

    monkeypatch.setattr(CharactersRAGDB, "_execute_migration_statements", original)
    retried = CharactersRAGDB(path, client_id="canvas-v68-retry")
    try:
        assert _version(path) == 68
        row = (
            retried.get_connection()
            .execute(
                "SELECT html, runtime_profile FROM canvas_revisions WHERE id = ?",
                (revision_id,),
            )
            .fetchone()
        )
        assert row is not None
        assert tuple(row) == (source, "canvas-v1")
    finally:
        retried.close_connection()
