import json
import sqlite3

import pytest

from tldw_chatbook.DB.Prompts_DB import (
    InputError,
    PromptsDatabase,
    add_or_update_prompt,
)


def seed_v2_prompt_database(database_path, *, name, user_prompt):
    """Create a real schema-v2 file with indexed structured metadata."""
    conn = sqlite3.connect(database_path)
    try:
        conn.row_factory = sqlite3.Row
        conn.executescript(
            f"""
            {PromptsDatabase._TABLES_SQL_V1}
            {PromptsDatabase._INDICES_SQL_V1}
            {PromptsDatabase._TRIGGERS_SQL_V1}
            ALTER TABLE Prompts ADD COLUMN prompt_format TEXT NOT NULL DEFAULT 'legacy';
            ALTER TABLE Prompts ADD COLUMN prompt_schema_version INTEGER;
            ALTER TABLE Prompts ADD COLUMN prompt_definition TEXT;
            UPDATE schema_version SET version = 2;
            {PromptsDatabase._FTS_TABLES_SQL}
            """
        )
        conn.execute(
            """
            INSERT INTO Prompts (
                name, author, details, system_prompt, user_prompt, uuid,
                last_modified, version, client_id, deleted, prompt_format,
                prompt_schema_version, prompt_definition
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                "Legacy Author",
                "structured metadata must survive",
                "legacy system",
                user_prompt,
                "00000000-0000-4000-8000-000000000002",
                "2026-08-01T00:00:00.000Z",
                1,
                "legacy-client",
                0,
                "structured",
                1,
                json.dumps({"schema_version": 1, "messages": []}),
            ),
        )
        prompt_id = conn.execute(
            "SELECT id FROM Prompts WHERE name = ?", (name,)
        ).fetchone()[0]
        conn.execute(
            """
            INSERT INTO prompts_fts (
                rowid, name, author, details, system_prompt, user_prompt
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                prompt_id,
                name,
                "Legacy Author",
                "structured metadata must survive",
                "legacy system",
                user_prompt,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def test_prompt_db_persists_structured_prompt_metadata():
    db = PromptsDatabase(":memory:", client_id="test-client")
    try:
        prompt_id, prompt_uuid, _ = db.add_prompt(
            "Structured Prompt",
            "Parity Tester",
            "Prompt with structured metadata",
            "legacy system",
            "legacy user",
        )

        db.update_prompt_by_id(
            prompt_id,
            {
                "prompt_format": "structured",
                "prompt_schema_version": 1,
                "prompt_definition": json.dumps(
                    {
                        "schema_version": 1,
                        "messages": [{"role": "system", "content": "hi"}],
                    }
                ),
            },
        )

        prompt = db.fetch_prompt_details(prompt_uuid, include_deleted=True)
        assert prompt["prompt_format"] == "structured"
        assert prompt["prompt_schema_version"] == 1
        assert json.loads(prompt["prompt_definition"])["schema_version"] == 1
    finally:
        db.close_connection()


def test_prompt_db_migrates_v1_database_for_structured_fields(tmp_path):
    db_path = tmp_path / "prompts_v1.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER NOT NULL
            );
            INSERT INTO schema_version (version) VALUES (1);

            CREATE TABLE IF NOT EXISTS Prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL COLLATE NOCASE,
                author TEXT,
                details TEXT,
                system_prompt TEXT,
                user_prompt TEXT,
                uuid TEXT UNIQUE NOT NULL,
                last_modified DATETIME NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                client_id TEXT NOT NULL,
                deleted BOOLEAN NOT NULL DEFAULT 0,
                prev_version INTEGER,
                merge_parent_uuid TEXT
            );

            CREATE TABLE IF NOT EXISTS PromptKeywordsTable (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keyword TEXT NOT NULL COLLATE NOCASE,
                uuid TEXT UNIQUE NOT NULL,
                last_modified DATETIME NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                client_id TEXT NOT NULL,
                deleted BOOLEAN NOT NULL DEFAULT 0,
                prev_version INTEGER,
                merge_parent_uuid TEXT
            );

            CREATE TABLE IF NOT EXISTS PromptKeywordLinks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_id INTEGER NOT NULL,
                keyword_id INTEGER NOT NULL,
                UNIQUE (prompt_id, keyword_id),
                FOREIGN KEY (prompt_id) REFERENCES Prompts(id) ON DELETE CASCADE,
                FOREIGN KEY (keyword_id) REFERENCES PromptKeywordsTable(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS sync_log (
                change_id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity TEXT NOT NULL,
                entity_uuid TEXT NOT NULL,
                operation TEXT NOT NULL CHECK(operation IN ('create','update','delete', 'link', 'unlink')),
                timestamp DATETIME NOT NULL,
                client_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                payload TEXT
            );

            INSERT INTO Prompts (
                name,
                author,
                details,
                system_prompt,
                user_prompt,
                uuid,
                last_modified,
                version,
                client_id,
                deleted
            ) VALUES (
                'Existing Prompt',
                'Legacy Author',
                'Migrated from schema v1',
                'system',
                'user',
                '00000000-0000-4000-8000-000000000001',
                '2026-04-19T00:00:00.000Z',
                1,
                'legacy-client',
                0
            );
            """
        )
        conn.commit()
    finally:
        conn.close()

    db = PromptsDatabase(db_path, client_id="test-client")
    try:
        cursor = db.get_connection().execute("PRAGMA table_info(Prompts)")
        columns = {row["name"] for row in cursor.fetchall()}
        assert {"prompt_format", "prompt_schema_version", "prompt_definition"}.issubset(
            columns
        )

        version = (
            db.get_connection()
            .execute("SELECT version FROM schema_version")
            .fetchone()["version"]
        )
        assert version == db._CURRENT_SCHEMA_VERSION

        prompt = db.fetch_prompt_details(
            "00000000-0000-4000-8000-000000000001", include_deleted=True
        )
        assert prompt["name"] == "Existing Prompt"
        assert prompt["prompt_format"] == "legacy"
        assert prompt["prompt_schema_version"] is None
        assert prompt["prompt_definition"] is None
    finally:
        db.close_connection()


def test_v2_migration_defaults_existing_rows_to_prompt_and_preserves_sync_and_fts(
    tmp_path,
):
    database_path = tmp_path / "prompts_v2.db"
    seed_v2_prompt_database(
        database_path, name="Existing", user_prompt="alpha searchable"
    )

    database = PromptsDatabase(database_path, client_id="migration-test")
    try:
        detail = database.fetch_prompt_details("Existing")
        assert (
            database._get_db_version(database.get_connection())
            == database._CURRENT_SCHEMA_VERSION
        )
        assert detail["artifact_type"] == "prompt"
        assert detail["prompt_format"] == "structured"
        assert detail["prompt_schema_version"] == 1
        assert json.loads(detail["prompt_definition"]) == {
            "schema_version": 1,
            "messages": [],
        }

        searched, total = database.search_prompts("searchable")
        assert total == 1
        assert searched[0]["name"] == "Existing"

        with pytest.raises(sqlite3.IntegrityError, match="Version must increment"):
            database.get_connection().execute(
                "UPDATE Prompts SET details = ? WHERE id = ?",
                ("illegal update", detail["id"]),
            )
    finally:
        database.close_connection()


def test_fresh_v3_database_has_prompt_artifact_default():
    database = PromptsDatabase(":memory:", client_id="fresh-schema")
    try:
        columns = {
            row["name"]: row
            for row in database.get_connection().execute("PRAGMA table_info(Prompts)")
        }
        assert columns["artifact_type"]["notnull"] == 1
        assert columns["artifact_type"]["dflt_value"] == "'prompt'"
    finally:
        database.close_connection()


def test_standalone_overwrite_helper_preserves_recipe_artifact_type():
    database = PromptsDatabase(":memory:", client_id="standalone-overwrite")
    try:
        add_or_update_prompt(
            database,
            name="Reusable Recipe",
            author="Author",
            details="initial",
            user_prompt="initial compiled text",
            prompt_format="structured",
            prompt_schema_version=2,
            prompt_definition={
                "kind": "block_recipe",
                "schema_version": 2,
                "lanes": [],
            },
            artifact_type="recipe",
        )
        add_or_update_prompt(
            database,
            name="Reusable Recipe",
            author="Author",
            details="overwritten",
            user_prompt="updated compiled text",
            artifact_type="recipe",
        )

        detail = database.fetch_prompt_details("Reusable Recipe")
        assert detail["artifact_type"] == "recipe"
        assert detail["details"] == "overwritten"

        with pytest.raises(InputError, match="artifact_type"):
            add_or_update_prompt(
                database,
                name="Invalid artifact",
                author=None,
                details=None,
                artifact_type="invalid",
            )
    finally:
        database.close_connection()
