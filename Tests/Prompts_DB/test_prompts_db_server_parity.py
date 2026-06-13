import json
import sqlite3

from tldw_chatbook.DB.Prompts_DB import PromptsDatabase


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
        assert {"prompt_format", "prompt_schema_version", "prompt_definition"}.issubset(columns)

        version = db.get_connection().execute("SELECT version FROM schema_version").fetchone()["version"]
        assert version == db._CURRENT_SCHEMA_VERSION

        prompt = db.fetch_prompt_details("00000000-0000-4000-8000-000000000001", include_deleted=True)
        assert prompt["name"] == "Existing Prompt"
        assert prompt["prompt_format"] == "legacy"
        assert prompt["prompt_schema_version"] is None
        assert prompt["prompt_definition"] is None
    finally:
        db.close_connection()
