-- Migration: Workspace registry V6 to V7 assistant defaults
-- Adds the per-workspace reference-backed default-assistant JSON column and
-- the agent backfill completion flag table (task: workspace assistant
-- defaults, storage layer).
--
-- Procedural step (not expressible as plain SQL, see the runner in
-- tldw_chatbook/DB/Workspace_DB.py `_initialize_schema`):
--   SQLite's ALTER TABLE has no IF NOT EXISTS form, so the runner guards
--   the ALTER below with a PRAGMA table_info(workspace_records) check --
--   a partially-migrated database (column present, version row missing)
--   converges instead of failing with "duplicate column name".
--   All statements run inside a single transaction so a mid-migration
--   failure rolls back atomically.

ALTER TABLE workspace_records ADD COLUMN assistant_defaults TEXT;

CREATE TABLE IF NOT EXISTS workspace_agent_backfill (
    key TEXT PRIMARY KEY,
    completed_at TEXT NOT NULL
);

INSERT OR IGNORE INTO schema_version (version) VALUES (7);
