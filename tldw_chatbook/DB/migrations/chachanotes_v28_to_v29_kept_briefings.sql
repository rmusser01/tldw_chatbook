-- Migration: ChaChaNotes V28 to V29 kept briefings/scripts (task-1780).
--
-- Adds `kept_briefings` and `kept_scripts`: user-kept copies of Subscriptions_DB
-- briefings/briefing_scripts that must survive watchlist deletion (spec
-- Docs/superpowers/specs/2026-08-01-kept-briefings-design.md). Every kept row
-- is self-interpreting with the subscriptions DB gone entirely -- all
-- provenance is denormalized onto the row; `source_briefing_id` /
-- `source_script_id` are kept only for tracing, never as live references
-- (deliberately NOT foreign keys -- these tables live in a different
-- database file, ChaChaNotes, so a cross-DB FK is not possible in SQLite
-- and would be meaningless if it were).
--
-- Deliberately NO sync columns (`client_id`/`version`/`deleted`): these
-- tables do not participate in ChaChaNotes sync in v1 (recorded divergence,
-- follow-up task filed at close-out). Deletion is a hard DELETE, not a
-- soft-delete flag flip. Deliberately NO FTS: kept content is browsed
-- through a dedicated modal (list + detail), not searched via the
-- cross-entity FTS surfaces this DB maintains for character cards,
-- conversations, messages, notes, and keywords.
--
-- `kept_scripts.kept_briefing_id` IS a real intra-ChaChaNotes FK with
-- `ON DELETE CASCADE ON UPDATE CASCADE` -- a kept script has no meaning once
-- its kept briefing is deleted, and this connection pool always runs with
-- `PRAGMA foreign_keys = ON` (see `_get_thread_connection`), so the cascade
-- is real, not just documentation.
--
-- `kept_scripts.source_script_id` is UNIQUE but nullable: a script cast
-- directly from a kept briefing (`source_script_id = NULL`) has no
-- subscriptions-side source. SQLite's UNIQUE constraint treats NULL values
-- as distinct from one another, so multiple NULL-source scripts under the
-- same (or different) kept briefing are permitted.

CREATE TABLE IF NOT EXISTS kept_briefings(
  id                     INTEGER PRIMARY KEY AUTOINCREMENT,
  source_briefing_id     INTEGER NOT NULL UNIQUE,
  watchlist_name         TEXT,
  body_markdown          TEXT NOT NULL,
  covers_through_item_id INTEGER,
  covers_from_ts         DATETIME,
  selection_mode         TEXT,
  model_used             TEXT,
  item_count             INTEGER NOT NULL DEFAULT 0,
  featured_count         INTEGER NOT NULL DEFAULT 0,
  overflow_count         INTEGER NOT NULL DEFAULT 0,
  origin                 TEXT NOT NULL CHECK(origin IN ('manual','scheduled')),
  original_created_at    DATETIME,
  kept_at                DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_kept_briefings_kept_at ON kept_briefings(kept_at DESC, id DESC);

CREATE TABLE IF NOT EXISTS kept_scripts(
  id                   INTEGER PRIMARY KEY AUTOINCREMENT,
  kept_briefing_id     INTEGER NOT NULL REFERENCES kept_briefings(id) ON DELETE CASCADE ON UPDATE CASCADE,
  source_script_id     INTEGER UNIQUE,
  preset_name          TEXT NOT NULL,
  roster_snapshot_json TEXT NOT NULL,
  turns_json           TEXT NOT NULL,
  model_used           TEXT,
  original_created_at  DATETIME,
  kept_at              DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_kept_scripts_briefing ON kept_scripts(kept_briefing_id);
UPDATE db_schema_version
   SET version = 29
 WHERE schema_name = 'rag_char_chat_schema'
   AND version = 28;
