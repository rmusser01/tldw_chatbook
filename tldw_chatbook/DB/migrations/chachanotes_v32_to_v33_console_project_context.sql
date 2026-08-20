-- ChaChaNotes v32 -> v33: local-only Console project-instruction controls.
-- DDL only. The column is deliberately absent from every conversations_sync_*
-- trigger condition and payload, so writes never enter sync_log.
--
-- SQLite has no ADD COLUMN IF NOT EXISTS. The Python migration runner checks
-- PRAGMA table_info(conversations), skips this ALTER when a partial migration
-- already added the column, and still applies the guarded version bump.

ALTER TABLE conversations ADD COLUMN console_project_context_json TEXT;
