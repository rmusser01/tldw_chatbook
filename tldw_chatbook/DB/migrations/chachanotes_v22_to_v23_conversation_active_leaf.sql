-- Migration: ChaChaNotes V22 to V23 — conversations.active_leaf_message_id
-- Adds a nullable, LOCAL-ONLY pointer column recording which leaf message is
-- the active branch tip for the native Console. It is intentionally NOT synced:
-- the setter writes it without bumping version/last_modified, so the
-- conversations_sync_* triggers never fire and the column is never in a sync
-- payload. The runner guards the ADD COLUMN with a PRAGMA check (SQLite has no
-- ADD COLUMN IF NOT EXISTS) so replayed/partial migrations are idempotent.

ALTER TABLE conversations ADD COLUMN active_leaf_message_id TEXT;

UPDATE db_schema_version
   SET version = 23
 WHERE schema_name = 'rag_char_chat_schema'
   AND version = 22;
