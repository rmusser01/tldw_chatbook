-- Migration: ChaChaNotes V25 to V26 — conversations.context_summary,
-- conversations.summary_boundary_message_id
-- Adds two nullable, LOCAL-ONLY columns recording the Console `/rewind`
-- "summarize up to here" boundary summary: an LLM recap of the active path
-- before ``summary_boundary_message_id``, used to compact the provider
-- payload while the visible transcript stays full. They are intentionally
-- NOT synced: the setter writes both atomically without bumping
-- version/last_modified, so the conversations_sync_* triggers never fire and
-- the columns are never in a sync payload. The runner guards each ADD COLUMN
-- with a PRAGMA check (SQLite has no ADD COLUMN IF NOT EXISTS) so
-- replayed/partial migrations are idempotent.

ALTER TABLE conversations ADD COLUMN context_summary TEXT;
ALTER TABLE conversations ADD COLUMN summary_boundary_message_id TEXT;

UPDATE db_schema_version
   SET version = 26
 WHERE schema_name = 'rag_char_chat_schema'
   AND version = 25;
