-- Migration from V24 to V25: Add message_generation_metadata sidecar table
--
-- This migration adds the `message_generation_metadata` table to store
-- image generation metadata (prompts, backend, model, seed, etc.) for messages.
-- The table is local-only (not synced), following the precedent of v19 (message_attachments)
-- and v24 (active_leaf_message_id).

CREATE TABLE IF NOT EXISTS message_generation_metadata(
  message_id      TEXT    NOT NULL REFERENCES messages(id) ON DELETE CASCADE ON UPDATE CASCADE,
  position        INTEGER NOT NULL CHECK (position >= 0),
  prompt          TEXT    NOT NULL,
  negative_prompt TEXT    NOT NULL DEFAULT '',
  backend         TEXT    NOT NULL,
  model           TEXT,
  seed            INTEGER,
  style           TEXT,
  params_json     TEXT    NOT NULL DEFAULT '{}',
  created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (message_id, position)
);
CREATE INDEX IF NOT EXISTS idx_msg_gen_meta_message ON message_generation_metadata(message_id);

UPDATE db_schema_version
   SET version = 25
 WHERE schema_name = 'rag_char_chat_schema'
   AND version = 24;
