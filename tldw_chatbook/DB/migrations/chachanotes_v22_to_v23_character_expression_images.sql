-- Migration: ChaChaNotes V22 to V23 character expression images
-- Adds the character_expression_images table for per-state reaction avatars
-- (state_id in thinking/speaking/error; idle reuses character_cards.image).
-- Local-only BLOB storage; no sync triggers (mirrors message_attachments).

CREATE TABLE IF NOT EXISTS character_expression_images(
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  character_id  INTEGER NOT NULL REFERENCES character_cards(id) ON DELETE CASCADE ON UPDATE CASCADE,
  state_id      TEXT    NOT NULL,
  image         BLOB    NOT NULL,
  mime          TEXT,
  created_at    TEXT    NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ','NOW')),
  updated_at    TEXT    NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ','NOW')),
  deleted       INTEGER NOT NULL DEFAULT 0,
  UNIQUE(character_id, state_id)
);
CREATE INDEX IF NOT EXISTS idx_char_expr_images_char ON character_expression_images(character_id);
UPDATE db_schema_version
   SET version = 23
 WHERE schema_name = 'rag_char_chat_schema'
   AND version = 22;
