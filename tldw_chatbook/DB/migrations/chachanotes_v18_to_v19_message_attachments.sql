-- Migration: ChaChaNotes V18 to V19 message attachments
-- Adds the message_attachments table for extra Console message attachments
-- (positions >= 1; position 0 stays in messages.image_data/image_mime_type).
-- No sync triggers are added here; sync wiring is tracked separately
-- (TASK-220).

CREATE TABLE IF NOT EXISTS message_attachments(
  message_id   TEXT    NOT NULL REFERENCES messages(id) ON DELETE CASCADE ON UPDATE CASCADE,
  position     INTEGER NOT NULL CHECK (position >= 1),
  data         BLOB    NOT NULL,
  mime_type    TEXT    NOT NULL,
  display_name TEXT    NOT NULL DEFAULT '',
  PRIMARY KEY (message_id, position)
);
CREATE INDEX IF NOT EXISTS idx_message_attachments_message ON message_attachments(message_id);
UPDATE db_schema_version
   SET version = 19
 WHERE schema_name = 'rag_char_chat_schema'
   AND version = 18;
