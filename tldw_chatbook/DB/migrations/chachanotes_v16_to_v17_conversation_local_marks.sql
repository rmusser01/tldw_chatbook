-- Migration: ChaChaNotes V16 to V17 local-only conversation marks
-- Adds durable local organization metadata without sync triggers.

CREATE TABLE IF NOT EXISTS conversation_local_marks (
  conversation_id TEXT NOT NULL,
  mark_type TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (conversation_id, mark_type)
);

CREATE INDEX IF NOT EXISTS idx_conversation_local_marks_type
  ON conversation_local_marks(mark_type, updated_at DESC, conversation_id);

UPDATE db_schema_version
   SET version = 17
 WHERE schema_name = 'rag_char_chat_schema'
   AND version = 16;
