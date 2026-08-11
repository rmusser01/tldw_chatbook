-- V33 -> V34: sparse visual-compaction representation preference
-- (ADR-054 / TASK-14914). Capability support remains request-time state.

ALTER TABLE console_conversation_context_policy
  ADD COLUMN compaction_representation TEXT
    CHECK(compaction_representation IN ('text_summary','visual_transcript','hybrid'));

UPDATE db_schema_version
   SET version = 34
 WHERE schema_name = 'rag_char_chat_schema'
   AND version = 33;
