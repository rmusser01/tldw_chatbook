-- ADR-147: local archive state independent of workflow/deletion. Applied by the
-- cursor-based migration under its existing transaction (no executescript).
ALTER TABLE conversations ADD COLUMN archived INTEGER NOT NULL DEFAULT 0
    CHECK (archived IN (0, 1));
CREATE INDEX idx_conversations_archive
    ON conversations(archived, deleted, last_modified DESC, id DESC);
UPDATE db_schema_version SET version = 71
    WHERE schema_name = 'rag_char_chat_schema' AND version = 70;
