-- Migration: ChaChaNotes V39 to V40 transcript annotations (task-17169).
--
-- Inline review annotations for Console selection feedback: a Comment on a
-- selected span persists here (in addition to its trajectory-sidecar audit
-- event) and renders as an inline marker on the anchored transcript row.
--
-- LOCAL-ONLY like message_trajectory_metadata: sync triggers in this DB are
-- opt-in per table and none are added here. Review notes are what THIS
-- device's operator said while reading THIS device's transcript.
--
-- Anchor: (conversation_id, row_key). The 2026-08-14 spec sketch wrote
-- "session_id", but native Console session ids are per-process; the durable
-- identity of a session is its persisted conversation, and the spec's own
-- row_key rule forbids runtime identities that orphan annotations on
-- reload. row_key is derived from persisted data (e.g. message:<db id>);
-- message_id is nullable because future row kinds may not be DB messages
-- (today, rows without a durable key are excluded from persistence).

CREATE TABLE IF NOT EXISTS transcript_annotations (
    annotation_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    row_key TEXT NOT NULL,
    message_id TEXT,
    quote_text TEXT NOT NULL,
    comment TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted BOOLEAN NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_transcript_annotations_conv_row
    ON transcript_annotations (conversation_id, row_key);

UPDATE db_schema_version
   SET version = 40
 WHERE schema_name = 'rag_char_chat_schema'
   AND version = 39;
