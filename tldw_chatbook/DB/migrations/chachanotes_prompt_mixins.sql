CREATE TABLE console_prompt_mixin_preferences (
    conversation_id TEXT PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
    schema_version INTEGER NOT NULL CHECK(schema_version > 0),
    revision INTEGER NOT NULL CHECK(revision >= 0),
    selection_json TEXT NOT NULL CHECK(length(CAST(selection_json AS BLOB)) <= 512000)
);
CREATE TABLE console_prompt_mixin_revisions (
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    snapshot_id TEXT NOT NULL,
    digest TEXT NOT NULL CHECK(length(digest) = 64),
    authority_key TEXT NOT NULL,
    backend TEXT NOT NULL CHECK(backend IN ('local', 'server')),
    record_uuid TEXT NOT NULL,
    record_version INTEGER NOT NULL CHECK(record_version > 0),
    name TEXT NOT NULL,
    placement TEXT NOT NULL CHECK(placement IN ('before', 'after')),
    body TEXT NOT NULL CHECK(length(body) > 0 AND length(body) <= 20000),
    PRIMARY KEY(conversation_id, snapshot_id)
);
CREATE INDEX idx_console_mixin_revision_digest
ON console_prompt_mixin_revisions(conversation_id, digest);
CREATE TRIGGER console_prompt_mixin_revision_immutable
BEFORE UPDATE ON console_prompt_mixin_revisions BEGIN
    SELECT RAISE(ABORT, 'mixin_immutable_revision');
END;
CREATE TABLE console_prompt_mixin_attempts (
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    attempt_id TEXT NOT NULL,
    schema_version INTEGER NOT NULL CHECK(schema_version = 1),
    user_message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
    assistant_message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
    retry_of_attempt_id TEXT,
    replaces_attempt_id TEXT,
    result_revision_id TEXT REFERENCES console_trace_semantic_revisions(revision_id) ON DELETE SET NULL,
    trace_call_id TEXT REFERENCES console_trace_calls(call_id) ON DELETE SET NULL,
    expected_item_count INTEGER NOT NULL CHECK(expected_item_count >= 0),
    PRIMARY KEY(conversation_id, attempt_id),
    FOREIGN KEY(conversation_id, replaces_attempt_id)
        REFERENCES console_prompt_mixin_attempts(conversation_id, attempt_id)
        DEFERRABLE INITIALLY DEFERRED,
    FOREIGN KEY(conversation_id, retry_of_attempt_id)
        REFERENCES console_prompt_mixin_attempts(conversation_id, attempt_id)
        DEFERRABLE INITIALLY DEFERRED
);
CREATE INDEX idx_console_mixin_attempt_user ON console_prompt_mixin_attempts(user_message_id);
CREATE INDEX idx_console_mixin_attempt_assistant ON console_prompt_mixin_attempts(assistant_message_id);
CREATE TRIGGER console_prompt_mixin_attempt_anchor_insert
BEFORE INSERT ON console_prompt_mixin_attempts BEGIN
    SELECT RAISE(ABORT, 'mixin_anchor_ownership') WHERE
        NEW.result_revision_id IS NOT NULL OR NEW.trace_call_id IS NOT NULL OR
        (NEW.replaces_attempt_id IS NOT NULL AND NOT EXISTS (
            SELECT 1 FROM console_prompt_mixin_attempts p WHERE p.conversation_id=NEW.conversation_id
              AND p.attempt_id=NEW.replaces_attempt_id AND p.user_message_id=NEW.user_message_id)) OR
        (NEW.user_message_id IS NOT NULL AND NOT EXISTS (
            SELECT 1 FROM messages WHERE id=NEW.user_message_id
              AND conversation_id=NEW.conversation_id AND sender='user')) OR
        (NEW.assistant_message_id IS NOT NULL AND NOT EXISTS (
            SELECT 1 FROM messages WHERE id=NEW.assistant_message_id
              AND conversation_id=NEW.conversation_id AND sender='assistant'));
END;
CREATE TRIGGER console_prompt_mixin_attempt_immutable
BEFORE UPDATE ON console_prompt_mixin_attempts BEGIN
    SELECT RAISE(ABORT, 'mixin_immutable_attempt') WHERE
        NEW.conversation_id IS NOT OLD.conversation_id OR
        NEW.attempt_id IS NOT OLD.attempt_id OR
        NEW.schema_version IS NOT OLD.schema_version OR
        NEW.retry_of_attempt_id IS NOT OLD.retry_of_attempt_id OR
        NEW.replaces_attempt_id IS NOT OLD.replaces_attempt_id OR
        (NEW.result_revision_id IS NOT OLD.result_revision_id AND OLD.result_revision_id IS NOT NULL
         AND (NEW.result_revision_id IS NOT NULL OR EXISTS (
             SELECT 1 FROM console_trace_semantic_revisions WHERE revision_id=OLD.result_revision_id))) OR
        (NEW.trace_call_id IS NOT OLD.trace_call_id AND OLD.trace_call_id IS NOT NULL
         AND (NEW.trace_call_id IS NOT NULL OR EXISTS (
             SELECT 1 FROM console_trace_calls WHERE call_id=OLD.trace_call_id))) OR
        NEW.expected_item_count IS NOT OLD.expected_item_count OR
        (NEW.user_message_id IS NOT OLD.user_message_id AND NEW.user_message_id IS NOT NULL) OR
        (NEW.assistant_message_id IS NOT OLD.assistant_message_id AND NEW.assistant_message_id IS NOT NULL);
    SELECT RAISE(ABORT, 'mixin_result_seal_ownership') WHERE
        NEW.result_revision_id IS NOT OLD.result_revision_id AND NEW.result_revision_id IS NOT NULL
        AND NOT EXISTS (
            SELECT 1 FROM console_trace_semantic_revisions r
            JOIN messages m ON m.id=r.live_message_id
            WHERE r.revision_id=NEW.result_revision_id AND r.source_message_id=NEW.assistant_message_id
              AND r.source_conversation_id=NEW.conversation_id AND r.normalized_role='assistant'
              AND m.conversation_id=NEW.conversation_id AND m.deleted=0
              AND m.assistant_generation_state='complete' AND NEW.replaces_attempt_id IS NOT NULL
              AND json_extract(m.metadata_json,'$.mixin_attempt_id')=NEW.attempt_id
              AND json_extract(m.metadata_json,'$.mixin_snapshot_schema_version')=1
              AND r.revision_sequence=(SELECT MAX(revision_sequence) FROM console_trace_semantic_revisions
                  WHERE live_message_id=m.id));
    SELECT RAISE(ABORT, 'mixin_trace_seal_ownership') WHERE
        NEW.trace_call_id IS NOT OLD.trace_call_id AND NEW.trace_call_id IS NOT NULL
        AND NOT EXISTS (
            SELECT 1 FROM console_trace_calls c
            JOIN console_trace_owners o ON o.owner_id=c.owner_id
            JOIN console_dispatch_checkpoints d ON d.conversation_id=o.conversation_id
            WHERE c.call_id=NEW.trace_call_id AND c.turn_id=NEW.user_message_id
              AND c.call_sequence=0 AND o.conversation_id=NEW.conversation_id
              AND d.attempt_id=NEW.attempt_id AND d.assistant_message_id=NEW.assistant_message_id
              AND d.user_message_id=NEW.user_message_id AND d.mixin_snapshot_schema_version=1);
END;
CREATE TABLE console_prompt_mixin_attempt_items (
    conversation_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
    snapshot_id TEXT NOT NULL,
    PRIMARY KEY(conversation_id, attempt_id, ordinal),
    FOREIGN KEY(conversation_id, attempt_id)
        REFERENCES console_prompt_mixin_attempts(conversation_id, attempt_id) ON DELETE CASCADE,
    FOREIGN KEY(conversation_id, snapshot_id)
        REFERENCES console_prompt_mixin_revisions(conversation_id, snapshot_id)
);
CREATE TRIGGER console_prompt_mixin_item_immutable
BEFORE UPDATE ON console_prompt_mixin_attempt_items BEGIN
    SELECT RAISE(ABORT, 'mixin_immutable_item');
END;
ALTER TABLE console_dispatch_checkpoints ADD COLUMN mixin_snapshot_schema_version INTEGER;
CREATE TRIGGER console_prompt_mixin_message_marker_preserve
AFTER UPDATE OF metadata_json ON messages
WHEN json_valid(OLD.metadata_json)
 AND json_type(OLD.metadata_json, '$.mixin_snapshot_schema_version') IS NOT NULL
 AND (NOT json_valid(NEW.metadata_json)
      OR json_extract(NEW.metadata_json, '$.mixin_attempt_id') IS NOT json_extract(OLD.metadata_json, '$.mixin_attempt_id')
      OR json_extract(NEW.metadata_json, '$.mixin_snapshot_schema_version') IS NOT json_extract(OLD.metadata_json, '$.mixin_snapshot_schema_version'))
 AND NOT EXISTS (
    SELECT 1 FROM console_prompt_mixin_attempts AS successor
    JOIN console_prompt_mixin_attempts AS predecessor
      ON predecessor.conversation_id=successor.conversation_id
     AND predecessor.attempt_id=successor.retry_of_attempt_id
    WHERE successor.conversation_id=NEW.conversation_id
      AND successor.assistant_message_id=NEW.id
      AND predecessor.assistant_message_id=NEW.id
      AND predecessor.user_message_id=successor.user_message_id
      AND successor.attempt_id=json_extract(NEW.metadata_json, '$.mixin_attempt_id')
      AND predecessor.attempt_id=json_extract(OLD.metadata_json, '$.mixin_attempt_id')
      AND json_type(NEW.metadata_json, '$.mixin_snapshot_schema_version')='integer'
      AND json_extract(NEW.metadata_json, '$.mixin_snapshot_schema_version')=1
 )
BEGIN
    UPDATE messages SET metadata_json=json_set(CASE WHEN json_type(NEW.metadata_json) = 'object' THEN NEW.metadata_json ELSE '{}' END,
        '$.mixin_attempt_id', json_extract(OLD.metadata_json, '$.mixin_attempt_id'),
        '$.mixin_snapshot_schema_version', json_extract(OLD.metadata_json, '$.mixin_snapshot_schema_version'))
    WHERE id=NEW.id;
END;
-- Soft deletion already removed the title; purge must not remove it twice.
DROP TRIGGER conversations_ad;
CREATE TRIGGER conversations_ad
AFTER DELETE ON conversations BEGIN
  INSERT INTO conversations_fts(conversations_fts,rowid,title)
  SELECT 'delete',old.rowid,old.title
  WHERE old.deleted = 0 AND old.title IS NOT NULL;
END;
