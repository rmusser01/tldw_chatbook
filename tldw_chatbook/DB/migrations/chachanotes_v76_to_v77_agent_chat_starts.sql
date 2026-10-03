-- Native agent-chat-start receipts. Local-only operational ownership.
CREATE TABLE console_dispatch_checkpoints_v77 (
    assistant_message_id TEXT PRIMARY KEY
        REFERENCES messages(id) ON DELETE CASCADE,
    user_message_id TEXT NOT NULL
        REFERENCES messages(id) ON DELETE CASCADE,
    conversation_id TEXT NOT NULL
        REFERENCES conversations(id) ON DELETE CASCADE,
    schema_version INTEGER NOT NULL DEFAULT 1
        CHECK(schema_version > 0),
    preparation_id TEXT NOT NULL UNIQUE,
    attempt_id TEXT NOT NULL,
    state TEXT NOT NULL
        CHECK(state IN ('accepted', 'dispatch_started')),
    checkpoint_revision INTEGER NOT NULL DEFAULT 1
        CHECK(checkpoint_revision > 0),
    user_message_version INTEGER NOT NULL
        CHECK(user_message_version > 0),
    assistant_message_version INTEGER NOT NULL
        CHECK(assistant_message_version > 0),
    origin TEXT NOT NULL CHECK(origin IN ('manual', 'queued', 'agent_chat_start')),
    queue_entry_id TEXT,
    agent_chat_start_attempt_id TEXT UNIQUE,
    frozen_authority_json TEXT NOT NULL,
    resolved_destination_json TEXT NOT NULL,
    reconstructability_json TEXT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK ((origin = 'queued' AND queue_entry_id IS NOT NULL)
        OR (origin IN ('manual', 'agent_chat_start') AND queue_entry_id IS NULL)),
    CHECK ((origin = 'agent_chat_start' AND agent_chat_start_attempt_id IS NOT NULL
            AND length(agent_chat_start_attempt_id) BETWEEN 1 AND 200)
        OR (origin IN ('manual', 'queued') AND agent_chat_start_attempt_id IS NULL))
);

INSERT INTO console_dispatch_checkpoints_v77 (assistant_message_id, user_message_id, conversation_id, schema_version, preparation_id, attempt_id, state, checkpoint_revision, user_message_version, assistant_message_version, origin, queue_entry_id, frozen_authority_json, resolved_destination_json, reconstructability_json, created_at, updated_at)
SELECT assistant_message_id, user_message_id, conversation_id, schema_version, preparation_id, attempt_id, state, checkpoint_revision, user_message_version, assistant_message_version, origin, queue_entry_id, frozen_authority_json, resolved_destination_json, reconstructability_json, created_at, updated_at FROM console_dispatch_checkpoints;
DROP TABLE console_dispatch_checkpoints;
ALTER TABLE console_dispatch_checkpoints_v77 RENAME TO console_dispatch_checkpoints;
CREATE INDEX idx_console_dispatch_checkpoint_conversation
    ON console_dispatch_checkpoints(conversation_id);
CREATE INDEX idx_console_dispatch_checkpoints_user_message
    ON console_dispatch_checkpoints(user_message_id);
