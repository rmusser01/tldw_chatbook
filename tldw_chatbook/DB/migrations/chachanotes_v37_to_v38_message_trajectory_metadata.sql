-- Migration: ChaChaNotes V37 to V38 message trajectory metadata sidecar.
--
-- The trajectory ledger is a LOCAL-ONLY projection of per-turn step timing:
-- it records what THIS device observed while producing messages (event
-- kinds, first-token/completion timestamps, model/provider, step payloads).
-- It is deliberately excluded from sync triggers and sync serialization;
-- every device rebuilds its own copy from its own runs.

CREATE TABLE IF NOT EXISTS message_trajectory_metadata (
    message_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    turn_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    event_kind TEXT NOT NULL,
    step_started_at REAL,
    first_token_at REAL,
    completed_at REAL,
    model TEXT,
    provider TEXT,
    payload_json TEXT,
    PRIMARY KEY (message_id, event_kind, seq),
    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
);

-- Ledger-ordering guarantee: seq is strictly unique per conversation.
CREATE UNIQUE INDEX IF NOT EXISTS idx_message_trajectory_conv_seq
    ON message_trajectory_metadata (conversation_id, seq);

CREATE INDEX IF NOT EXISTS idx_message_trajectory_turn
    ON message_trajectory_metadata (conversation_id, turn_id, seq);

CREATE INDEX IF NOT EXISTS idx_message_trajectory_msg
    ON message_trajectory_metadata (message_id);

UPDATE db_schema_version
   SET version = 38
 WHERE schema_name = 'rag_char_chat_schema'
   AND version = 37;
