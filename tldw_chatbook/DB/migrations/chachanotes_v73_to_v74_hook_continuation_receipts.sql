-- Body-free, device-local Stop admission identities. Terminal checkpoints may
-- disappear; recoverable parent history must continue to suppress duplicates.
CREATE TABLE console_hook_continuation_receipts (
    parent_turn_id TEXT NOT NULL CHECK(length(parent_turn_id) > 0),
    stop_event_id TEXT NOT NULL CHECK(length(stop_event_id) > 0),
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    parent_assistant_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    assistant_message_id TEXT NOT NULL,
    chain_id TEXT NOT NULL CHECK(length(chain_id) > 0),
    admitted_turns INTEGER NOT NULL CHECK(admitted_turns BETWEEN 1 AND 3),
    initiator TEXT NOT NULL CHECK(initiator = 'hook_continuation'),
    PRIMARY KEY(parent_turn_id, stop_event_id)
);
CREATE INDEX idx_hook_continuation_receipts_conversation
    ON console_hook_continuation_receipts(conversation_id);
