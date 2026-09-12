ALTER TABLE message_exchanges
ADD COLUMN capture_detail TEXT NOT NULL DEFAULT 'safe'
CHECK (capture_detail IN ('safe', 'full'));

CREATE INDEX idx_message_exchanges_capture_detail
ON message_exchanges(capture_detail, message_id);

CREATE TABLE console_conversation_capture_policy(
  conversation_id TEXT PRIMARY KEY NOT NULL
    REFERENCES conversations(id) ON DELETE CASCADE,
  capture_detail TEXT NOT NULL CHECK (capture_detail IN ('safe', 'full')),
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
