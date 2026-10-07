-- Pending agent reports belong to their saved chat, not execution authority.
CREATE TABLE fleet_progress_messages (
  sequence INTEGER PRIMARY KEY AUTOINCREMENT,
  message_id TEXT NOT NULL UNIQUE CHECK(length(message_id) BETWEEN 1 AND 128),
  conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  handle_id TEXT NOT NULL CHECK(length(handle_id) BETWEEN 1 AND 128),
  run_id TEXT NOT NULL CHECK(length(run_id) BETWEEN 1 AND 128),
  parent_run_id TEXT NOT NULL CHECK(length(parent_run_id) BETWEEN 1 AND 128),
  chain_id TEXT CHECK(chain_id IS NULL OR length(chain_id) BETWEEN 1 AND 128),
  agent TEXT NOT NULL CHECK(length(agent) BETWEEN 1 AND 80),
  body TEXT NOT NULL CHECK(length(body) BETWEEN 1 AND 2000),
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_fleet_progress_conversation_sequence
  ON fleet_progress_messages(conversation_id, sequence);
