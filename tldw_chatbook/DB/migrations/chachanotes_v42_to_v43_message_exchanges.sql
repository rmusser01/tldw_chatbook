-- ChaChaNotes v42 -> v43: local-only per-message exchange captures
-- (Console Conversation Inspector, task-18300).
--
-- Regenerate/rewind can leave more than one captured run behind a single
-- message row; this table holds every run (raw capture bytes) keyed by
-- (message_id, run_tag, seq) so the Inspector can render exchange history
-- without collapsing it into the one "winning" message row.
--
-- LOCAL-ONLY: no sync trigger references this table and rows here are
-- never added to any sync_log payload -- same rule as v18/v19
-- message_attachments and v29/v30 usage_json.
--
-- DDL only. The schema-version bump is done separately in the runner
-- (`CharactersRAGDB._migrate_from_v40_to_v41`), guarded by a rowcount
-- check on the UPDATE, matching the v29->v30 usage_json precedent.

CREATE TABLE IF NOT EXISTS message_exchanges(
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  message_id   TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
  run_tag      TEXT NOT NULL,
  seq          INTEGER NOT NULL,
  status       TEXT NOT NULL,
  abandoned    BOOLEAN NOT NULL DEFAULT 0,
  capture_blob BLOB NOT NULL,
  created_at   TEXT NOT NULL,
  UNIQUE(message_id, run_tag, seq)
);

CREATE INDEX IF NOT EXISTS idx_message_exchanges_message
  ON message_exchanges(message_id);
