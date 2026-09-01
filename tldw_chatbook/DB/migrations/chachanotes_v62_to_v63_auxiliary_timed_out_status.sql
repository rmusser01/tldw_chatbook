-- TASK-26016 (review Critical): AuxiliaryAttemptStatus.TIMED_OUT was added
-- to the Python enum but the v33 CHECK constraint on
-- console_auxiliary_attempts.status still rejected 'timed_out', so every
-- real compaction timeout raised IntegrityError from inside the timeout
-- handler and left the ledger row stuck at 'started'. SQLite cannot alter
-- a CHECK in place; rebuild the table with the widened constraint.

ALTER TABLE console_auxiliary_attempts
RENAME TO console_auxiliary_attempts_v62;

CREATE TABLE console_auxiliary_attempts(
  operation_id             TEXT PRIMARY KEY NOT NULL,
  conversation_id          TEXT NOT NULL
                               REFERENCES conversations(id)
                               ON DELETE CASCADE ON UPDATE CASCADE,
  purpose                  TEXT NOT NULL,
  provider                 TEXT NOT NULL,
  model                    TEXT NOT NULL,
  requested_output_cap     INTEGER NOT NULL CHECK(requested_output_cap > 0),
  estimated_input_tokens   INTEGER NOT NULL CHECK(estimated_input_tokens >= 0),
  status                   TEXT NOT NULL
                                CHECK(status IN ('started','succeeded','failed','cancelled','stale','timed_out')),
  started_at               DATETIME NOT NULL,
  finished_at              DATETIME,
  elapsed_ms               INTEGER CHECK(elapsed_ms >= 0),
  pricing_provenance_json  TEXT,
  provider_usage_json      TEXT
);

INSERT INTO console_auxiliary_attempts(
  operation_id, conversation_id, purpose, provider, model,
  requested_output_cap, estimated_input_tokens, status, started_at,
  finished_at, elapsed_ms, pricing_provenance_json, provider_usage_json
)
SELECT
  operation_id, conversation_id, purpose, provider, model,
  requested_output_cap, estimated_input_tokens, status, started_at,
  finished_at, elapsed_ms, pricing_provenance_json, provider_usage_json
FROM console_auxiliary_attempts_v62;

DROP TABLE console_auxiliary_attempts_v62;

CREATE INDEX IF NOT EXISTS idx_console_aux_attempts_conversation_started
  ON console_auxiliary_attempts(conversation_id, started_at DESC);
