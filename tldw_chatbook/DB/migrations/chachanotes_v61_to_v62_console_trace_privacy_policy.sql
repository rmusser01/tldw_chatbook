ALTER TABLE console_conversation_capture_policy
RENAME TO console_conversation_capture_policy_v57;

CREATE TABLE console_conversation_capture_policy(
  conversation_id TEXT PRIMARY KEY NOT NULL
    REFERENCES conversations(id) ON DELETE CASCADE,
  capture_detail TEXT NULL CHECK (capture_detail IN ('safe', 'full')),
  capture_enabled INTEGER NULL CHECK (capture_enabled IN (0, 1)),
  pii_redaction_enabled INTEGER NULL CHECK (pii_redaction_enabled IN (0, 1)),
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CHECK (
    capture_detail IS NOT NULL
    OR capture_enabled IS NOT NULL
    OR pii_redaction_enabled IS NOT NULL
  )
);

INSERT INTO console_conversation_capture_policy(
  conversation_id,
  capture_detail,
  updated_at
)
SELECT conversation_id, capture_detail, updated_at
FROM console_conversation_capture_policy_v57;

DROP TABLE console_conversation_capture_policy_v57;
