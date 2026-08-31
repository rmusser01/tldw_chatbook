-- V59 -> V60: scoped immutable Notes publication intents.
--
-- The encrypted general-outbox projection may live in another database, so
-- the immutable source payload remains Notes-owned until that projection is
-- accepted. Organization receipts remain owned by the shipped v59 schema.

CREATE TABLE note_sync_publication_intents(
  intent_id TEXT PRIMARY KEY,
  server_profile_id TEXT NOT NULL,
  dataset_id TEXT NOT NULL,
  note_id TEXT NOT NULL,
  operation TEXT NOT NULL CHECK(operation IN ('create', 'update', 'delete')),
  base_version INTEGER,
  entity_version INTEGER NOT NULL CHECK(entity_version >= 1),
  request_fingerprint TEXT NOT NULL CHECK(
    length(request_fingerprint) = 64
    AND request_fingerprint = lower(request_fingerprint)
    AND request_fingerprint NOT GLOB '*[^0-9a-f]*'
  ),
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  outbox_client_envelope_id TEXT,
  copied_at TEXT,
  acknowledged_at TEXT,
  cancelled_at TEXT,
  UNIQUE(
    server_profile_id, dataset_id, note_id, entity_version, operation
  )
);

CREATE INDEX idx_note_sync_publication_intents_pending
  ON note_sync_publication_intents(
    server_profile_id, dataset_id, note_id, entity_version, intent_id
  )
  WHERE acknowledged_at IS NULL AND cancelled_at IS NULL;
