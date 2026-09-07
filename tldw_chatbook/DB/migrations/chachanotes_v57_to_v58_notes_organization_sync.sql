-- V57 -> V58: portable identities and durable Notes organization sync state.
--
-- Portable organization stores logical identifiers and protocol state only.
-- Filesystem paths, note content, secrets, and cross-database outbox rows do
-- not belong in these tables.

ALTER TABLE keywords ADD COLUMN sync_id TEXT;
ALTER TABLE keyword_collections ADD COLUMN sync_id TEXT;
ALTER TABLE note_folders ADD COLUMN sync_id TEXT;

CREATE TABLE notes_organization_sync_intents(
  intent_id TEXT PRIMARY KEY,
  intent_sequence INTEGER NOT NULL,
  predecessor_intent_id TEXT REFERENCES notes_organization_sync_intents(intent_id),
  server_profile_id TEXT NOT NULL,
  dataset_id TEXT NOT NULL,
  domain TEXT NOT NULL CHECK(domain IN (
    'notes.keyword', 'notes.keyword_link',
    'notes.keyword_collection', 'notes.keyword_collection_link',
    'notes.folder', 'notes.folder_link'
  )),
  object_id TEXT NOT NULL,
  operation TEXT NOT NULL CHECK(operation IN ('upsert', 'tombstone')),
  schema_version INTEGER NOT NULL CHECK(schema_version = 1),
  encryption_policy TEXT NOT NULL CHECK(encryption_policy = 'server_trusted_v1'),
  payload_json TEXT NOT NULL,
  payload_hash TEXT NOT NULL CHECK(
    length(payload_hash) = 64 AND payload_hash = lower(payload_hash)
    AND payload_hash NOT GLOB '*[^0-9a-f]*'
  ),
  routing_metadata_json TEXT NOT NULL DEFAULT '{}',
  base_server_cursor TEXT,
  base_object_revision INTEGER,
  base_object_hash TEXT,
  dependency_refs_json TEXT NOT NULL DEFAULT '[]',
  source_version INTEGER NOT NULL CHECK(source_version >= 1),
  created_at TEXT NOT NULL,
  outbox_client_envelope_id TEXT,
  copied_at TEXT,
  acknowledged_at TEXT,
  CHECK(
    (base_server_cursor IS NULL) = (base_object_revision IS NULL)
    AND (base_object_revision IS NULL) = (base_object_hash IS NULL)
  ),
  CHECK(outbox_client_envelope_id IS NULL OR outbox_client_envelope_id = intent_id),
  UNIQUE(server_profile_id, dataset_id, intent_sequence),
  UNIQUE(server_profile_id, dataset_id, domain, object_id, source_version, operation)
);
CREATE INDEX idx_notes_organization_intents_pending
  ON notes_organization_sync_intents(server_profile_id, dataset_id, intent_sequence)
  WHERE acknowledged_at IS NULL;

CREATE TABLE notes_organization_heads(
  server_profile_id TEXT NOT NULL,
  dataset_id TEXT NOT NULL,
  domain TEXT NOT NULL CHECK(domain IN (
    'notes.keyword', 'notes.keyword_link',
    'notes.keyword_collection', 'notes.keyword_collection_link',
    'notes.folder', 'notes.folder_link'
  )),
  object_id TEXT NOT NULL,
  operation TEXT NOT NULL CHECK(operation IN ('upsert', 'tombstone')),
  schema_version INTEGER NOT NULL CHECK(schema_version = 1),
  encryption_policy TEXT NOT NULL CHECK(encryption_policy = 'server_trusted_v1'),
  payload_json TEXT NOT NULL,
  payload_hash TEXT NOT NULL CHECK(
    length(payload_hash) = 64 AND payload_hash = lower(payload_hash)
    AND payload_hash NOT GLOB '*[^0-9a-f]*'
  ),
  object_revision INTEGER NOT NULL CHECK(object_revision >= 1),
  object_hash TEXT NOT NULL CHECK(
    length(object_hash) = 64 AND object_hash = lower(object_hash)
    AND object_hash NOT GLOB '*[^0-9a-f]*'
  ),
  server_cursor TEXT NOT NULL,
  deleted INTEGER NOT NULL CHECK(deleted IN (0, 1)),
  apply_state TEXT NOT NULL CHECK(apply_state IN ('pending', 'applied', 'blocked')),
  applied_at TEXT,
  updated_at TEXT NOT NULL,
  PRIMARY KEY(server_profile_id, dataset_id, domain, object_id),
  CHECK(
    (operation = 'upsert' AND deleted = 0)
    OR (operation = 'tombstone' AND deleted = 1)
  ),
  CHECK(
    (apply_state = 'applied' AND applied_at IS NOT NULL)
    OR (apply_state <> 'applied' AND applied_at IS NULL)
  )
);
CREATE INDEX idx_notes_organization_heads_cursor
  ON notes_organization_heads(server_profile_id, dataset_id, server_cursor);

CREATE TABLE notes_organization_sync_checkpoints(
  server_profile_id TEXT NOT NULL,
  dataset_id TEXT NOT NULL,
  local_state TEXT NOT NULL CHECK(local_state IN (
    'local_only', 'initializing', 'pulling', 'adoption_review', 'ready', 'failed'
  )),
  server_state TEXT NOT NULL CHECK(server_state IN (
    'unknown', 'absent', 'initializing', 'ready', 'failed', 'incompatible'
  )),
  bootstrap_id TEXT,
  captured_count INTEGER NOT NULL DEFAULT 0 CHECK(captured_count >= 0),
  expected_count INTEGER NOT NULL DEFAULT 0 CHECK(expected_count >= 0),
  error_code TEXT,
  pull_cursor TEXT,
  inventory_phase TEXT NOT NULL DEFAULT 'not_started' CHECK(inventory_phase IN (
    'not_started', 'resources', 'links', 'tombstones', 'complete'
  )),
  last_inventory_key TEXT,
  updated_at TEXT NOT NULL,
  PRIMARY KEY(server_profile_id, dataset_id),
  CHECK(captured_count <= expected_count),
  CHECK(
    local_state <> 'ready'
    OR (server_state = 'ready' AND inventory_phase = 'complete' AND error_code IS NULL)
  )
);

CREATE TABLE notes_organization_adoption_reviews(
  review_id TEXT PRIMARY KEY,
  server_profile_id TEXT NOT NULL,
  dataset_id TEXT NOT NULL,
  domain TEXT NOT NULL CHECK(domain IN (
    'notes.keyword', 'notes.keyword_collection', 'notes.folder'
  )),
  local_object_id TEXT NOT NULL,
  remote_object_id TEXT,
  collision_key TEXT NOT NULL,
  display_name TEXT NOT NULL,
  portable_path TEXT,
  state TEXT NOT NULL CHECK(state IN ('open', 'resolved')),
  resolution TEXT CHECK(resolution IN ('merge', 'rename_local', 'keep_local')),
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  resolved_at TEXT,
  UNIQUE(server_profile_id, dataset_id, domain, local_object_id),
  CHECK(
    (state = 'open' AND resolution IS NULL AND resolved_at IS NULL)
    OR (state = 'resolved' AND resolution IS NOT NULL AND resolved_at IS NOT NULL)
  )
);
CREATE INDEX idx_notes_organization_adoption_reviews_open
  ON notes_organization_adoption_reviews(server_profile_id, dataset_id, domain, created_at)
  WHERE state = 'open';

CREATE TABLE note_folder_sync_suppressions(
  note_id TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE ON UPDATE CASCADE,
  folder_sync_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY(note_id, folder_sync_id)
);
