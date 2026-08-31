-- V58 -> V59: content-free receipts for atomic Notes organization saves.

CREATE TABLE note_organization_receipts(
  receipt_id TEXT PRIMARY KEY,
  note_id TEXT NOT NULL REFERENCES notes(id),
  requested_folder_name TEXT,
  requested_folder_sync_id TEXT,
  requested_keywords_json TEXT NOT NULL,
  review_id TEXT,
  collision_ids_json TEXT NOT NULL DEFAULT '[]',
  note_version INTEGER NOT NULL,
  organization_version TEXT NOT NULL CHECK(
    length(organization_version) = 64
    AND organization_version = lower(organization_version)
    AND organization_version NOT GLOB '*[^0-9a-f]*'
  ),
  state TEXT NOT NULL CHECK(state IN (
    'pending_organization', 'placement_review'
  )),
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  CHECK(
    (state = 'pending_organization'
      AND review_id IS NULL
      AND collision_ids_json = '[]')
    OR (state = 'placement_review'
      AND review_id IS NOT NULL
      AND collision_ids_json <> '[]')
  )
);

CREATE UNIQUE INDEX uq_note_organization_receipts_unresolved_note
  ON note_organization_receipts(note_id);

CREATE INDEX idx_notes_organization_heads_note_subject
  ON notes_organization_heads(
    CASE
      WHEN domain = 'notes.folder_link'
        THEN json_extract(payload_json, '$.note_id')
      WHEN domain = 'notes.keyword_link'
        AND json_extract(payload_json, '$.subject_type') = 'note'
        THEN json_extract(payload_json, '$.subject_id')
    END,
    domain,
    server_profile_id,
    dataset_id,
    object_id
  )
  WHERE domain = 'notes.folder_link'
     OR (domain = 'notes.keyword_link'
         AND json_extract(payload_json, '$.subject_type') = 'note');

CREATE INDEX idx_notes_organization_intents_note_subject_latest
  ON notes_organization_sync_intents(
    CASE
      WHEN domain = 'notes.folder_link'
        THEN json_extract(payload_json, '$.note_id')
      WHEN domain = 'notes.keyword_link'
        AND json_extract(payload_json, '$.subject_type') = 'note'
        THEN json_extract(payload_json, '$.subject_id')
    END,
    server_profile_id,
    dataset_id,
    domain,
    object_id,
    intent_sequence DESC
  )
  WHERE domain = 'notes.folder_link'
     OR (domain = 'notes.keyword_link'
         AND json_extract(payload_json, '$.subject_type') = 'note');
