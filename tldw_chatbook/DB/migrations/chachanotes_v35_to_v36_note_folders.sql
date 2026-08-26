PRAGMA foreign_keys = ON;

CREATE TABLE note_folders(
  id              TEXT PRIMARY KEY,
  parent_id       TEXT REFERENCES note_folders(id),
  name            TEXT NOT NULL,
  normalized_name TEXT NOT NULL,
  path            TEXT NOT NULL,
  normalized_path TEXT NOT NULL,
  version         INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
  deleted         INTEGER NOT NULL DEFAULT 0 CHECK(deleted IN (0, 1)),
  created_at      TEXT NOT NULL,
  modified_at     TEXT NOT NULL,
  CHECK(parent_id IS NULL OR parent_id <> id)
);
CREATE UNIQUE INDEX uq_note_folders_active_normalized_path
  ON note_folders(normalized_path) WHERE deleted = 0;
CREATE INDEX idx_note_folders_active_parent
  ON note_folders(parent_id, normalized_name) WHERE deleted = 0;

CREATE TABLE note_folder_memberships(
  id              TEXT PRIMARY KEY,
  folder_id       TEXT NOT NULL REFERENCES note_folders(id),
  note_id         TEXT NOT NULL REFERENCES notes(id),
  ownership       TEXT NOT NULL CHECK(ownership IN ('manual', 'managed')),
  owner_id        TEXT NOT NULL DEFAULT '',
  owner_active    INTEGER NOT NULL DEFAULT 1 CHECK(owner_active IN (0, 1)),
  version         INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
  deleted         INTEGER NOT NULL DEFAULT 0 CHECK(deleted IN (0, 1)),
  created_at      TEXT NOT NULL,
  modified_at     TEXT NOT NULL,
  CHECK(
    (ownership = 'manual' AND owner_id = '' AND owner_active = 1) OR
    (ownership = 'managed' AND length(owner_id) > 0)
  )
);
CREATE UNIQUE INDEX uq_note_folder_memberships_active_owner
  ON note_folder_memberships(folder_id, note_id, ownership, owner_id)
  WHERE deleted = 0;
CREATE INDEX idx_note_folder_memberships_active_folder
  ON note_folder_memberships(folder_id, note_id) WHERE deleted = 0;
CREATE INDEX idx_note_folder_memberships_active_note
  ON note_folder_memberships(note_id, folder_id) WHERE deleted = 0;
CREATE INDEX idx_note_folder_memberships_restore_review
  ON note_folder_memberships(owner_active, owner_id)
  WHERE deleted = 0 AND ownership = 'managed';
CREATE INDEX idx_note_folder_memberships_managed_owner
  ON note_folder_memberships(
    owner_id, deleted, folder_id, note_id, owner_active
  ) WHERE ownership = 'managed';

UPDATE db_schema_version SET version = 36
 WHERE schema_name = 'rag_char_chat_schema' AND version = 35;
