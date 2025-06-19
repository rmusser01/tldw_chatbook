-- Migration: Add file sync fields to notes table
-- This migration adds fields necessary for bi-directional file synchronization

-- Add sync-related columns to the notes table
ALTER TABLE notes ADD COLUMN file_path_on_disk TEXT UNIQUE;
ALTER TABLE notes ADD COLUMN relative_file_path_on_disk TEXT;
ALTER TABLE notes ADD COLUMN sync_root_folder TEXT;
ALTER TABLE notes ADD COLUMN last_synced_disk_file_hash TEXT;
ALTER TABLE notes ADD COLUMN last_synced_disk_file_mtime REAL;
ALTER TABLE notes ADD COLUMN is_externally_synced BOOLEAN NOT NULL DEFAULT 0;
ALTER TABLE notes ADD COLUMN sync_strategy TEXT DEFAULT NULL CHECK(sync_strategy IN ('disk_to_db', 'db_to_disk', 'bidirectional'));
ALTER TABLE notes ADD COLUMN sync_excluded BOOLEAN NOT NULL DEFAULT 0;
ALTER TABLE notes ADD COLUMN file_extension TEXT DEFAULT '.md';

-- Add indexes for efficient sync operations
CREATE INDEX IF NOT EXISTS idx_notes_file_path_on_disk ON notes(file_path_on_disk) WHERE file_path_on_disk IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_notes_sync_root_folder ON notes(sync_root_folder) WHERE sync_root_folder IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_notes_is_externally_synced ON notes(is_externally_synced);
CREATE INDEX IF NOT EXISTS idx_notes_sync_excluded ON notes(sync_excluded);

-- Create sync_sessions table to track sync operations
CREATE TABLE IF NOT EXISTS sync_sessions(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT UNIQUE NOT NULL,  -- UUID for the sync session
  sync_root_folder TEXT NOT NULL,
  sync_direction TEXT NOT NULL CHECK(sync_direction IN ('disk_to_db', 'db_to_disk', 'bidirectional')),
  conflict_resolution TEXT NOT NULL CHECK(conflict_resolution IN ('ask', 'disk_wins', 'db_wins', 'newer_wins')),
  started_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  completed_at DATETIME,
  status TEXT NOT NULL CHECK(status IN ('running', 'completed', 'failed', 'cancelled')),
  total_files INTEGER DEFAULT 0,
  processed_files INTEGER DEFAULT 0,
  conflicts_found INTEGER DEFAULT 0,
  errors_count INTEGER DEFAULT 0,
  client_id TEXT NOT NULL,
  summary TEXT  -- JSON summary of the sync operation
);

CREATE INDEX IF NOT EXISTS idx_sync_sessions_status ON sync_sessions(status);
CREATE INDEX IF NOT EXISTS idx_sync_sessions_started ON sync_sessions(started_at);

-- Create sync_conflicts table to track unresolved conflicts
CREATE TABLE IF NOT EXISTS sync_conflicts(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL REFERENCES sync_sessions(session_id),
  note_id TEXT REFERENCES notes(id),
  file_path TEXT NOT NULL,
  conflict_type TEXT NOT NULL CHECK(conflict_type IN ('both_changed', 'deleted_on_disk', 'deleted_in_db')),
  db_content_hash TEXT,
  disk_content_hash TEXT,
  db_modified_time DATETIME,
  disk_modified_time REAL,
  resolution TEXT CHECK(resolution IN ('use_db', 'use_disk', 'merge', 'skip', NULL)),
  resolved_at DATETIME,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sync_conflicts_session ON sync_conflicts(session_id);
CREATE INDEX IF NOT EXISTS idx_sync_conflicts_note ON sync_conflicts(note_id);
CREATE INDEX IF NOT EXISTS idx_sync_conflicts_resolution ON sync_conflicts(resolution);