-- ADR-166: unknown historical archive episodes remain undated.
ALTER TABLE workspace_records ADD COLUMN archived_at TEXT;
ALTER TABLE workspace_records ADD COLUMN archive_revision INTEGER NOT NULL DEFAULT 0
    CHECK (archive_revision >= 0);
