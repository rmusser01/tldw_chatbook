BEGIN IMMEDIATE;

DELETE FROM workspace_memberships
WHERE item_type = 'note' AND role = 'note_pending';

CREATE TABLE research_quick_note_receipts (
    receipt_id TEXT PRIMARY KEY CHECK (length(trim(receipt_id)) BETWEEN 1 AND 1024),
    data_source TEXT NOT NULL DEFAULT 'local' CHECK (data_source = 'local'),
    workspace_id TEXT NOT NULL CHECK (length(trim(workspace_id)) BETWEEN 1 AND 1024),
    local_user_id TEXT NOT NULL CHECK (length(trim(local_user_id)) BETWEEN 1 AND 1024),
    operation_token TEXT NOT NULL CHECK (length(trim(operation_token)) BETWEEN 1 AND 1024),
    operation_kind TEXT NOT NULL CHECK (operation_kind IN ('create', 'delete')),
    canonical_note_id TEXT NOT NULL CHECK (length(trim(canonical_note_id)) BETWEEN 1 AND 1024),
    expected_version INTEGER DEFAULT NULL CHECK (
        (operation_kind = 'create' AND expected_version IS NULL)
        OR
        (operation_kind = 'delete'
         AND expected_version IS NOT NULL
         AND expected_version >= 1)
    ),
    state TEXT NOT NULL DEFAULT 'pending' CHECK (
        state IN ('pending', 'owner_committed')
    ),
    revision INTEGER NOT NULL DEFAULT 1 CHECK (
        (state = 'pending' AND revision = 1)
        OR (state = 'owner_committed' AND revision >= 2)
    ),
    created_at TEXT NOT NULL CHECK (length(trim(created_at)) BETWEEN 1 AND 128),
    updated_at TEXT NOT NULL CHECK (length(trim(updated_at)) BETWEEN 1 AND 128),
    FOREIGN KEY(workspace_id)
        REFERENCES workspace_records(workspace_id)
        ON DELETE CASCADE,
    UNIQUE(workspace_id, local_user_id, operation_token, operation_kind)
);

CREATE INDEX idx_research_quick_note_receipts_reconcile
ON research_quick_note_receipts (
    local_user_id,
    state,
    updated_at,
    receipt_id
);

CREATE INDEX idx_research_quick_note_receipts_owner
ON research_quick_note_receipts (
    workspace_id,
    local_user_id,
    operation_kind,
    canonical_note_id
);

INSERT OR IGNORE INTO schema_version (version) VALUES (4);

COMMIT;
