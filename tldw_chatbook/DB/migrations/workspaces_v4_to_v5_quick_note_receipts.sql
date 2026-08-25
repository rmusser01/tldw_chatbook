BEGIN IMMEDIATE;

-- Early development builds briefly projected blank legacy note_pending rows
-- before their canonical Notes owner could be verified.  Quarantine that
-- unreleased representation, then replace the proof-less receipt format rather
-- than guessing ownership during a Workspace-only migration.
DELETE FROM workspace_memberships AS membership
WHERE membership.item_type = 'note'
  AND membership.role = 'note'
  AND membership.item_id LIKE 'research-note-%'
  AND trim(membership.title) = '';

DROP TABLE research_quick_note_receipts;

CREATE TABLE research_quick_note_receipts (
    receipt_id TEXT PRIMARY KEY CHECK (length(trim(receipt_id)) BETWEEN 1 AND 1024),
    data_source TEXT NOT NULL DEFAULT 'local' CHECK (data_source = 'local'),
    server_profile_id TEXT NOT NULL DEFAULT '' CHECK (server_profile_id = ''),
    principal_id TEXT NOT NULL DEFAULT '' CHECK (principal_id = ''),
    workspace_id TEXT NOT NULL CHECK (length(trim(workspace_id)) BETWEEN 1 AND 1024),
    local_user_id TEXT NOT NULL CHECK (length(trim(local_user_id)) BETWEEN 1 AND 1024),
    operation_token TEXT NOT NULL CHECK (length(trim(operation_token)) BETWEEN 1 AND 1024),
    operation_kind TEXT NOT NULL CHECK (operation_kind IN ('create', 'delete')),
    canonical_note_id TEXT NOT NULL CHECK (length(trim(canonical_note_id)) BETWEEN 1 AND 1024),
    owner_proof TEXT NOT NULL CHECK (length(trim(owner_proof)) BETWEEN 32 AND 256),
    lease_token TEXT NOT NULL CHECK (length(trim(lease_token)) BETWEEN 32 AND 256),
    lease_expires_at TEXT NOT NULL CHECK (length(trim(lease_expires_at)) BETWEEN 1 AND 128),
    expected_version INTEGER DEFAULT NULL CHECK (
        (operation_kind = 'create' AND expected_version IS NULL)
        OR
        (operation_kind = 'delete'
         AND expected_version IS NOT NULL
         AND expected_version >= 1)
    ),
    state TEXT NOT NULL DEFAULT 'pending' CHECK (
        state IN ('pending', 'owner_committed', 'blocked')
    ),
    revision INTEGER NOT NULL DEFAULT 1 CHECK (
        (state = 'pending' AND revision >= 1)
        OR (state IN ('owner_committed', 'blocked') AND revision >= 2)
    ),
    failure_count INTEGER NOT NULL DEFAULT 0 CHECK (failure_count BETWEEN 0 AND 3),
    next_retry_at TEXT NOT NULL CHECK (length(trim(next_retry_at)) BETWEEN 1 AND 128),
    blocked_reason_code TEXT NOT NULL DEFAULT '' CHECK (
        blocked_reason_code IN (
            '', 'proof_mismatch', 'owner_conflict',
            'owner_unavailable', 'registry_failure'
        )
    ),
    created_at TEXT NOT NULL CHECK (length(trim(created_at)) BETWEEN 1 AND 128),
    updated_at TEXT NOT NULL CHECK (length(trim(updated_at)) BETWEEN 1 AND 128),
    CHECK (state <> 'blocked' OR blocked_reason_code <> ''),
    CHECK (
        julianday(created_at) IS NOT NULL
        AND julianday(updated_at) IS NOT NULL
        AND julianday(lease_expires_at) IS NOT NULL
        AND julianday(next_retry_at) IS NOT NULL
        AND julianday(updated_at) >= julianday(created_at)
        AND julianday(lease_expires_at) >= julianday(created_at)
        AND julianday(next_retry_at) >= julianday(created_at)
    ),
    FOREIGN KEY(workspace_id)
        REFERENCES workspace_records(workspace_id)
        ON DELETE CASCADE,
    UNIQUE(
        data_source, server_profile_id, principal_id, workspace_id,
        local_user_id, operation_token, operation_kind
    )
);

CREATE INDEX idx_research_quick_note_receipts_reconcile
ON research_quick_note_receipts (
    local_user_id,
    state,
    next_retry_at,
    lease_expires_at,
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

INSERT OR IGNORE INTO schema_version (version) VALUES (5);

COMMIT;
