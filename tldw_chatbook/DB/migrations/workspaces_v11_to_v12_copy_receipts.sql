CREATE TABLE IF NOT EXISTS research_copy_operations (
    transfer_id TEXT PRIMARY KEY,
    manifest_hash TEXT NOT NULL UNIQUE CHECK (length(manifest_hash) = 64),
    manifest_json TEXT NOT NULL CHECK (json_valid(manifest_json) AND length(manifest_json) <= 16777216),
    source_data_source TEXT NOT NULL CHECK (source_data_source IN ('local', 'server')),
    source_server_profile_id TEXT NOT NULL DEFAULT '',
    source_principal_id TEXT NOT NULL DEFAULT '',
    source_workspace_id TEXT NOT NULL CHECK (length(source_workspace_id) BETWEEN 1 AND 256),
    destination_data_source TEXT NOT NULL CHECK (destination_data_source IN ('local', 'server')),
    destination_server_profile_id TEXT NOT NULL DEFAULT '',
    destination_principal_id TEXT NOT NULL DEFAULT '',
    destination_workspace_id TEXT NOT NULL CHECK (length(destination_workspace_id) BETWEEN 1 AND 256),
    replacement_confirmed INTEGER NOT NULL CHECK (replacement_confirmed IN (0, 1)),
    server_migration_id TEXT NOT NULL DEFAULT '',
    server_migration_finalized INTEGER NOT NULL DEFAULT 0 CHECK (server_migration_finalized IN (0, 1)),
    status TEXT NOT NULL CHECK (status IN (
        'confirmed', 'in_progress', 'completed', 'partially_completed',
        'rolled_back', 'failed_retryable', 'failed_terminal'
    )),
    lease_token TEXT NOT NULL DEFAULT '',
    lease_expires_at INTEGER,
    revision INTEGER NOT NULL DEFAULT 1 CHECK (typeof(revision) = 'integer' AND revision > 0),
    created_at TEXT NOT NULL,
    updated_at INTEGER NOT NULL CHECK (typeof(updated_at) = 'integer' AND updated_at >= 0),
    CHECK (transfer_id = 'copy-' || manifest_hash),
    CHECK (source_data_source <> destination_data_source),
    CHECK (
        (source_data_source = 'local' AND source_server_profile_id = '' AND source_principal_id = '')
        OR (source_data_source = 'server' AND source_server_profile_id <> '' AND source_principal_id <> '')
    ),
    CHECK (
        (destination_data_source = 'local' AND destination_server_profile_id = '' AND destination_principal_id = '')
        OR (destination_data_source = 'server' AND destination_server_profile_id <> '' AND destination_principal_id <> '')
    ),
    CHECK (server_migration_id = '' OR destination_data_source = 'server'),
    CHECK (server_migration_finalized = 0 OR server_migration_id <> ''),
    CHECK (
        (status = 'in_progress' AND lease_token <> '' AND typeof(lease_expires_at) = 'integer')
        OR (status <> 'in_progress' AND lease_token = '' AND lease_expires_at IS NULL)
    )
);

CREATE TABLE IF NOT EXISTS research_copy_items (
    transfer_id TEXT NOT NULL REFERENCES research_copy_operations(transfer_id) ON DELETE CASCADE,
    item_key TEXT NOT NULL CHECK (length(item_key) = 64),
    ordinal INTEGER NOT NULL CHECK (typeof(ordinal) = 'integer' AND ordinal >= 0),
    stage TEXT NOT NULL CHECK (stage IN ('planned', 'destination_saved', 'associated', 'verified')),
    destination_owner_id TEXT NOT NULL DEFAULT '',
    destination_owner_version INTEGER,
    destination_snapshot_hash TEXT NOT NULL DEFAULT '',
    failure TEXT NOT NULL DEFAULT '' CHECK (failure IN (
        '', 'owner_unavailable', 'source_changed', 'destination_changed', 'authority_changed',
        'association_failed', 'verification_failed', 'server_migration_failed'
    )),
    retryable INTEGER NOT NULL DEFAULT 0 CHECK (retryable IN (0, 1)),
    rolled_back INTEGER NOT NULL DEFAULT 0 CHECK (rolled_back IN (0, 1)),
    PRIMARY KEY (transfer_id, item_key),
    UNIQUE (transfer_id, ordinal),
    CHECK (destination_owner_version IS NULL OR (typeof(destination_owner_version) = 'integer' AND destination_owner_version >= 0)),
    CHECK (destination_snapshot_hash = '' OR length(destination_snapshot_hash) = 64),
    CHECK (
        (stage = 'planned' AND destination_owner_id = '' AND destination_owner_version IS NULL AND destination_snapshot_hash = '')
        OR (stage <> 'planned' AND destination_owner_id <> '' AND (destination_owner_version IS NOT NULL OR destination_snapshot_hash <> ''))
    ),
    CHECK (retryable = 0 OR failure <> ''),
    CHECK (stage <> 'verified' OR failure = ''),
    CHECK (rolled_back = 0 OR (stage <> 'planned' AND failure = '' AND retryable = 0))
);

CREATE INDEX IF NOT EXISTS idx_research_copy_source
ON research_copy_operations(source_data_source, source_server_profile_id, source_principal_id, source_workspace_id, status, updated_at);

CREATE INDEX IF NOT EXISTS idx_research_copy_destination
ON research_copy_operations(destination_data_source, destination_server_profile_id, destination_principal_id, destination_workspace_id, status, updated_at);
