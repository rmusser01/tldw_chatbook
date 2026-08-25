BEGIN IMMEDIATE;

CREATE TABLE research_source_operations (
    operation_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    data_source TEXT NOT NULL CHECK (data_source IN ('local', 'server')),
    server_profile_id TEXT NOT NULL DEFAULT '',
    principal_id TEXT NOT NULL DEFAULT '',
    workspace_id TEXT NOT NULL,
    ingest_job_id TEXT NOT NULL DEFAULT '',
    canonical_item_type TEXT NOT NULL CHECK (
        canonical_item_type IN ('local_library', 'server_media')
    ),
    canonical_item_id TEXT NOT NULL DEFAULT '',
    workspace_source_id TEXT NOT NULL DEFAULT '',
    desired_selected INTEGER NOT NULL DEFAULT 1 CHECK (desired_selected IN (0, 1)),
    catalog_status TEXT NOT NULL DEFAULT 'pending' CHECK (
        catalog_status IN ('pending', 'in_progress', 'succeeded', 'failed')
    ),
    association_status TEXT NOT NULL DEFAULT 'pending' CHECK (
        association_status IN ('pending', 'in_progress', 'succeeded', 'failed')
    ),
    readiness_status TEXT NOT NULL DEFAULT 'pending' CHECK (
        readiness_status IN ('pending', 'in_progress', 'succeeded', 'failed')
    ),
    error_stage TEXT DEFAULT NULL CHECK (
        error_stage IS NULL OR error_stage IN ('catalog', 'association', 'readiness')
    ),
    error_code TEXT NOT NULL DEFAULT '',
    error_message TEXT NOT NULL DEFAULT '',
    revision INTEGER NOT NULL DEFAULT 1 CHECK (revision >= 1),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (
        (data_source = 'local'
         AND server_profile_id = ''
         AND principal_id = ''
         AND canonical_item_type = 'local_library')
        OR
        (data_source = 'server'
         AND server_profile_id <> ''
         AND canonical_item_type = 'server_media')
    )
);

CREATE INDEX idx_research_source_operations_incomplete
ON research_source_operations (
    catalog_status,
    association_status,
    readiness_status,
    created_at,
    operation_id
);

INSERT OR IGNORE INTO schema_version (version) VALUES (3);

COMMIT;
