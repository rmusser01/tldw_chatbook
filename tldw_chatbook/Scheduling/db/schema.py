"""SQLite DDL for the Scheduling module.

JSON columns are stored as TEXT and parsed/serialized in Python.
All datetime columns store UTC ISO-8601 strings.
"""

CREATE_SCHEMA_SQL = """
-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

-- Reminder tasks (local and synced)
CREATE TABLE IF NOT EXISTS reminder_tasks (
    id TEXT PRIMARY KEY,
    server_id TEXT,
    owner_id TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT,
    schedule_kind TEXT NOT NULL,
    run_at TEXT,
    cron TEXT,
    timezone TEXT,
    enabled INTEGER NOT NULL DEFAULT 1,
    last_status TEXT,
    next_run_at TEXT,
    last_run_at TEXT,
    missed_at TEXT,
    link_type TEXT,
    link_id TEXT,
    link_url TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT,
    sync_version INTEGER NOT NULL DEFAULT 0,
    UNIQUE (owner_id, server_id)
);

CREATE INDEX IF NOT EXISTS idx_reminder_tasks_owner_enabled_next_run
    ON reminder_tasks (owner_id, enabled, next_run_at);

CREATE INDEX IF NOT EXISTS idx_reminder_tasks_owner_last_status
    ON reminder_tasks (owner_id, last_status);

CREATE INDEX IF NOT EXISTS idx_reminder_tasks_server_id
    ON reminder_tasks (server_id);

-- Automation definitions
CREATE TABLE IF NOT EXISTS automation_definitions (
    id TEXT PRIMARY KEY,
    server_id TEXT,
    owner_id TEXT NOT NULL,
    family TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    lifecycle TEXT NOT NULL,
    health TEXT NOT NULL,
    schedule TEXT,
    input TEXT,
    config TEXT,
    visibility_policy TEXT,
    notification_policy TEXT,
    approval_policy TEXT,
    version INTEGER NOT NULL DEFAULT 1,
    preview_id TEXT,
    created_by TEXT,
    updated_by TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT,
    archived_at TEXT,
    UNIQUE (owner_id, server_id)
);

CREATE INDEX IF NOT EXISTS idx_automation_definitions_owner_lifecycle_health
    ON automation_definitions (owner_id, lifecycle, health);

CREATE INDEX IF NOT EXISTS idx_automation_definitions_owner_family
    ON automation_definitions (owner_id, family);

CREATE INDEX IF NOT EXISTS idx_automation_definitions_server_id
    ON automation_definitions (server_id);

-- Automation previews (create/update validation previews)
CREATE TABLE IF NOT EXISTS automation_previews (
    id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    mode TEXT,
    family TEXT NOT NULL,
    definition_id TEXT,
    definition_version INTEGER,
    status TEXT NOT NULL,
    payload_hash TEXT,
    normalized_config TEXT,
    validation_errors TEXT,
    warnings TEXT,
    visibility_policy TEXT,
    schedule_preview TEXT,
    redaction_policy TEXT,
    expires_at TEXT,
    created_by TEXT,
    created_at TEXT NOT NULL,
    consumed_at TEXT,
    created_definition_id TEXT
);

-- Audit events for automation definitions
CREATE TABLE IF NOT EXISTS automation_audit_events (
    id TEXT PRIMARY KEY,
    definition_id TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    actor TEXT NOT NULL,
    summary TEXT NOT NULL,
    before TEXT,
    after TEXT,
    request_id TEXT,
    idempotency_key TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_automation_audit_events_definition_created_at
    ON automation_audit_events (definition_id, created_at);

-- Sync state per owner
CREATE TABLE IF NOT EXISTS sync_state (
    owner_id TEXT PRIMARY KEY,
    last_pull_at TEXT,
    last_push_at TEXT,
    last_conflict_at TEXT,
    sync_errors TEXT
);

-- Mapping between local and server IDs
CREATE TABLE IF NOT EXISTS sync_mapping (
    local_id TEXT NOT NULL,
    server_id TEXT,
    primitive TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (local_id, primitive, owner_id)
);

CREATE INDEX IF NOT EXISTS idx_sync_mapping_server_primitive_owner
    ON sync_mapping (server_id, primitive, owner_id);

-- Soft-delete tombstones for sync
CREATE TABLE IF NOT EXISTS sync_tombstones (
    local_id TEXT NOT NULL,
    primitive TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    deleted_at TEXT NOT NULL,
    pushed_at TEXT,
    PRIMARY KEY (local_id, primitive, owner_id)
);

-- Conflicts surfaced during sync reconciliation
CREATE TABLE IF NOT EXISTS sync_conflicts (
    id TEXT PRIMARY KEY,
    local_id TEXT NOT NULL,
    primitive TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    server_state TEXT,
    local_state TEXT,
    server_state_at TEXT,
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    resolution TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0
);
"""
