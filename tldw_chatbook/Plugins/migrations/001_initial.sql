CREATE TABLE installations (
    installation_id TEXT PRIMARY KEY NOT NULL,
    revision_digest TEXT,
    activation_default INTEGER NOT NULL DEFAULT 0 CHECK (activation_default IN (0, 1))
);
CREATE TABLE revisions (
    installation_id TEXT NOT NULL REFERENCES installations(installation_id),
    revision_digest TEXT NOT NULL,
    inspection_json TEXT NOT NULL,
    PRIMARY KEY (installation_id, revision_digest)
);
CREATE TABLE components (
    installation_id TEXT NOT NULL,
    revision_digest TEXT NOT NULL,
    component_id TEXT NOT NULL,
    definition_json TEXT NOT NULL,
    PRIMARY KEY (installation_id, revision_digest, component_id),
    FOREIGN KEY (installation_id, revision_digest) REFERENCES revisions
);
CREATE TABLE selections (
    installation_id TEXT NOT NULL,
    revision_digest TEXT NOT NULL,
    component_id TEXT NOT NULL,
    selected INTEGER NOT NULL CHECK (selected IN (0, 1)),
    PRIMARY KEY (installation_id, revision_digest, component_id),
    FOREIGN KEY (installation_id, revision_digest, component_id) REFERENCES components
);
CREATE TABLE activation (
    installation_id TEXT NOT NULL REFERENCES installations(installation_id),
    workspace_id TEXT NOT NULL,
    intent TEXT NOT NULL CHECK (intent IN ('inherit', 'enabled', 'disabled')),
    PRIMARY KEY (installation_id, workspace_id)
);
CREATE TABLE sources (
    source_id TEXT PRIMARY KEY NOT NULL,
    installation_id TEXT NOT NULL REFERENCES installations(installation_id),
    source_json TEXT NOT NULL
);
CREATE TABLE mappings (
    installation_id TEXT NOT NULL REFERENCES installations(installation_id),
    mapping_id TEXT NOT NULL,
    mapping_json TEXT NOT NULL,
    PRIMARY KEY (installation_id, mapping_id)
);
CREATE TABLE authority_generations (
    installation_id TEXT NOT NULL REFERENCES installations(installation_id),
    scope_kind TEXT NOT NULL CHECK (scope_kind IN ('installation', 'global_default', 'workspace')),
    workspace_id TEXT NOT NULL DEFAULT '',
    generation INTEGER NOT NULL CHECK (generation >= 0),
    revoked INTEGER NOT NULL DEFAULT 0 CHECK (revoked IN (0, 1)),
    PRIMARY KEY (installation_id, scope_kind, workspace_id),
    CHECK ((scope_kind = 'workspace' AND workspace_id != '') OR
           (scope_kind != 'workspace' AND workspace_id = ''))
);
CREATE TABLE operations (
    operation_id TEXT PRIMARY KEY NOT NULL,
    installation_id TEXT NOT NULL,
    phase TEXT NOT NULL,
    intent_json TEXT NOT NULL
);
CREATE TABLE data_roots (
    root_id TEXT PRIMARY KEY NOT NULL,
    installation_id TEXT NOT NULL,
    workspace_id TEXT,
    path TEXT NOT NULL UNIQUE,
    generation INTEGER NOT NULL CHECK (generation >= 0),
    deletion_fenced INTEGER NOT NULL DEFAULT 0 CHECK (deletion_fenced IN (0, 1))
);
CREATE TABLE processes (
    token TEXT PRIMARY KEY NOT NULL,
    operation_id TEXT NOT NULL,
    installation_id TEXT NOT NULL,
    workspace_id TEXT,
    revision_digest TEXT NOT NULL,
    owner_session TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('pending', 'published', 'unresolved', 'settled')),
    kind TEXT NOT NULL CHECK (kind IN ('pending_launch', 'active_run', 'idle_connection', 'archived_history')),
    provenance_json TEXT
);
CREATE INDEX process_recovery ON processes(installation_id, state, owner_session);
CREATE INDEX process_revision ON processes(installation_id, revision_digest, kind, state);
CREATE TABLE receipts (
    receipt_id TEXT PRIMARY KEY NOT NULL,
    operation_id TEXT NOT NULL,
    receipt_json TEXT NOT NULL
);
CREATE TRIGGER immutable_revisions BEFORE UPDATE ON revisions BEGIN
    SELECT RAISE(ABORT, 'immutable revision');
END;
CREATE TRIGGER immutable_components BEFORE UPDATE ON components BEGIN
    SELECT RAISE(ABORT, 'immutable component');
END;
