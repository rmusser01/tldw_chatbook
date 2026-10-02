ALTER TABLE data_roots ADD COLUMN custody_json TEXT;
ALTER TABLE data_roots ADD COLUMN cleanup_json TEXT;
ALTER TABLE processes ADD COLUMN root_coverage TEXT NOT NULL DEFAULT 'unknown' CHECK (root_coverage IN ('unknown', 'qualified_none', 'known'));
ALTER TABLE processes ADD COLUMN root_grants_json TEXT;
CREATE TABLE root_users (
    usage_token TEXT PRIMARY KEY NOT NULL,
    root_id TEXT NOT NULL,
    root_generation INTEGER NOT NULL CHECK (root_generation >= 0),
    binding_digest TEXT NOT NULL,
    owner_token TEXT NOT NULL REFERENCES processes(token),
    access TEXT NOT NULL CHECK (access IN ('read', 'write', 'read_write')),
    state TEXT NOT NULL CHECK (state IN ('held', 'unresolved', 'released')),
    UNIQUE (owner_token, root_id, root_generation)
);
CREATE INDEX root_user_recovery ON root_users(root_id, state, root_generation);
CREATE INDEX root_user_owner ON root_users(owner_token);
PRAGMA user_version=4;
