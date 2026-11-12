CREATE TABLE marketplace_sources (
    source_id TEXT PRIMARY KEY NOT NULL,
    binding_json TEXT NOT NULL,
    binding_mac TEXT NOT NULL,
    snapshot_json TEXT,
    refresh_error TEXT
);
PRAGMA user_version=5;
