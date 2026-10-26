CREATE TABLE mobile_consolidation_batches (
    operation_id TEXT NOT NULL,
    source_fingerprint TEXT NOT NULL,
    batch_key TEXT NOT NULL,
    mapping_json TEXT NOT NULL,
    committed_at TEXT NOT NULL,
    PRIMARY KEY (operation_id, batch_key)
);
CREATE TABLE mobile_import_authority_holds (
    operation_id TEXT NOT NULL,
    authority_kind TEXT NOT NULL,
    authority_id TEXT NOT NULL,
    reviewed_at TEXT,
    PRIMARY KEY (authority_kind, authority_id)
);
INSERT OR IGNORE INTO schema_version (version) VALUES (9);
