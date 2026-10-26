CREATE TABLE mobile_consolidation_batches (
    operation_id TEXT NOT NULL,
    source_fingerprint TEXT NOT NULL,
    batch_key TEXT NOT NULL,
    mapping_json TEXT NOT NULL,
    committed_at TEXT NOT NULL,
    PRIMARY KEY (operation_id, batch_key)
);
