CREATE TABLE console_command_receipts (
    command_id TEXT PRIMARY KEY,
    owner_epoch TEXT NOT NULL,
    console_id TEXT NOT NULL,
    action TEXT NOT NULL,
    target_json TEXT NOT NULL,
    payload_fingerprint TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN (
        'pending', 'accepted', 'queued', 'applied', 'refused', 'outcome_unknown'
    )),
    issued_ns INTEGER NOT NULL,
    expires_ns INTEGER NOT NULL CHECK (expires_ns > issued_ns),
    native_id TEXT,
    reason TEXT,
    updated_at TEXT NOT NULL
);
INSERT OR IGNORE INTO schema_version (version) VALUES (25);
