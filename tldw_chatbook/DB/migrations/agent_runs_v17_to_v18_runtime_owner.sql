-- ADR-135 / TASK-32037. Explicit recovery alone populates this owner fence.
BEGIN IMMEDIATE;
CREATE TABLE IF NOT EXISTS automatic_work_runtime_owner (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    owner_id TEXT NOT NULL CHECK (length(owner_id) BETWEEN 1 AND 256)
);
INSERT OR IGNORE INTO schema_version (version) VALUES (18);
COMMIT;
