-- ADR-285: bounded whole-wave Change Review evidence; no execution authority.
CREATE TABLE workflow_change_windows (
    id TEXT PRIMARY KEY CHECK(length(CAST(id AS BLOB)) BETWEEN 1 AND 256),
    workflow_id TEXT NOT NULL REFERENCES workflow_runs(id),
    phase_attempt_id TEXT NOT NULL REFERENCES workflow_phase_attempts(id),
    ordinal INTEGER NOT NULL CHECK(typeof(ordinal)='integer' AND ordinal BETWEEN 1 AND 64),
    anchor_attempt_id TEXT REFERENCES workflow_task_attempts(id),
    roots_json TEXT NOT NULL CHECK(length(CAST(roots_json AS BLOB)) <= 32768),
    state TEXT NOT NULL CHECK(state IN ('active','unused','completed','disclosed_failure','unresolved','removed')),
    baseline TEXT NOT NULL DEFAULT 'pending' CHECK(baseline IN ('pending','completed','disclosed_failure','unresolved')),
    ending TEXT NOT NULL DEFAULT 'pending' CHECK(ending IN ('pending','completed','disclosed_failure','unresolved','unused')),
    snapshot_count INTEGER CHECK(snapshot_count IS NULL OR (typeof(snapshot_count)='integer' AND snapshot_count BETWEEN 0 AND 64)),
    reserved_bytes INTEGER NOT NULL CHECK(reserved_bytes IN (0,65536)),
    UNIQUE(workflow_id, ordinal),
    CHECK((state='removed' AND reserved_bytes=0 AND roots_json='[]') OR (state!='removed' AND reserved_bytes=65536)),
    CHECK(state!='unused' OR (anchor_attempt_id IS NULL AND baseline='completed' AND ending='unused')),
    CHECK(state!='completed' OR (anchor_attempt_id IS NOT NULL AND baseline='completed' AND ending='completed')),
    CHECK(state NOT IN ('active','unresolved') OR ending IN ('pending','unresolved')),
    CHECK((state IN ('active','unresolved') AND snapshot_count IS NULL) OR (state NOT IN ('active','unresolved') AND snapshot_count IS NOT NULL)),
    CHECK(state NOT IN ('unused','removed') OR snapshot_count=0)
);
CREATE TABLE workflow_change_window_refs (
    window_id TEXT NOT NULL REFERENCES workflow_change_windows(id),
    snapshot_id INTEGER NOT NULL UNIQUE REFERENCES change_snapshots(id),
    PRIMARY KEY(window_id,snapshot_id)
);
CREATE TRIGGER workflow_change_window_identity
BEFORE UPDATE OF id,workflow_id,phase_attempt_id,ordinal ON workflow_change_windows
BEGIN SELECT RAISE(ABORT, 'workflow window identity is immutable'); END;
CREATE TRIGGER workflow_change_window_anchor
BEFORE UPDATE OF anchor_attempt_id ON workflow_change_windows
WHEN OLD.anchor_attempt_id IS NOT NULL AND NEW.anchor_attempt_id IS NOT OLD.anchor_attempt_id
BEGIN SELECT RAISE(ABORT, 'workflow window anchor is immutable'); END;
INSERT INTO schema_version(version) VALUES (28);
