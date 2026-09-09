-- ADR-141. Apply once to v15; runtime migration guards the column.
BEGIN IMMEDIATE;
ALTER TABLE automatic_wake_attempts ADD COLUMN attempt_kind TEXT NOT NULL DEFAULT 'fleet_wake' CHECK (attempt_kind IN ('fleet_wake', 'goal_iteration'));

CREATE TABLE IF NOT EXISTS goal_runs (
    id TEXT PRIMARY KEY,
    launch_id TEXT NOT NULL UNIQUE,
    payload_hash TEXT NOT NULL,
    request_json TEXT NOT NULL CHECK (length(CAST(request_json AS BLOB)) <= 131072),
    conversation_id TEXT NOT NULL UNIQUE,
    chain_id TEXT NOT NULL UNIQUE REFERENCES automatic_work_chains(id),
    revision INTEGER NOT NULL DEFAULT 1 CHECK (revision > 0),
    status TEXT NOT NULL DEFAULT 'starting',
    pause_reason TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS goal_iterations (
    id TEXT PRIMARY KEY,
    goal_id TEXT NOT NULL REFERENCES goal_runs(id),
    ordinal INTEGER NOT NULL CHECK (ordinal > 0),
    launch_id TEXT NOT NULL UNIQUE,
    attempt_id TEXT NOT NULL UNIQUE REFERENCES automatic_wake_attempts(id),
    run_id TEXT REFERENCES agent_runs(id),
    status TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 1 CHECK (revision > 0),
    check_results_json TEXT NOT NULL DEFAULT '[]' CHECK (length(CAST(check_results_json AS BLOB)) <= 131072),
    evidence_refs_json TEXT NOT NULL DEFAULT '[]' CHECK (length(CAST(evidence_refs_json AS BLOB)) <= 8192),
    UNIQUE(goal_id, ordinal)
);
CREATE TABLE IF NOT EXISTS goal_reports (
    id TEXT PRIMARY KEY,
    goal_id TEXT NOT NULL REFERENCES goal_runs(id),
    iteration_id TEXT UNIQUE REFERENCES goal_iterations(id),
    payload_json TEXT NOT NULL CHECK (length(CAST(payload_json AS BLOB)) <= 65536)
);
CREATE TRIGGER IF NOT EXISTS goal_launch_immutable
BEFORE UPDATE OF launch_id, payload_hash, request_json, conversation_id, chain_id ON goal_runs
WHEN OLD.launch_id IS NOT NEW.launch_id OR OLD.payload_hash IS NOT NEW.payload_hash
 OR OLD.request_json IS NOT NEW.request_json OR OLD.conversation_id IS NOT NEW.conversation_id
 OR OLD.chain_id IS NOT NEW.chain_id
BEGIN SELECT RAISE(ABORT, 'goal launch is immutable'); END;

INSERT OR IGNORE INTO schema_version (version) VALUES (16);
COMMIT;
