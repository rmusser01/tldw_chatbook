CREATE TABLE workflow_runs (
    run_id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    revision_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    operation_id TEXT NOT NULL,
    payload_digest TEXT NOT NULL,
    snapshot_json TEXT NOT NULL CHECK(json_valid(snapshot_json)),
    manifest_json TEXT NOT NULL CHECK(json_valid(manifest_json)),
    generation INTEGER NOT NULL DEFAULT 1 CHECK(generation > 0),
    event_sequence INTEGER NOT NULL DEFAULT 0 CHECK(event_sequence >= 0),
    status TEXT NOT NULL CHECK(status IN ('running','pausing','paused','stopping','cancelled','timed_out','completed','failed','needs_review','waiting_human','needs_permission')),
    step_index INTEGER NOT NULL DEFAULT 0 CHECK(step_index >= 0),
    attempt INTEGER NOT NULL DEFAULT 0 CHECK(attempt >= 0),
    outputs_json TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(outputs_json)),
    wait_generation INTEGER,
    error_code TEXT,
    active_seconds REAL NOT NULL DEFAULT 0 CHECK(active_seconds >= 0),
    active_since REAL,
    created_at REAL NOT NULL,
    UNIQUE(profile_id, operation_id)
);
CREATE TABLE workflow_attempts (
    run_id TEXT NOT NULL REFERENCES workflow_runs(run_id),
    step_id TEXT NOT NULL,
    attempt_number INTEGER NOT NULL CHECK(attempt_number > 0),
    request_json TEXT NOT NULL CHECK(json_valid(request_json)),
    owner_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('reserved','started','succeeded','failed','waiting_human','needs_permission','abandoned')),
    outcome_json TEXT CHECK(outcome_json IS NULL OR json_valid(outcome_json)),
    PRIMARY KEY(run_id, step_id, attempt_number)
);
CREATE TABLE workflow_events (
    run_id TEXT NOT NULL REFERENCES workflow_runs(run_id),
    sequence INTEGER NOT NULL CHECK(sequence > 0),
    generation INTEGER NOT NULL CHECK(generation > 0),
    kind TEXT NOT NULL CHECK(kind IN ('attempt_reserved','attempt_started','attempt_finished','control_requested','wait_opened','wait_decided','policy_amended','recovery_recorded')),
    payload_json TEXT NOT NULL CHECK(json_valid(payload_json)),
    PRIMARY KEY(run_id, sequence)
);
CREATE TABLE workflow_waits (
    run_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    attempt_number INTEGER NOT NULL,
    generation INTEGER NOT NULL CHECK(generation > 0),
    kind TEXT NOT NULL CHECK(kind IN ('waiting_human','needs_permission','needs_review')),
    status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','approved','rejected','expired','cancelled')),
    payload_json TEXT NOT NULL CHECK(json_valid(payload_json)),
    PRIMARY KEY(run_id, generation),
    FOREIGN KEY(run_id, step_id, attempt_number) REFERENCES workflow_attempts(run_id, step_id, attempt_number)
);
CREATE TABLE workflow_budget_ledger (
    run_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    attempt_number INTEGER NOT NULL,
    input_units INTEGER NOT NULL CHECK(input_units >= 0),
    output_units INTEGER NOT NULL CHECK(output_units >= 0),
    token_units INTEGER NOT NULL CHECK(token_units >= 0),
    reserved_bytes INTEGER NOT NULL CHECK(reserved_bytes >= 0),
    output_bytes INTEGER NOT NULL DEFAULT 0 CHECK(output_bytes >= 0),
    artifact_bytes INTEGER NOT NULL DEFAULT 0 CHECK(artifact_bytes = 0),
    PRIMARY KEY(run_id, step_id, attempt_number),
    FOREIGN KEY(run_id, step_id, attempt_number) REFERENCES workflow_attempts(run_id, step_id, attempt_number)
);
CREATE TABLE workflow_effects (
    effect_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    attempt_number INTEGER NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('reserved','started','succeeded','failed','waiting_human','needs_permission','abandoned')),
    receipt_json TEXT CHECK(receipt_json IS NULL OR json_valid(receipt_json)),
    UNIQUE(run_id, step_id, attempt_number),
    FOREIGN KEY(run_id, step_id, attempt_number) REFERENCES workflow_attempts(run_id, step_id, attempt_number)
);
CREATE TABLE workflow_runtime_capacity (
    slot INTEGER PRIMARY KEY CHECK(slot = 1),
    run_id TEXT UNIQUE REFERENCES workflow_runs(run_id),
    owner_id TEXT,
    CHECK ((run_id IS NULL) = (owner_id IS NULL))
);
INSERT INTO workflow_runtime_capacity(slot, run_id) VALUES (1, NULL);
PRAGMA user_version = 2;
