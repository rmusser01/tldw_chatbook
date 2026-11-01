-- ADR-285 additive passive workflow storage.
BEGIN IMMEDIATE;
CREATE TABLE IF NOT EXISTS workflow_runs (
    id TEXT PRIMARY KEY,
    launch_id TEXT NOT NULL UNIQUE,
    payload_hash TEXT NOT NULL,
    request_json TEXT NOT NULL CHECK(length(CAST(request_json AS BLOB)) <= 131072),
    conversation_id TEXT NOT NULL,
    chain_id TEXT NOT NULL UNIQUE REFERENCES automatic_work_chains(id),
    control_revision INTEGER NOT NULL DEFAULT 1 CHECK(typeof(control_revision)='integer' AND control_revision>0),
    progress_cursor INTEGER NOT NULL DEFAULT 0 CHECK(typeof(progress_cursor)='integer' AND progress_cursor>=0),
    status TEXT NOT NULL DEFAULT 'awaiting_plan' CHECK(status IN ('awaiting_plan','awaiting_approval','running','paused','stopped','completed','closed','recovery_required','removed','awaiting_resume')),
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_workflow_conversation ON workflow_runs(conversation_id, created_at DESC, id);
CREATE TABLE IF NOT EXISTS workflow_plan_revisions (
    workflow_id TEXT NOT NULL REFERENCES workflow_runs(id),
    revision INTEGER NOT NULL CHECK(typeof(revision)='integer' AND revision>0),
    payload_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL CHECK(length(CAST(payload_json AS BLOB)) <= 131072),
    PRIMARY KEY(workflow_id, revision)
);
CREATE TABLE IF NOT EXISTS workflow_phases (
    workflow_id TEXT NOT NULL,
    plan_revision INTEGER NOT NULL,
    phase_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK(ordinal BETWEEN 1 AND 6),
    PRIMARY KEY(workflow_id, plan_revision, phase_id),
    UNIQUE(workflow_id, plan_revision, ordinal),
    FOREIGN KEY(workflow_id, plan_revision) REFERENCES workflow_plan_revisions(workflow_id, revision)
);
CREATE TABLE IF NOT EXISTS workflow_tasks (
    workflow_id TEXT NOT NULL,
    plan_revision INTEGER NOT NULL,
    phase_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK(ordinal BETWEEN 1 AND 12),
    PRIMARY KEY(workflow_id, plan_revision, phase_id, task_id),
    UNIQUE(workflow_id, plan_revision, task_id),
    FOREIGN KEY(workflow_id, plan_revision, phase_id) REFERENCES workflow_phases(workflow_id, plan_revision, phase_id)
);
CREATE TABLE IF NOT EXISTS workflow_phase_attempts (
    id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    plan_revision INTEGER NOT NULL,
    phase_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK(typeof(ordinal)='integer' AND ordinal>0),
    native_attempt_id TEXT UNIQUE REFERENCES automatic_wake_attempts(id),
    status TEXT NOT NULL CHECK(status IN ('prepared','accepted','completed','failed','uncertain','closed')),
    retirement TEXT NOT NULL DEFAULT 'pending' CHECK(retirement IN ('pending','drained','unknown')),
    UNIQUE(workflow_id, plan_revision, phase_id, ordinal),
    FOREIGN KEY(workflow_id, plan_revision, phase_id) REFERENCES workflow_phases(workflow_id, plan_revision, phase_id)
);
CREATE TABLE IF NOT EXISTS workflow_task_attempts (
    id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    plan_revision INTEGER NOT NULL,
    phase_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK(typeof(ordinal)='integer' AND ordinal>0),
    run_id TEXT UNIQUE REFERENCES agent_runs(id),
    native_attempt_id TEXT UNIQUE REFERENCES automatic_wake_attempts(id),
    status TEXT NOT NULL CHECK(status IN ('prepared','accepted','completed','failed','uncertain','closed')),
    retirement TEXT NOT NULL DEFAULT 'pending' CHECK(retirement IN ('pending','drained','unknown')),
    UNIQUE(id, workflow_id),
    UNIQUE(workflow_id, plan_revision, task_id, ordinal),
    FOREIGN KEY(workflow_id, plan_revision, phase_id, task_id) REFERENCES workflow_tasks(workflow_id, plan_revision, phase_id, task_id)
);
CREATE TABLE IF NOT EXISTS workflow_decisions (
    id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    plan_revision INTEGER NOT NULL,
    phase_id TEXT NOT NULL,
    control_revision INTEGER NOT NULL CHECK(control_revision>0),
    payload_json TEXT NOT NULL CHECK(length(CAST(payload_json AS BLOB)) <= 131072),
    UNIQUE(workflow_id, control_revision),
    FOREIGN KEY(workflow_id, plan_revision, phase_id) REFERENCES workflow_phases(workflow_id, plan_revision, phase_id)
);
CREATE TABLE IF NOT EXISTS workflow_payload_reservations (
    attempt_id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    bytes INTEGER NOT NULL CHECK(typeof(bytes)='integer' AND bytes BETWEEN 1 AND 131072),
    state TEXT NOT NULL CHECK(state IN ('reserved','published','removed')),
    FOREIGN KEY(attempt_id, workflow_id) REFERENCES workflow_task_attempts(id, workflow_id)
);
CREATE TABLE IF NOT EXISTS workflow_results (
    attempt_id TEXT PRIMARY KEY REFERENCES workflow_payload_reservations(attempt_id),
    workflow_id TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL CHECK(length(CAST(payload_json AS BLOB)) <= 131072),
    FOREIGN KEY(attempt_id, workflow_id) REFERENCES workflow_task_attempts(id, workflow_id)
);
CREATE TABLE IF NOT EXISTS workflow_active_intervals (
    id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL REFERENCES workflow_runs(id),
    owner_id TEXT NOT NULL,
    started_at REAL NOT NULL CHECK(started_at>=0),
    ended_at REAL CHECK(ended_at>=started_at),
    charged_seconds REAL NOT NULL DEFAULT 0 CHECK(charged_seconds>=0)
);
CREATE TRIGGER IF NOT EXISTS workflow_launch_immutable
BEFORE UPDATE OF id, launch_id, payload_hash, request_json, conversation_id, chain_id ON workflow_runs
BEGIN SELECT RAISE(ABORT, 'workflow launch is immutable'); END;
CREATE TRIGGER IF NOT EXISTS workflow_plan_immutable
BEFORE UPDATE ON workflow_plan_revisions
BEGIN SELECT RAISE(ABORT, 'workflow plan is immutable'); END;
CREATE TRIGGER IF NOT EXISTS workflow_decision_immutable
BEFORE UPDATE ON workflow_decisions
BEGIN SELECT RAISE(ABORT, 'workflow decision is immutable'); END;
CREATE TRIGGER IF NOT EXISTS workflow_result_immutable
BEFORE UPDATE ON workflow_results
BEGIN SELECT RAISE(ABORT, 'workflow result is immutable'); END;
INSERT OR IGNORE INTO schema_version(version) VALUES (25);
COMMIT;
