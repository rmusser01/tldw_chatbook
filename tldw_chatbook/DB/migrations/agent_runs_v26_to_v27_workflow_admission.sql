-- ADR-285 workflow admission; existing v25 rows and identities are retained.
BEGIN IMMEDIATE;
ALTER TABLE automatic_wake_attempts RENAME COLUMN attempt_kind TO old_attempt_kind;
ALTER TABLE automatic_wake_attempts ADD COLUMN attempt_kind TEXT NOT NULL DEFAULT 'fleet_wake' CHECK (attempt_kind IN ('fleet_wake', 'goal_iteration', 'workflow_phase'));
UPDATE automatic_wake_attempts SET attempt_kind=old_attempt_kind;
ALTER TABLE automatic_wake_attempts DROP COLUMN old_attempt_kind;
ALTER TABLE workflow_phase_attempts RENAME TO workflow_phase_attempts_v26;
CREATE TABLE workflow_phase_attempts (
    id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL REFERENCES workflow_runs(id),
    plan_revision INTEGER,
    phase_id TEXT,
    ordinal INTEGER NOT NULL CHECK(typeof(ordinal)='integer' AND ordinal>0),
    native_attempt_id TEXT UNIQUE REFERENCES automatic_wake_attempts(id),
    status TEXT NOT NULL CHECK(status IN ('prepared','accepted','completed','failed','uncertain','closed')),
    retirement TEXT NOT NULL DEFAULT 'pending' CHECK(retirement IN ('pending','drained','unknown')),
    purpose TEXT NOT NULL DEFAULT 'execution' CHECK(purpose IN ('planning','execution')),
    command_id TEXT UNIQUE,
    control_revision INTEGER CHECK(control_revision>0),
    result_bytes INTEGER NOT NULL DEFAULT 0 CHECK(result_bytes BETWEEN 0 AND 131072),
    CHECK((purpose='execution' AND plan_revision IS NOT NULL AND phase_id IS NOT NULL)
       OR (purpose='planning' AND phase_id IS NULL)),
    UNIQUE(workflow_id, plan_revision, phase_id, ordinal),
    FOREIGN KEY(workflow_id, plan_revision, phase_id) REFERENCES workflow_phases(workflow_id, plan_revision, phase_id)
);
INSERT INTO workflow_phase_attempts (id,workflow_id,plan_revision,phase_id,ordinal,native_attempt_id,status,retirement) SELECT id,workflow_id,plan_revision,phase_id,ordinal,native_attempt_id,status,retirement FROM workflow_phase_attempts_v26;
DROP TABLE workflow_phase_attempts_v26;
CREATE TABLE workflow_task_claims (
    task_attempt_id TEXT PRIMARY KEY REFERENCES workflow_task_attempts(id),
    phase_attempt_id TEXT NOT NULL REFERENCES workflow_phase_attempts(id),
    child_reservation_id TEXT UNIQUE REFERENCES automatic_work_reservations(id),
    launch_run_id TEXT NOT NULL UNIQUE
);
ALTER TABLE workflow_active_intervals ADD COLUMN clock_owner_id TEXT;
ALTER TABLE workflow_active_intervals ADD COLUMN started_monotonic REAL;
ALTER TABLE workflow_active_intervals ADD COLUMN observed_monotonic REAL;
CREATE UNIQUE INDEX idx_workflow_open_interval ON workflow_active_intervals(workflow_id) WHERE ended_at IS NULL;
CREATE INDEX idx_workflow_task_phase ON workflow_task_claims(phase_attempt_id);
INSERT INTO schema_version(version) VALUES (27);
COMMIT;
