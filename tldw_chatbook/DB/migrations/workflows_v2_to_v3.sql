ALTER TABLE workflow_waits RENAME TO workflow_waits_v2;
CREATE TABLE workflow_waits (
    run_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    attempt_number INTEGER NOT NULL,
    generation INTEGER NOT NULL CHECK(generation > 0),
    kind TEXT NOT NULL CHECK(kind IN ('waiting_human','needs_permission','needs_review')),
    status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','decided','approved','rejected','expired','cancelled')),
    payload_json TEXT NOT NULL CHECK(json_valid(payload_json)),
    actor_id TEXT,
    effect_digest TEXT,
    opened_at TEXT,
    deadline_at TEXT,
    engine_timeout_seconds INTEGER,
    response_timeout_seconds INTEGER,
    decision_json TEXT CHECK(decision_json IS NULL OR json_valid(decision_json)),
    decided_at TEXT,
    PRIMARY KEY(run_id, generation),
    FOREIGN KEY(run_id, step_id, attempt_number) REFERENCES workflow_attempts(run_id, step_id, attempt_number)
);
INSERT INTO workflow_waits(run_id,step_id,attempt_number,generation,kind,status,payload_json)
SELECT run_id,step_id,attempt_number,generation,kind,status,payload_json FROM workflow_waits_v2;
DROP TABLE workflow_waits_v2;
ALTER TABLE workflow_runs ADD COLUMN limits_json TEXT CHECK(limits_json IS NULL OR json_valid(limits_json));
ALTER TABLE workflow_attempts ADD COLUMN active_seconds REAL NOT NULL DEFAULT 0 CHECK(active_seconds >= 0);
ALTER TABLE workflow_attempts ADD COLUMN active_since REAL;
CREATE TABLE workflow_runtime_guard (
    slot INTEGER PRIMARY KEY CHECK(slot=1),
    device INTEGER,
    inode INTEGER,
    CHECK((device IS NULL) = (inode IS NULL))
);
INSERT INTO workflow_runtime_guard(slot) VALUES (1);
PRAGMA user_version = 3;
