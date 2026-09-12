-- ADR-134 / ADR-135 / TASK-32036. Reference for the guarded runtime migration.
-- Execute once against v16; the runtime checks column presence before ALTER.
BEGIN IMMEDIATE;
ALTER TABLE agent_runs ADD COLUMN work_chain_id TEXT REFERENCES automatic_work_chains(id);

CREATE TABLE IF NOT EXISTS automatic_work_chains (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    root_submission_id TEXT NOT NULL UNIQUE,
    limits_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'paused', 'review_required')),
    pause_reason TEXT,
    created_at REAL NOT NULL,
    started_at REAL,
    deadline_at REAL,
    clock_owner_id TEXT,
    started_monotonic REAL,
    last_observed_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS automatic_work_reservations (
    id TEXT PRIMARY KEY,
    chain_id TEXT NOT NULL REFERENCES automatic_work_chains(id),
    owner_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('generation', 'child_launch', 'model_call', 'tokens')),
    amount INTEGER NOT NULL CHECK (typeof(amount) = 'integer' AND amount > 0),
    state TEXT NOT NULL CHECK (state IN ('reserved', 'committed', 'released', 'settled', 'uncertain')),
    actual_amount INTEGER CHECK (actual_amount IS NULL OR (typeof(actual_amount) = 'integer' AND actual_amount >= 0)),
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_automatic_reservations_chain
    ON automatic_work_reservations(chain_id);
CREATE TABLE IF NOT EXISTS automatic_wake_attempts (
    id TEXT PRIMARY KEY,
    chain_id TEXT NOT NULL REFERENCES automatic_work_chains(id),
    conversation_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    generation_reservation_id TEXT NOT NULL UNIQUE REFERENCES automatic_work_reservations(id),
    run_ids_json TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('prepared', 'accepted', 'completed', 'aborted', 'review_required')),
    created_at REAL NOT NULL,
    accepted_at REAL,
    completed_at REAL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_automatic_wake_conversation_active
    ON automatic_wake_attempts(conversation_id) WHERE state IN ('prepared', 'accepted');
CREATE TABLE IF NOT EXISTS automatic_wake_claims (
    run_id TEXT PRIMARY KEY REFERENCES agent_runs(id),
    attempt_id TEXT NOT NULL REFERENCES automatic_wake_attempts(id)
);
CREATE INDEX IF NOT EXISTS idx_automatic_claims_attempt
    ON automatic_wake_claims(attempt_id);
CREATE TRIGGER IF NOT EXISTS automatic_chain_identity_immutable
BEFORE UPDATE OF conversation_id, root_submission_id, limits_json ON automatic_work_chains
WHEN OLD.conversation_id IS NOT NEW.conversation_id
  OR OLD.root_submission_id IS NOT NEW.root_submission_id
  OR OLD.limits_json IS NOT NEW.limits_json
BEGIN SELECT RAISE(ABORT, 'automatic chain identity is immutable'); END;
CREATE TRIGGER IF NOT EXISTS automatic_run_chain_immutable
BEFORE UPDATE OF work_chain_id ON agent_runs
WHEN OLD.work_chain_id IS NOT NULL AND OLD.work_chain_id IS NOT NEW.work_chain_id
BEGIN SELECT RAISE(ABORT, 'run chain is immutable'); END;
CREATE TRIGGER IF NOT EXISTS automatic_run_chain_scope_insert
BEFORE INSERT ON agent_runs WHEN NEW.work_chain_id IS NOT NULL
AND NOT EXISTS (SELECT 1 FROM automatic_work_chains WHERE id=NEW.work_chain_id
                AND conversation_id=NEW.conversation_id)
BEGIN SELECT RAISE(ABORT, 'run chain scope mismatch'); END;
CREATE TRIGGER IF NOT EXISTS automatic_run_chain_scope_update
BEFORE UPDATE OF work_chain_id, conversation_id ON agent_runs
WHEN NEW.work_chain_id IS NOT NULL
AND NOT EXISTS (SELECT 1 FROM automatic_work_chains WHERE id=NEW.work_chain_id
                AND conversation_id=NEW.conversation_id)
BEGIN SELECT RAISE(ABORT, 'run chain scope mismatch'); END;

INSERT OR IGNORE INTO schema_version (version) VALUES (17);
COMMIT;
