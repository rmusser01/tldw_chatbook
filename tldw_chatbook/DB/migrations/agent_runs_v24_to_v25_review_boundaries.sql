-- ADR-281 body-free review-boundary claims; executed and catalog-compared by migration tests.

BEGIN IMMEDIATE;
ALTER TABLE automatic_wake_attempts ADD COLUMN review_boundary_id TEXT;
CREATE TABLE IF NOT EXISTS automatic_review_boundary_claims (
    boundary_id TEXT PRIMARY KEY,
    attempt_id TEXT NOT NULL UNIQUE REFERENCES automatic_wake_attempts(id),
    owner_id TEXT NOT NULL,
    chain_id TEXT NOT NULL REFERENCES automatic_work_chains(id),
    review_id TEXT NOT NULL,
    slot_id TEXT NOT NULL,
    slot_attempt_id TEXT NOT NULL,
    source_epoch INTEGER NOT NULL,
    context_revision INTEGER NOT NULL,
    grant_id TEXT NOT NULL,
    target_fingerprint TEXT NOT NULL,
    team_fingerprint TEXT NOT NULL,
    override_fingerprint TEXT NOT NULL,
    predecessor_run_ids_json TEXT NOT NULL,
    predecessor_settlement_ids_json TEXT NOT NULL,
    UNIQUE(owner_id, chain_id, review_id, slot_id, slot_attempt_id)
);
INSERT INTO schema_version(version) VALUES (25);
COMMIT;
