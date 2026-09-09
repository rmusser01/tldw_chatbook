-- ADR-141: private bounded goal evidence, reserved capacity and settled tombstones.
BEGIN IMMEDIATE;
CREATE TABLE IF NOT EXISTS goal_checkpoints (
    attempt_id TEXT PRIMARY KEY REFERENCES automatic_wake_attempts(id),
    goal_id TEXT NOT NULL REFERENCES goal_runs(id),
    payload_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL CHECK(length(CAST(payload_json AS BLOB)) <= 131072)
);
CREATE TABLE IF NOT EXISTS goal_evidence (
    id TEXT PRIMARY KEY,
    goal_id TEXT NOT NULL REFERENCES goal_runs(id),
    attempt_id TEXT NOT NULL REFERENCES automatic_wake_attempts(id),
    payload_json TEXT NOT NULL CHECK(length(CAST(payload_json AS BLOB)) <= 131072)
);
CREATE TABLE IF NOT EXISTS goal_payload_reservations (
    attempt_id TEXT PRIMARY KEY REFERENCES automatic_wake_attempts(id),
    goal_id TEXT NOT NULL REFERENCES goal_runs(id),
    bytes INTEGER NOT NULL CHECK(bytes >= 0)
);
DROP TRIGGER IF EXISTS goal_launch_immutable;
CREATE TRIGGER goal_launch_immutable
BEFORE UPDATE OF launch_id, payload_hash, request_json, conversation_id, chain_id ON goal_runs
WHEN OLD.launch_id IS NOT NEW.launch_id OR OLD.payload_hash IS NOT NEW.payload_hash
 OR (OLD.request_json IS NOT NEW.request_json AND NOT (NEW.request_json='' AND NEW.status='removed'
     AND OLD.status IN ('completed','paused','awaiting_result_review')
     AND NOT EXISTS (SELECT 1 FROM goal_iterations i JOIN automatic_wake_attempts a ON a.id=i.attempt_id
         WHERE i.goal_id=OLD.id AND a.state IN ('prepared','accepted','review_required'))))
 OR OLD.conversation_id IS NOT NEW.conversation_id OR OLD.chain_id IS NOT NEW.chain_id
BEGIN SELECT RAISE(ABORT, 'goal launch is immutable'); END;

UPDATE goal_reports SET payload_json = json_remove(json_set(payload_json,
 '$.candidate_draft', COALESCE(json_extract(payload_json,'$.draft'),''),
 '$.evidence_ids', json(COALESCE(json_extract(payload_json,'$.evidence_refs'),'[]')),
 '$.completion_recommended', json('false')), '$.draft', '$.evidence_refs')
 WHERE json_valid(payload_json) AND (json_type(payload_json,'$.draft') IS NOT NULL OR json_type(payload_json,'$.evidence_refs') IS NOT NULL);

INSERT OR IGNORE INTO schema_version (version) VALUES (17);
COMMIT;
