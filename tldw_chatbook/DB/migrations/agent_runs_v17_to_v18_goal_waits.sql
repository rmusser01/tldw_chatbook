-- ADR-141: durable retry scheduling metadata; allowance stays in automatic_work.
BEGIN IMMEDIATE;
CREATE TABLE IF NOT EXISTS goal_waits (
    goal_id TEXT PRIMARY KEY REFERENCES goal_runs(id),
    retry_at REAL NOT NULL,
    reason TEXT NOT NULL
);
DROP TRIGGER IF EXISTS goal_launch_immutable;
CREATE TRIGGER goal_launch_immutable
BEFORE UPDATE OF launch_id, payload_hash, request_json, conversation_id, chain_id ON goal_runs
WHEN OLD.launch_id IS NOT NEW.launch_id OR OLD.payload_hash IS NOT NEW.payload_hash
 OR (OLD.request_json IS NOT NEW.request_json AND NOT (NEW.request_json='' AND NEW.status='removed'
     AND OLD.status IN ('completed','paused','awaiting_result_review','stopped')
     AND NOT EXISTS (SELECT 1 FROM goal_iterations i JOIN automatic_wake_attempts a ON a.id=i.attempt_id
         WHERE i.goal_id=OLD.id AND a.state IN ('prepared','accepted','review_required'))))
 OR OLD.conversation_id IS NOT NEW.conversation_id OR OLD.chain_id IS NOT NEW.chain_id
BEGIN SELECT RAISE(ABORT, 'goal launch is immutable'); END;
INSERT OR IGNORE INTO schema_version (version) VALUES (18);
COMMIT;
