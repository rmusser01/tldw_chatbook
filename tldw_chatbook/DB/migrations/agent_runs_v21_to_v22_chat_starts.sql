-- ADR-211: local conversation chains share an immutable direct allowance root.
PRAGMA foreign_keys=ON;
BEGIN IMMEDIATE;
ALTER TABLE automatic_work_chains ADD COLUMN allowance_root_chain_id TEXT REFERENCES automatic_work_chains(id);
DROP TRIGGER IF EXISTS automatic_chain_identity_immutable;
CREATE TRIGGER IF NOT EXISTS automatic_chain_identity_immutable
BEFORE UPDATE OF id, conversation_id, root_submission_id, limits_json, allowance_root_chain_id ON automatic_work_chains
WHEN OLD.id IS NOT NEW.id
  OR OLD.allowance_root_chain_id IS NOT NEW.allowance_root_chain_id
  OR OLD.conversation_id IS NOT NEW.conversation_id
  OR OLD.root_submission_id IS NOT NEW.root_submission_id
  OR OLD.limits_json IS NOT NEW.limits_json
BEGIN SELECT RAISE(ABORT, 'automatic chain identity is immutable'); END;
CREATE INDEX IF NOT EXISTS idx_automatic_chains_allowance_root
    ON automatic_work_chains(allowance_root_chain_id);
CREATE TRIGGER IF NOT EXISTS automatic_chain_root_insert
BEFORE INSERT ON automatic_work_chains
WHEN NEW.allowance_root_chain_id IS NOT NULL
AND (NEW.id=NEW.allowance_root_chain_id OR NOT EXISTS (
    SELECT 1 FROM automatic_work_chains WHERE id=NEW.allowance_root_chain_id
    AND allowance_root_chain_id IS NULL))
BEGIN SELECT RAISE(ABORT, 'automatic allowance must name a direct root'); END;
CREATE TRIGGER IF NOT EXISTS automatic_chain_root_update
BEFORE UPDATE OF allowance_root_chain_id ON automatic_work_chains
WHEN NEW.allowance_root_chain_id IS NOT NULL
AND (NEW.id=NEW.allowance_root_chain_id OR NOT EXISTS (
    SELECT 1 FROM automatic_work_chains WHERE id=NEW.allowance_root_chain_id
    AND allowance_root_chain_id IS NULL))
BEGIN SELECT RAISE(ABORT, 'automatic allowance must name a direct root'); END;
CREATE TABLE IF NOT EXISTS automatic_chat_start_attempts (
    id TEXT PRIMARY KEY,
    source_run_id TEXT NOT NULL REFERENCES agent_runs(id),
    source_chain_id TEXT NOT NULL REFERENCES automatic_work_chains(id),
    chain_id TEXT NOT NULL UNIQUE REFERENCES automatic_work_chains(id),
    conversation_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    session_incarnation TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    draft_revision INTEGER NOT NULL CHECK (typeof(draft_revision)='integer' AND draft_revision>=0),
    context_epoch INTEGER NOT NULL CHECK (typeof(context_epoch)='integer' AND context_epoch>=0),
    request_fingerprint TEXT NOT NULL CHECK (length(request_fingerprint)=64 AND request_fingerprint NOT GLOB '*[^0-9a-f]*'),
    generation_reservation_id TEXT NOT NULL UNIQUE REFERENCES automatic_work_reservations(id),
    state TEXT NOT NULL CHECK (state IN ('prepared', 'accepted', 'completed', 'aborted', 'review_required')),
    created_at REAL NOT NULL,
    accepted_at REAL,
    completed_at REAL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_automatic_chat_start_conversation_active
    ON automatic_chat_start_attempts(conversation_id) WHERE state IN ('prepared', 'accepted');
CREATE TRIGGER IF NOT EXISTS automatic_chat_start_identity_immutable
BEFORE UPDATE OF id, source_run_id, source_chain_id, chain_id, conversation_id,
    session_id, session_incarnation, owner_id, draft_revision, context_epoch,
    request_fingerprint, generation_reservation_id ON automatic_chat_start_attempts
WHEN OLD.id IS NOT NEW.id OR OLD.source_run_id IS NOT NEW.source_run_id
  OR OLD.source_chain_id IS NOT NEW.source_chain_id OR OLD.chain_id IS NOT NEW.chain_id
  OR OLD.conversation_id IS NOT NEW.conversation_id OR OLD.session_id IS NOT NEW.session_id
  OR OLD.session_incarnation IS NOT NEW.session_incarnation OR OLD.owner_id IS NOT NEW.owner_id
  OR OLD.draft_revision IS NOT NEW.draft_revision OR OLD.context_epoch IS NOT NEW.context_epoch
  OR OLD.request_fingerprint IS NOT NEW.request_fingerprint
  OR OLD.generation_reservation_id IS NOT NEW.generation_reservation_id
BEGIN SELECT RAISE(ABORT, 'chat start identity is immutable'); END;
INSERT OR IGNORE INTO schema_version (version) VALUES (22);
COMMIT;
