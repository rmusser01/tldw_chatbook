-- V60 -> V61: content-free, dataset-scoped Agent Lessons seed state.
--
-- Folder creation belongs to the Notes-service readiness boundary. This
-- migration records only the monotonic fact needed to avoid recreating a
-- user-renamed or user-deleted conventional folder.
--
-- seed_fingerprint is an opaque, content-free digest. The Notes service owns
-- its canonical preimage and must include the evidence category
-- (coordinator_created, exact_root_reuse, or remote_history_upsert) so those
-- three observations never collapse to the same receipt. Raw category or
-- user content does not belong in this table.

CREATE TABLE agent_lessons_seed_state(
  profile_id TEXT NOT NULL,
  dataset_id TEXT NOT NULL,
  scope_mode TEXT NOT NULL CHECK(scope_mode IN ('local_only', 'synchronized')),
  state TEXT NOT NULL CHECK(state IN ('unknown', 'not_seeded', 'seeded')),
  folder_sync_id TEXT,
  seed_fingerprint TEXT NOT NULL CHECK(
    length(seed_fingerprint) = 64
    AND seed_fingerprint = lower(seed_fingerprint)
    AND seed_fingerprint NOT GLOB '*[^0-9a-f]*'
  ),
  PRIMARY KEY(profile_id, dataset_id)
) WITHOUT ROWID;

CREATE TRIGGER agent_lessons_seed_state_monotonic_update
BEFORE UPDATE OF scope_mode, state ON agent_lessons_seed_state
WHEN (OLD.state = 'seeded' AND NEW.state <> 'seeded')
 OR (
   OLD.scope_mode = 'synchronized'
   AND (
     NEW.scope_mode <> 'synchronized'
     OR (OLD.state = 'not_seeded' AND NEW.state = 'unknown')
   )
 )
BEGIN
  SELECT RAISE(ABORT, 'agent_lessons_seed_state cannot regress');
END;
