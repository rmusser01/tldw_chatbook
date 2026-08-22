-- ChaChaNotes v43 -> v44: recoverable copy of a discarded sync side
-- (task-19554).
--
-- Before this, `sync_conflicts` held only `db_content_hash` and
-- `disk_content_hash`. When the Notes sync engine resolved a `both_changed`
-- conflict it overwrote the losing side wholesale and the discarded text was
-- unrecoverable -- a SHA-256 is not a backup.
--
-- These three columns are written ONLY when a side is actually discarded, not
-- on every conflict DETECTION: a conflict that is merely recorded (the `ask`
-- policy, or a strategy that declines to apply) destroys nothing and needs no
-- copy. That keeps this from becoming a second unbounded full-content shadow
-- of `notes` the way `sync_log` already is.
--
--   losing_side         'db' | 'disk' -- which side was thrown away
--   losing_content      that side's text, verbatim, for reconstruction
--   preserved_file_path the sidecar written next to the note file, which is
--                       the copy a user actually recovers from (a rename);
--                       this row is the second copy for when that file is
--                       gone.
--
-- DDL only. The schema-version bump is a separate rowcount-guarded UPDATE in
-- the runner (`CharactersRAGDB._migrate_from_v43_to_v44`), matching the
-- v42->v43 precedent.

ALTER TABLE sync_conflicts ADD COLUMN losing_side TEXT;

ALTER TABLE sync_conflicts ADD COLUMN losing_content TEXT;

ALTER TABLE sync_conflicts ADD COLUMN preserved_file_path TEXT;
