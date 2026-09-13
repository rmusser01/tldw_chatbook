-- ChaChaNotes v72 -> v73: persisted note-to-note link relation (task-32186).
--
-- "Which notes link to this one" was a leading-wildcard LIKE over the
-- unindexed `notes.content`: every active body read and sorted on every note
-- open, work that grows with the vault and that no index can serve. The
-- relation is written where the body is written instead, and read by an
-- indexed lookup on the target.
--
-- Deliberately NO sync columns and no sync-log triggers: this is derived data,
-- rebuildable from `notes.content` at any time (the migration step backfills
-- it exactly that way), so replicating the edges would be replicating the same
-- fact twice and inviting them to disagree. Same call the FTS shadow tables
-- make.
--
-- `target_note_id` is deliberately NOT a foreign key. A body may link to a note
-- that does not exist yet (an import inserts sources and targets in one
-- transaction, in no guaranteed order) or no longer exists at all; an immediate
-- FK check would reject the first and silently drop the second. Reads join to
-- `notes`, so a target that names nothing is simply invisible.

CREATE TABLE IF NOT EXISTS note_links(
  source_note_id TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE ON UPDATE CASCADE,
  target_note_id TEXT NOT NULL,
  PRIMARY KEY(source_note_id, target_note_id)
);

-- The backlink lookup's whole point: find the sources for one target without
-- touching the corpus. Covering, so the index answers the WHERE and hands the
-- join its keys.
CREATE INDEX IF NOT EXISTS idx_note_links_target
  ON note_links(target_note_id, source_note_id);
