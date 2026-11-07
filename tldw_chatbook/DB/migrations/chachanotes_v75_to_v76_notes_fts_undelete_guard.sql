-- ChaChaNotes v75 -> v76: repair the `notes_au` FTS update trigger body
-- (task-19565).
--
-- `notes_au` originally shipped without its `deleted = 0` guards, so a
-- soft-deleted note was never removed from `notes_fts` and a restored note
-- was never re-inserted: FTS search returned deleted notes and missed
-- restored ones. Because the v4 base script only runs on fresh databases,
-- the repair shipped as a runtime self-heal that DROP/reCREATEd the trigger
-- on EVERY database open -- a permanent startup fixup standing in for a
-- migration. This step replaces that: every database passes through
-- v75 -> v76 exactly once, so the correct body ships once and the
-- self-heal is deleted.
--
-- The recreated body is byte-identical (whitespace-normalized) to the one
-- the v4 base script creates, so a fresh bootstrap and a chain-migrated
-- database converge on the same sqlite_master row; the trigger census
-- (Tests/ChaChaNotesDB/test_trigger_census.py) pins it from now on.

DROP TRIGGER IF EXISTS notes_au;

CREATE TRIGGER notes_au
AFTER UPDATE ON notes BEGIN
  INSERT INTO notes_fts(notes_fts,rowid,title,content)
  SELECT 'delete',old.rowid,old.title,old.content
  WHERE old.deleted = 0;

  INSERT INTO notes_fts(rowid,title,content)
  SELECT new.rowid,new.title,new.content
  WHERE new.deleted = 0;
END;

UPDATE db_schema_version
   SET version = 76
 WHERE schema_name = 'rag_char_chat_schema'
   AND version = 75;
