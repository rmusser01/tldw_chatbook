-- Migration: ChaChaNotes V20 to V21 world_book_entries priority
-- Adds an `priority INTEGER DEFAULT 0` column to `world_book_entries` for
-- entry injection priority / budget-survival weight (Roleplay P2c), and
-- redefines the world_book_entries_sync_* triggers so edits to the new
-- column are reflected in sync_log. The runner guards the ADD COLUMN with a
-- PRAGMA check (SQLite has no ADD COLUMN IF NOT EXISTS) so replayed/partial
-- migrations are idempotent.

ALTER TABLE world_book_entries ADD COLUMN priority INTEGER DEFAULT 0;

DROP TRIGGER IF EXISTS world_book_entries_sync_create;
DROP TRIGGER IF EXISTS world_book_entries_sync_update;

CREATE TRIGGER world_book_entries_sync_create
AFTER INSERT ON world_book_entries BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('world_book_entries', CAST(NEW.id AS TEXT), 'create', NEW.last_modified,
         (SELECT client_id FROM world_books WHERE id = NEW.world_book_id), 1,
         json_object('id', NEW.id, 'world_book_id', NEW.world_book_id, 'keys', NEW.keys,
                     'content', NEW.content, 'enabled', NEW.enabled, 'position', NEW.position,
                     'insertion_order', NEW.insertion_order, 'priority', NEW.priority,
                     'selective', NEW.selective, 'secondary_keys', NEW.secondary_keys,
                     'case_sensitive', NEW.case_sensitive, 'extensions', NEW.extensions,
                     'created_at', NEW.created_at, 'last_modified', NEW.last_modified));
END;

CREATE TRIGGER world_book_entries_sync_update
AFTER UPDATE ON world_book_entries
WHEN OLD.keys IS NOT NEW.keys OR
     OLD.content IS NOT NEW.content OR
     OLD.enabled IS NOT NEW.enabled OR
     OLD.position IS NOT NEW.position OR
     OLD.insertion_order IS NOT NEW.insertion_order OR
     OLD.priority IS NOT NEW.priority OR
     OLD.selective IS NOT NEW.selective OR
     OLD.secondary_keys IS NOT NEW.secondary_keys OR
     OLD.case_sensitive IS NOT NEW.case_sensitive OR
     OLD.extensions IS NOT NEW.extensions
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('world_book_entries', CAST(NEW.id AS TEXT), 'update', NEW.last_modified,
         (SELECT client_id FROM world_books WHERE id = NEW.world_book_id), 1,
         json_object('id', NEW.id, 'world_book_id', NEW.world_book_id, 'keys', NEW.keys,
                     'content', NEW.content, 'enabled', NEW.enabled, 'position', NEW.position,
                     'insertion_order', NEW.insertion_order, 'priority', NEW.priority,
                     'selective', NEW.selective, 'secondary_keys', NEW.secondary_keys,
                     'case_sensitive', NEW.case_sensitive, 'extensions', NEW.extensions,
                     'created_at', NEW.created_at, 'last_modified', NEW.last_modified));
END;

UPDATE db_schema_version
   SET version = 21
 WHERE schema_name = 'rag_char_chat_schema'
   AND version = 20;
