-- ChaChaNotes v30 -> v31: local-only per-message structured metadata
-- (task-2364: engine provenance, interrupted flag, transcript status).
-- DDL only. NOTE: no trigger DDL — metadata_json is LOCAL-ONLY and must
-- never reach sync_log (same rule as the v29->v30 usage_json column and the
-- v19/v24/v25/v26 local-only migrations).
--
-- This file stays a PLAIN, unconditional ALTER: SQLite has no
-- "ADD COLUMN IF NOT EXISTS". The Python runner
-- (`CharactersRAGDB._migrate_from_v30_to_v31`) checks
-- `PRAGMA table_info(messages)` first and skips executing this script when
-- `metadata_json` is already present, still applying the version bump. Keep
-- the two in step if either side changes.

ALTER TABLE messages ADD COLUMN metadata_json TEXT DEFAULT NULL;
