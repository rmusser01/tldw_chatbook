-- ChaChaNotes v29 -> v30: local-only per-message usage (cost ticker PR1).
-- DDL only. NOTE: no trigger DDL — usage_json is LOCAL-ONLY and must never
-- reach sync_log (same rule as v19/v24/v25/v26 local-only migrations).
--
-- This file stays a PLAIN, unconditional ALTER: SQLite has no
-- "ADD COLUMN IF NOT EXISTS". The Python runner
-- (`CharactersRAGDB._migrate_from_v29_to_v30`) checks
-- `PRAGMA table_info(messages)` first and skips executing this script when
-- `usage_json` is already present, still applying the version bump. Keep the
-- two in step if either side changes.

ALTER TABLE messages ADD COLUMN usage_json TEXT DEFAULT NULL;
