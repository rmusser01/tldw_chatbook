-- ChaChaNotes v29 -> v30: local-only per-message usage (cost ticker PR1).
-- DDL only. NOTE: no trigger DDL — usage_json is LOCAL-ONLY and must never
-- reach sync_log (same rule as v19/v24/v25/v26 local-only migrations).

ALTER TABLE messages ADD COLUMN usage_json TEXT DEFAULT NULL;
