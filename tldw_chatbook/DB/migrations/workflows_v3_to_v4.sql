ALTER TABLE workflow_runtime_guard ADD COLUMN legacy_ownership_unverified INTEGER NOT NULL DEFAULT 1 CHECK(legacy_ownership_unverified IN (0,1));
PRAGMA user_version = 4;
