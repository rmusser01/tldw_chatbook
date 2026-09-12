-- ADR-157 / ADR-158 / TASK-13154.7. Reference for the guarded runtime migration.
-- Execute once against v18; the runtime checks column presence before ALTER.
BEGIN IMMEDIATE;
ALTER TABLE agent_definitions ADD COLUMN max_wall_seconds REAL;
INSERT OR IGNORE INTO schema_version (version) VALUES (19);
COMMIT;
