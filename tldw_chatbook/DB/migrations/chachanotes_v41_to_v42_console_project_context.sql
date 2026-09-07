-- ChaChaNotes schema migration: v41 -> v42
--
-- Add local-only Console project-context persistence. This column is excluded
-- from sync triggers and import/export payloads by design.

ALTER TABLE conversations ADD COLUMN console_project_context_json TEXT;
