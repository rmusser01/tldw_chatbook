-- Migration: Workspace registry V1 to V2 case-insensitive unique names
-- Adds a case-insensitive uniqueness guarantee on non-archived workspace
-- names so two workspaces can never collide (e.g. "Client A" vs "client a").
--
-- Procedural step (not expressible as plain SQL, see the runner in
-- tldw_chatbook/DB/Workspace_DB.py `_initialize_schema`):
--   Before the index below can be created, any pre-existing non-archived
--   duplicate names (case-insensitive) must be deduped. The runner:
--     1. Reads all non-archived (workspace_id, name) rows, ordered by
--        created_at ASC, workspace_id ASC.
--     2. Builds a `reserved` set of every existing name's stripped,
--        casefolded form up front (covers retained rows AND any
--        pre-existing "Name (n)"-shaped names).
--     3. For each duplicate group (by casefolded name), keeps the first
--        (earliest-created) row untouched and renames the rest to
--        "<original name> (n)", probing n upward until the candidate's
--        casefolded form is not in `reserved`, then reserving it.
--     4. Applies the renames as UPDATE workspace_records SET name = ...
--        WHERE workspace_id = ... statements.
--   All of the above renames plus the statements below run inside a single
--   transaction so a mid-migration failure rolls back atomically.

CREATE UNIQUE INDEX IF NOT EXISTS idx_workspace_records_name_ci
    ON workspace_records (lower(name)) WHERE archived = 0;

INSERT OR IGNORE INTO schema_version (version) VALUES (2);
