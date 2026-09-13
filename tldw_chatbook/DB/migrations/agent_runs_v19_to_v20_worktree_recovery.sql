-- ADR-155 / ADR-158. Reference for the guarded runtime migration.
BEGIN IMMEDIATE;
CREATE TABLE IF NOT EXISTS agent_worktrees (
    run_id TEXT PRIMARY KEY REFERENCES agent_runs(id),
    workspace_id TEXT NOT NULL,
    binding_id TEXT NOT NULL,
    locator_fingerprint TEXT NOT NULL,
    repo_root TEXT NOT NULL,
    repo_identity TEXT NOT NULL,
    git_common_dir TEXT NOT NULL,
    git_common_identity TEXT NOT NULL,
    child_path TEXT NOT NULL,
    child_identity TEXT NOT NULL,
    branch TEXT NOT NULL,
    base_sha TEXT NOT NULL,
    execution_id TEXT NOT NULL,
    writer_state TEXT NOT NULL DEFAULT 'held'
        CHECK(writer_state IN ('held', 'drained', 'uncertain')),
    mutation_state TEXT NOT NULL DEFAULT 'unresolved'
        CHECK(mutation_state IN ('unresolved', 'applying', 'merging', 'discarding',
            'applied', 'merged', 'discarded_cleanup_pending', 'uncertain')),
    operation_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_agent_worktrees_scope
    ON agent_worktrees(workspace_id, binding_id, run_id);
INSERT OR IGNORE INTO schema_version (version) VALUES (20);
COMMIT;
