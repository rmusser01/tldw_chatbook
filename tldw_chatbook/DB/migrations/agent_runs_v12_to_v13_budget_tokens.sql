-- ADR-131 / TASK-18311. Reference for the guarded migration in AgentRunsDB.
-- Execute once against v12; the runtime checks column presence before ALTER.
BEGIN IMMEDIATE;
ALTER TABLE agent_runs ADD COLUMN budget_tokens INTEGER
    CHECK (budget_tokens IS NULL OR
           (typeof(budget_tokens) = 'integer' AND budget_tokens >= 0));
INSERT OR IGNORE INTO schema_version (version) VALUES (13);
COMMIT;
