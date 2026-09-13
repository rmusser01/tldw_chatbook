-- AgentRuns schema migration: v18 -> v19
--
-- ADR-147, TASK-32477 (agent provider routing): preset routing fields on
-- agent_definitions (provider, params_json) and the resolved-target snapshot
-- on agent_runs (resolved_provider, resolved_model, resolved_base_url,
-- resolved_params_json -- where a spawned run actually went, recorded once
-- at spawn and read back verbatim on resume/continuation).
--
-- Numbering: planned as v15 -> v16 when the branch forked; renumbered to
-- v18 -> v19 after dev landed its own v16 (budget_tokens), v17
-- (automatic_work), and v18 (runtime_owner) via #2641.
--
-- Applied at runtime by AgentRunsDB._initialize_schema's PRAGMA-guarded
-- idempotent ALTERs; this file is the per-version audit record, matching
-- the repo's migration-file convention. Run it only against a database
-- whose agent_definitions/agent_runs tables lack these columns (plain
-- ALTER TABLE is not idempotent on its own).

ALTER TABLE agent_definitions ADD COLUMN provider TEXT NOT NULL DEFAULT '';
ALTER TABLE agent_definitions ADD COLUMN params_json TEXT NOT NULL DEFAULT '{}';

ALTER TABLE agent_runs ADD COLUMN resolved_provider TEXT;
ALTER TABLE agent_runs ADD COLUMN resolved_model TEXT;
ALTER TABLE agent_runs ADD COLUMN resolved_base_url TEXT;
ALTER TABLE agent_runs ADD COLUMN resolved_params_json TEXT;
