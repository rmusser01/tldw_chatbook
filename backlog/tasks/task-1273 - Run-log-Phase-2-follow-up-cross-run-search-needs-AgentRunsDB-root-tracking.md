---
id: TASK-1273
title: 'Run log Phase 2 follow-up: cross-run search needs AgentRunsDB root tracking'
status: To Do
assignee: []
created_date: '2026-07-29 00:20'
labels:
  - agents
  - run-log
dependencies:
  - TASK-1271
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-1271 (run-log Phase 2: aggregation, slicing, cross-run search) investigated cross-run search and deferred it here rather than build it against a guess. The log is per-run: a run's directory is named by its bare run_id under <root>/.agent-runs/<run_id>/, with no per-conversation grouping on disk, and each run's MANIFEST records only run-level metadata (run_id, model, api_endpoint, allowed_tools, budget, status, superseded_run_id, total_tokens) -- neither conversation_id nor the resolved root the log was written under. AgentRunsDB.agent_runs does map conversation_id -> run_id (indexed), which cross-run search needs, but it has no column recording which root (a bound workspace folder, or the sandbox fallback) resolve_log_root() chose for that run -- that choice is resolved fresh on every run and never persisted. So a historical run's log directory cannot be located from its run_id alone without either assuming the current workspace binding matches every historical run (silently wrong whenever it does not: workspace folder reconfigured, different session, different sandbox root) or adding a schema column. The design spec's own §2.2 deliberately kept AgentRunsDB out of Phase 1's retrieval path, and this is exactly the kind of DB-schema change task-1271 was told not to improvise.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 agent_runs gains a column recording the resolved log root (or full log_dir path) at the point RunLogWriter.bind() creates a run's directory, so a later run can locate an earlier run's log deterministically
- [ ] #2 A cross-run search tool (or an extension to search_run_log) lets the primary agent search a named set of a conversation's earlier runs (e.g. its immediately preceding run, or all runs for the conversation) using the same bounded literal/regex search already implemented in run_log_search.py
- [ ] #3 Runs written before this migration (no recorded root) degrade gracefully -- reported as unlocatable rather than raising or silently skipped without explanation
- [ ] #4 Cross-run search remains primary-agent only, mirroring search_run_log/run_log_stats/run_log_slice's own isolation gate
<!-- AC:END -->
