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

## Revised analysis (2026-07-28, after TASK-870 landed)

The original deferral said locating a historical run's log "needs either an unsafe
assumption or a schema change". That was more pessimistic than the code warrants.
Two pieces that already exist close most of the gap:

1. **Runs ARE queryable by conversation.** `agent_runs` carries an indexed
   `conversation_id` column, and `AgentRunsDB.list_runs(conversation_id, ...)`
   already returns a conversation's runs newest-first. So enumerating which runs
   belong to a conversation needs no new schema at all.
2. **A run's log directory is already resolvable by id.** TASK-870 added
   `run_log.resolve_existing_log_dir(run_id)`, the read-only counterpart to
   `RunLogWriter.bind()`: it resolves the current root and returns that run's log
   directory if it exists, without creating anything.

Composing those two gives working cross-run search today:
`list_runs(conversation_id)` -> `resolve_existing_log_dir(run_id)` ->
`load_records` -> `search_records`.

**The one real gap** is narrower than "we cannot find the logs": nothing records
which ROOT a given run's log was written under. So if the log root changed between
runs — the user bound, rebound or unbound a workspace folder — older logs are not
under the current root and will not be found. That is a graceful degradation (those
runs report as unavailable), not a correctness hazard: a run whose log IS found is
always genuinely that run's log, because the directory is keyed by run id.

So there are two honest options rather than one blocker:

- **(a) Best-effort, no schema change.** Search every run whose log resolves under
  the current root; report the rest as unavailable rather than silently omitting
  them. Ships now, correct for the common case where the root is stable.
- **(b) Exact, with a migration.** Record the resolved log path (or root) on the
  run record at write time, so a run's log is locatable regardless of later
  workspace changes. Completes the feature; costs a schema version bump.

(a) does not preclude (b) — a stored path can be preferred when present and the
current-root probe kept as the fallback for runs predating the column.

**Dependency worth noting:** this builds on `resolve_existing_log_dir`, which is on
the TASK-870 branch (PR #1082), and on the Phase 2 query tools (PR #1078). Both must
land before this can be implemented.
