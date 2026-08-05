---
id: TASK-1271
title: 'Run log Phase 2 — aggregation, slicing and cross-run search'
status: Done
assignee:
  - '@claude'
created_date: '2026-07-28 00:00'
updated_date: '2026-07-29 00:21'
labels:
  - agents
  - run-log
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 1 of programmatic run memory shipped in PR #1066: every model turn, tool call and tool
result is appended losslessly to a segmented log, and `search_run_log` lets the primary
agent query the current run's log with literal search, structured filters, match-centred
rendering and `offset` paging.

Phase 2 in the design spec (§10) is the layer above that: letting an agent *compute over*
its history rather than only retrieve from it, and reaching beyond the current run.

Motivation from the source paper (PRO-LONG, arXiv 2607.20064v2): its ablation found the
read side is where the gains live — read-only 23.1%, plus grep 27.2%, plus programmatic
computation 38.3%. Phase 1 delivers roughly the middle tier. Aggregation is the cheap,
bounded way to move toward the third without introducing code execution (which is a separate
deferred decision, spec §11).

Scope:

- **`run_log_stats`** — bounded aggregation over the current run: counts grouped by tool,
  record type, or status; token totals; error rates. Answers "which tool have I called
  most, and how often did it fail?" without paging the whole log through context.
- **`run_log_slice`** — retrieve a contiguous record range as a unit, so an agent can
  reconstruct a stretch of its own reasoning rather than assembling it from separate hits.
- **Cross-run search** — the log is per-run today. Searching across a conversation's earlier
  runs is what makes the log useful for "what did I try last time?".

Consider whether cross-run search belongs in the file layer (glob across run directories) or
should reuse `AgentRunsDB` for indexing, and record the reasoning — the spec's §2.2
deliberately kept the DB out of the Phase 1 retrieval path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An agent can obtain aggregate statistics over its run log without paging individual records through its context
- [x] #2 An agent can retrieve a contiguous range of records as a coherent unit
- [x] #3 Cross-run search is available, or a recorded decision explains why it is deferred and what it would take
- [x] #4 Every new tool is bounded so a single call cannot blow the context window, consistent with how `search_run_log` bounds its own output
- [x] #5 New tools follow the established runtime-tool pattern (name constant, `RUNTIME_TOOL_NAMES`, schema, optional `LoopDeps` field, dispatch branch guarded by `deps.X is not None`, primary-agent gating where isolation requires it)
- [x] #6 Tests cover each tool's bounds and its behaviour on an empty, a partial, and a multi-segment log
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Extend the existing query layer (run_log_search.py) with two pure, bounded functions: compute_stats/format_stats (aggregation, output bounded by distinct group count, never by record count) and slice_records/format_slice (contiguous range, output bounded by a fixed MAX_SLICE_RECORDS cap regardless of requested range width or log size; format_slice reuses format_results rather than writing a second renderer).
2. Register RUN_LOG_STATS_TOOL_NAME / RUN_LOG_SLICE_TOOL_NAME in agent_models.py and RUNTIME_TOOL_NAMES.
3. Add ToolSchema definitions in tool_catalog.py, mirroring SEARCH_RUN_LOG_TOOL_SCHEMA's shape and boundedness language.
4. Add two optional LoopDeps fields in agent_runtime.py plus dispatch branches in run_agent_loop's elif chain, guarded by deps.X is not None, appended after the search_run_log branch.
5. Wire real closures in agent_service.py's _run_one, gated identically to search_run_log: schema disclosure under the existing three-part log_active gate, LoopDeps wiring under agent_kind == AGENT_KIND_PRIMARY.
6. Investigate cross-run search: inspect the on-disk run-directory layout, MANIFEST contents, and AgentRunsDB's schema to determine whether a conversation's earlier runs can be located without a DB change. Record the decision either way per spec §2.2's instruction to keep the DB out of the retrieval path unless truly required.
7. Tests: pure-function tests extending Tests/Agents/test_run_log_search.py (empty log, boundedness under a large synthetic log, filters, slicing edge cases); a new Tests/Agents/test_run_log_stats_slice_runtime_tools.py covering schema registration, loop dispatch, sub-agent isolation (schema-not-disclosed + dispatch-refused), real-closure junk-argument coercion (string/null/nested-object/list/huge/negative), empty/single/multi-segment log behaviour, and boundedness against a large real on-disk log.
8. Manually mutation-test the primary-agent gate (temporarily wire run_log_stats/run_log_slice unconditionally, confirm the isolation test fails, restore) rather than leaving it unverified.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented run_log_stats and run_log_slice as the seventh/eighth runtime
tools, following search_run_log's established pattern exactly: name
constants in agent_models.py + RUNTIME_TOOL_NAMES membership, ToolSchema
definitions in tool_catalog.py, optional LoopDeps fields defaulting to
None in agent_runtime.py plus dispatch branches guarded by
`deps.X is not None`, and service wiring in agent_service.py gated
identically to search_run_log (schema disclosure under the three-part
log_active gate; LoopDeps wiring under agent_kind == AGENT_KIND_PRIMARY).

run_log_stats aggregates counts/error-counts/content-bytes grouped by
tool, record type, status, or agent kind. Bounded by construction: output
is one line per DISTINCT group value, never per record -- group_by is
restricted to those four metadata fields (small, log-independent
cardinality) specifically so a per-record-unique dimension (a record
number, a call id) can never be chosen. Token totals are NOT reported:
RunLogRecord carries no per-record token count (only a whole-run total,
recorded in MANIFEST once the run ends), and fabricating a per-group
estimate would silently disagree with the run's own authoritative
accounting. content_bytes is reported instead as an exact, always-
available proxy.

run_log_slice retrieves a contiguous record-number range as one unit,
capped at MAX_SLICE_RECORDS (50) regardless of how wide the requested
range is or how large the log has grown, and reuses format_results for
per-record rendering rather than writing a second renderer (guards
against reintroducing the TASK-1250 "match not in the rendered body"
defect class in a new place).

Mutation-tested the primary-agent gate: temporarily wired both closures
unconditionally (removing the `agent_kind == AGENT_KIND_PRIMARY` check),
confirmed test_subagent_cannot_call_run_log_stats_or_run_log_slice fails
without it (a child's call stopped being refused), then restored the
gate and confirmed the test passes again.

Cross-run search: investigated, DEFERRED to task-1273. Run directories
are named by bare run_id under <root>/.agent-runs/<run_id>/ with no
per-conversation grouping on disk, and each run's MANIFEST records only
run-level metadata (run_id, model, api_endpoint, allowed_tools, budget,
status, superseded_run_id, total_tokens) -- neither conversation_id nor
the resolved root the log was written under. AgentRunsDB.agent_runs does
map conversation_id -> run_id, which cross-run search needs, but has no
column recording which root resolve_log_root() chose for a given run --
that choice is resolved fresh every run and never persisted anywhere.
Locating a historical run's log from just its run_id therefore requires
either assuming the current workspace binding matches every historical
run (silently wrong whenever it doesn't) or a schema addition. Per spec
§2.2's instruction to keep AgentRunsDB out of the retrieval path unless
truly required, and per this task's own instruction not to improvise a
DB change, this was left as task-1273 rather than built against a guess.

Files touched:
- tldw_chatbook/Agents/agent_models.py (name constants, RUNTIME_TOOL_NAMES)
- tldw_chatbook/Agents/run_log_search.py (compute_stats, format_stats,
  slice_records, format_slice, STATS_GROUP_BY_FIELDS, DEFAULT_SLICE_WIDTH,
  MAX_SLICE_RECORDS)
- tldw_chatbook/Agents/tool_catalog.py (RUN_LOG_STATS_TOOL_SCHEMA,
  RUN_LOG_SLICE_TOOL_SCHEMA)
- tldw_chatbook/Agents/agent_runtime.py (LoopDeps fields, dispatch branches)
- tldw_chatbook/Agents/agent_service.py (real closures, schema/deps wiring,
  RUN_LOG_PROMPT_SECTION extended to mention both new tools)
- Tests/Agents/test_run_log_search.py (pure-function coverage, extended)
- Tests/Agents/test_run_log_stats_slice_runtime_tools.py (new: schema,
  dispatch, isolation, junk-args via the real closures, empty/single/
  multi-segment logs, boundedness on a large synthetic log)
- Tests/Agents/test_agent_models.py, Tests/Agents/test_install_skill_runtime_tool.py
  (RUNTIME_TOOL_NAMES exhaustive-set pins updated -- the same maintenance
  every prior runtime-tool addition required in these two files)
- backlog/tasks/task-1273 (cross-run search follow-up, filed rather than
  built here)
<!-- SECTION:NOTES:END -->
