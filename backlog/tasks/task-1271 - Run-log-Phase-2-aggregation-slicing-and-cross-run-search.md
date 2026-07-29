---
id: TASK-1271
title: Run log Phase 2 — aggregation, slicing and cross-run search
status: To Do
assignee: []
created_date: '2026-07-28 00:00'
labels: [agents, run-log]
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
- [ ] #1 An agent can obtain aggregate statistics over its run log without paging individual records through its context
- [ ] #2 An agent can retrieve a contiguous range of records as a coherent unit
- [ ] #3 Cross-run search is available, or a recorded decision explains why it is deferred and what it would take
- [ ] #4 Every new tool is bounded so a single call cannot blow the context window, consistent with how `search_run_log` bounds its own output
- [ ] #5 New tools follow the established runtime-tool pattern (name constant, `RUNTIME_TOOL_NAMES`, schema, optional `LoopDeps` field, dispatch branch guarded by `deps.X is not None`, primary-agent gating where isolation requires it)
- [ ] #6 Tests cover each tool's bounds and its behaviour on an empty, a partial, and a multi-segment log
<!-- AC:END -->
