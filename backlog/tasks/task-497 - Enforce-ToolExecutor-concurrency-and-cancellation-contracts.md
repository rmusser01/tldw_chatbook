---
id: TASK-497
title: Enforce ToolExecutor concurrency and cancellation contracts
status: To Do
assignee: []
created_date: '2026-07-24 13:17'
updated_date: '2026-07-24 13:19'
labels:
  - tools
  - reliability
  - workers
dependencies: []
references:
  - backlog/decisions/024-bounded-evaluation-and-tool-worker-execution.md
documentation:
  - Docs/superpowers/specs/2026-07-24-tool-worker-contracts-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make ToolExecutor's configured worker limit effective and ensure timeout and cancellation leave truthful execution history without swallowing cancellation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Timed-out tool calls record a terminal timeout result and do not block later calls
- [ ] #2 Cancelled tool calls record cancelled history and propagate cancellation to the caller
- [ ] #3 Batch execution preserves request order and isolates ordinary per-tool failures
- [ ] #4 Global executor reload replaces configuration without referencing retired worker-pool state
- [ ] #5 Regression tests verify limits, order, timeout, cancellation, and reload behavior
- [ ] #6 The unused thread-pool lifecycle is removed without changing supported async tool behavior
- [ ] #7 Configured max_workers and timeout_seconds are validated as positive bounds before use
<!-- AC:END -->
