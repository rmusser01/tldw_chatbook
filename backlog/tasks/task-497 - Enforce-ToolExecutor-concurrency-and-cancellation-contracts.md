---
id: TASK-497
title: Enforce ToolExecutor concurrency and cancellation contracts
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-24 13:17'
updated_date: '2026-07-24 15:45'
labels:
  - tools
  - reliability
  - workers
dependencies:
  - TASK-492
references:
  - backlog/decisions/024-bounded-evaluation-and-tool-worker-execution.md
documentation:
  - Docs/superpowers/specs/2026-07-24-tool-worker-contracts-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make ToolExecutor's configured worker limit effective, ensure timeout and cancellation leave truthful execution history without swallowing cancellation, and close MCP bridge coroutines when cross-thread submission rejects before ownership transfer.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Timed-out tool calls record a terminal timeout result and do not block later calls
- [ ] #2 Cancelled tool calls record cancelled history and propagate cancellation to the caller
- [ ] #3 Batch execution preserves request order, isolates ordinary per-tool failures, and cancels and drains every unfinished sibling before propagating cancellation or unexpected child control flow
- [ ] #4 Global executor reload replaces configuration without referencing retired worker-pool state
- [ ] #5 Regression tests verify limits, order, timeout, single-call and batch cancellation cleanup, absence of orphan-task warnings, and reload behavior
- [ ] #6 The unused thread-pool lifecycle is removed without changing supported async tool behavior
- [ ] #7 Configured max_workers and timeout_seconds are validated as positive bounds before use
- [ ] #8 Tool history remains bounded and payload-free with exactly one terminal record per started call, including queued and executing cancellation
- [ ] #9 A rejected cross-thread MCP submission closes the unsubmitted coroutine and the closed-loop regression passes with unawaited-coroutine warnings treated as errors
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/024-bounded-evaluation-and-tool-worker-execution.md
Reason: TASK-497 implements ADR-024 and preserves completed TASK-492's metadata-only terminal history boundary.

1. Pin constructor validation, true peak concurrency, and queue-versus-execution timeout behavior with red tests.
2. Apply one semaphore to actual tool execution and make every begun cancellation path append one terminal payload-free record and re-raise.
3. Preserve request order while explicitly cancelling and draining every batch child on cancellation or unexpected control flow.
4. Remove the unused thread-pool reload/shutdown lifecycle without adding a replacement manager.
5. Close MCP execution coroutines when cross-thread submission rejects before ownership transfer, with RuntimeWarning promoted to error.
6. Run warning-strict Tool/MCP tests, relevant agent/chat integrations, TASK-492 privacy regressions, lint, compile, and diff verification before task reconciliation.

Detailed plan: Docs/superpowers/plans/2026-07-24-tool-worker-contracts.md
<!-- SECTION:PLAN:END -->
