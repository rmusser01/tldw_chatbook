---
id: TASK-497
title: Enforce ToolExecutor concurrency and cancellation contracts
status: Done
assignee:
  - '@codex'
created_date: '2026-07-24 13:17'
updated_date: '2026-07-24 16:36'
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
- [x] #1 Timed-out tool calls record a terminal timeout result and do not block later calls
- [x] #2 Cancelled tool calls record cancelled history and propagate cancellation to the caller
- [x] #3 Batch execution preserves request order, isolates ordinary per-tool failures, and cancels and drains every unfinished sibling before propagating cancellation or unexpected child control flow
- [x] #4 Global executor reload replaces configuration without referencing retired worker-pool state
- [x] #5 Regression tests verify limits, order, timeout, single-call and batch cancellation cleanup, absence of orphan-task warnings, and reload behavior
- [x] #6 The unused thread-pool lifecycle is removed without changing supported async tool behavior
- [x] #7 Configured max_workers and timeout_seconds are validated as positive bounds before use
- [x] #8 Tool history remains bounded and payload-free with exactly one terminal record per started call, including queued and executing cancellation
- [x] #9 A rejected cross-thread MCP submission closes the unsubmitted coroutine and the closed-loop regression passes with unawaited-coroutine warnings treated as errors
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Replaced the unused thread pool with one validated async semaphore around
  uncached `Tool.execute()` work; queue time is outside the per-execution
  timeout and capacity releases on every terminal path.
- Added terminal, metadata-only cancellation history across cache lookup,
  semaphore wait, tool execution, and cache write while preserving propagated
  `CancelledError`, bounded history, immediate results, and cache behavior.
- Made batches own explicit ordered child tasks and always cancel and drain
  unfinished work before propagating parent/child cancellation or unexpected
  control flow; ordinary tool failures remain ordered result dictionaries.
- Removed pool shutdown/destructor/reload state and updated the TASK-492
  privacy/sentinel regressions for the async-only lifecycle.
- Made MCP bridge coroutine ownership explicit: rejected pre-transfer
  submissions close the local coroutine once, while submitted coroutines
  remain owned by the target loop.
- Reused ADR-024; no new ADR was required.
- Verification: warning-strict ToolExecutor/MCP gate `76 passed`; full
  Tool/MCP/Console plus sentinel/privacy integration gate `188 passed`; Ruff,
  changed-source compileall, and `git diff --check` passed. The integration
  gate's single warning is the existing requests dependency-version warning.
<!-- SECTION:NOTES:END -->
