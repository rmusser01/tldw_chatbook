---
id: TASK-32829
title: Keep MCP lifecycle cancellation busy until cleanup finishes
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 01:17'
updated_date: '2026-09-19 02:00'
labels:
  - mcp
  - ui
  - lifecycle
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Cancelling a connection must not permit another action on the same server while the first operation is still cleaning up or erase a later operation busy state. Cancellation feedback must describe actual progress and permit a clean retry after completion.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Cancellation retains same-server admission until the original lifecycle and cleanup finish; other servers remain independent.
- [x] #2 Repeated or stale cancellation cannot interrupt cleanup or falsely report a finished success as cancelled.
- [x] #3 Cancellation before worker start does not launch the service or leak an unawaited coroutine; retry becomes available after settlement.
- [x] #4 Checking and cancelling feedback remains visible and truthful at compact and wide sizes; targeted and bounded native evidence documents transport limits.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: backlog/decisions/161-component-pattern-library.md. Reason: repair lifecycle worker admission and presentation lifetime within existing transport, timeout, permission and service contracts. 1. Pin slow cancellation cleanup, repeated Cancel, cancel-before-start and post-settlement retry with isolated mounted regressions. 2. Retain per-server ownership through actual worker settlement, defer service coroutine creation to execution, and make cancellation progress truthful. 3. Verify targeted lifecycle neighbors and real-app compact/wide dark/light cancellation and recovery evidence; document controlled delay and external transport boundaries. 4. Independent review, update the task and review ledgers, and save a separate bounded draft PR against dev. Preserve unrelated work and the separate PR2711/2712 gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Keep per-server lifecycle ownership until the original worker and final readiness collection settle. Inspector Cancel binds the displayed control, server and operation; stale/repeated intents cannot cancel replacements or interrupt cleanup. Lazy lifecycle/redraw callbacks prevent pre-start coroutine leaks. Ten new regressions and eleven UI neighbors pass; thirteen token/bundle checks pass (34 distinct focused cases). Four native dark/light compact/wide cells and twelve inspected captures qualify visible cancellation progress and retry with controlled client cleanup/failure through the real control-plane/persistence path. Native shutdown, lock, ten DBs, default-file fingerprints and source hashes pass. Seven preflight guards pass; no introduced Ruff diagnostics versus dev. Independent review findings fixed; final review has no blockers. Four unrelated service tests reproduce raw_source_selection_changed on unchanged dev and remain documented baseline debt. Existing ADR-161 applies; no new ADR or transport/security/token contract change. Evidence: Docs/superpowers/qa/2026-09-18-mcp-lifecycle-cancellation/README.md. The separate draft PR retains CI, remote review and owner visual approval gates.

Saved independently as draft PR #2713 against dev: https://github.com/rmusser01/tldw_chatbook/pull/2713. Current-head CI, accumulated review and owner visual approval remain merge gates; broader MCP external connection and execution review remains open.

Real stdio follow-up reproduced native eager-worker completion leaving a SUCCESS worker busy on unchanged dev. The existing observer already fixes it; added an isolated regression using asyncio.eager_task_factory, red on dev and green here, with immediate retry. Total now 35 distinct passing cases. Real comparison connection/refresh/disconnect/reconnect exited cleanly; cached catalog refresh is a separate next-review issue. No production change in this follow-up.
<!-- SECTION:NOTES:END -->
