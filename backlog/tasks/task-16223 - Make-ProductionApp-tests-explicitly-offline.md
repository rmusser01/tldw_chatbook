---
id: TASK-16223
title: Make ProductionApp tests explicitly offline
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 08:25'
updated_date: '2026-08-14 08:39'
labels:
  - tests
  - security
  - network
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove timing-dependent external model-catalog refresh attempts from ProductionApp tests while preserving production startup behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The performance-to-ProductionApp ordering no longer attempts network egress.
- [x] #2 ProductionApp and focused static gates pass apart from independently attributed stale assertions.
- [x] #3 ProductionApp tests replace only the model-catalog startup refresh with an offline no-op.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the timing-dependent egress with startup-performance followed by a ProductionApp route test
2. Configure the ProductionApp private sandbox to disable model catalog auto-refresh before app construction
3. Rerun the reproducing order and ProductionApp suite to distinguish egress errors from independent assertion drift
4. Run static checks, document evidence, and close the task

ADR required: no
ADR path: N/A
Reason: this is a test-isolation repair; production configuration and network policy are unchanged.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a ProductionApp-local autouse fixture that replaces only TldwCli._refresh_model_catalogs with an async no-op. Dedicated model-catalog tests retain the real refresh behavior; unrelated production-ownership tests no longer race a fake-key HTTPS refresh against the process-wide network guard.

Evidence: the exact app-import/startup/RAG-performance to ProductionApp ordering was RED with blocked HTTPS teardown attempts before the fix, then passed 94/94 in three fresh consecutive processes. The complete ProductionApp directory has no network teardown errors after the fix; its 18 remaining failures are independently attributed stale assertions and are not masked. Ruff lint/format and git diff checks pass.

ADR required: no; production behavior is unchanged.
<!-- SECTION:NOTES:END -->
