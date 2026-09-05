---
id: TASK-31679
title: Load destination CSS in Evals empty-state harness
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:20'
updated_date: '2026-09-05 18:26'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Evals layout tests exercise the real destination stylesheet and retain collapse and hit-test assertions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Three reported Evals empty-state geometry and collapse regressions pass
- [x] #2 The harness loads production stylesheet ownership without weakening assertions
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Test harness alignment with existing destination CSS loading.
1. Reproduce the three XML failures and inspect production CSS routing.
2. Add the missing destination CSS to the shared test harness.
3. Run focused empty-state tests and relevant harness regressions; record evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
EvalsHarness now loads the production screen_feature_evals.tcss beside the boot bundle because it pushes EvalsScreen directly instead of using the app route CSS loader. The missing split sheet caused all three reproduced geometry/collapse failures. All layout and click assertions remain intact; no production CSS change. Removed one pre-existing unused App import while checking this touched harness. Empty-state file104 passed61.14s; shared screen file85 passed122.37s. Ruff lint, changed-block formatter check and diff checks passed. Self-reviewed; no new ADR required for test-only fixture alignment.
<!-- SECTION:NOTES:END -->
