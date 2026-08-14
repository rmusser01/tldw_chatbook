---
id: TASK-16263
title: Reconcile Phase 3 and Phase 6 release harness contracts
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 17:19'
labels:
  - testing
  - ui
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore deterministic Phase 3/6 and Library navigation evidence after handoff composition, shell focus, provider recovery, targeted reconciliation, and dependency pin contracts evolved.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Study handoff tests assert the current mounted and hidden-state contracts.
- [x] #2 Release replay tests use current provider and Library canvas evidence.
- [x] #3 Footer and File Notes fixtures synchronize with current recompose and workspace protocols.
- [x] #4 All twelve reproduced regressions and affected modules pass with static checks.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve and classify the twelve checkpoint failures.
2. Apply minimal test-only updates at each changed contract boundary.
3. Run named regressions, then the full affected module slice.
4. Run Ruff, formatter, and diff checks and record closeout.

ADR required: no
ADR path: N/A
Reason: test-harness reconciliation only; no production architecture or behavior change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

- Reconciled twelve stale Phase 3/6, footer, and navigation assertions with the current mounted-widget, provider-recovery, focus, dependency-pin, and targeted Library reconciliation contracts.
- Added readiness waits where deferred home/navigation copy made isolated tests pass but combined-load execution race.
- Verification: the full eight-module affected slice passed with 160 tests; targeted Ruff lint and `git diff --check` passed.
- Ruff format still identifies the same three files that were already unformatted at the implementation base; their existing formatting was preserved to avoid unrelated churn.
- No production code, dependency, or ADR changes were required.
