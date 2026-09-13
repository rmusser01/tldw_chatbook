---
id: TASK-31763
title: Repair Console workbench contract regressions
status: Done
assignee: []
created_date: '2026-09-05 05:23'
updated_date: '2026-09-05 08:46'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate and repair current Console workbench layout, recovery, and inspector contract failures.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reproduced Console workbench failures pass
- [x] #2 Console workbench contract module passes in full
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the Console workbench contract module and classify each failure by layout, recovery, inspector, or stale test contract.
2. Apply the smallest test or production repair justified by current Console behavior, preserving existing service and UI boundaries.
3. Run focused regressions, the full Console workbench contract module, Ruff, and diff checks.

ADR required: no.
ADR path: N/A.
Reason: localized regression repair within established Console workbench contracts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Repaired the Console workbench contract cluster by aligning stale tests with current production behavior: explicitly reveal the Inspector before asserting visible content, pass structured readiness action metadata instead of parsing display copy, establish the full streaming snapshot before measuring equality-gated refreshes, load the production app stylesheet set in direct header harnesses, and verify narrow prompt actions through explicit scroll reachability. Removed a brittle historical minimum-draft-width assertion that was unrelated to the recovery-visibility contract. No production code change was required.

Verification: Tests/UI/test_console_workbench_contract.py — 72 passed; residual focused set — 4 passed; CSS build integrity — 19 passed; Ruff and git diff --check passed.

ADR required: no. ADR path: N/A. Reason: test-only maintenance for established Console contracts.
<!-- SECTION:NOTES:END -->
