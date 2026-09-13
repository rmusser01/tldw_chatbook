---
id: TASK-25713
title: Census warm-boot flakes on sys.modules mutation during iteration
status: Done
assignee:
  - '@Robert'
created_date: '2026-08-31 14:12'
updated_date: '2026-08-31 14:25'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The ui_ready module census script iterates sys.modules live while background threads import modules, raising RuntimeError: dictionary changed size during iteration and failing the UI latency guardrails job intermittently (observed on PR #2255 and another branch, 2026-08-31). Snapshot the dict before iterating.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The census script iterates a point-in-time snapshot of sys.modules, no live-dict iteration remains in the script,Census test passes repeatedly (3+ consecutive local runs),No other live sys.modules iterations in the test file
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Snapshot sys.modules before the census genexpr in _CENSUS_SCRIPT.\n2. Verify: three consecutive census runs green; grep for other live iterations in the file.\n3. PR, review, merge.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`_CENSUS_SCRIPT` now takes `list(sys.modules)` once at census time and the
genexpr iterates that snapshot; the in-dict `sys.modules[m]` lookups are
plain gets and safe. A point-in-time copy is also the honest census: every
counted module existed at the same instant instead of straddling whatever a
background thread imported mid-walk. Verified: three consecutive local
runs green; grep confirms no other live iterations in the file; Qodo review
clean (0 findings). ADR: none -- test-harness bug fix.
<!-- SECTION:NOTES:END -->
