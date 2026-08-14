---
id: TASK-16235
title: Reconcile runtime-policy full-app evidence with Library
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 09:31'
updated_date: '2026-08-14 09:32'
labels:
  - runtime-policy
  - testing
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep full-application runtime-source coordination evidence on the canonical Library destination and ensure bounded coordinator diagnostics render their exception categories.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Runtime-policy full-app tests mount Library instead of retired MediaScreen
- [x] #2 Precommit failure and CAS-rejection surfaces remain unchanged
- [x] #3 Successful runtime changes invoke the current screen callback once and contain callback failures
- [x] #4 Coordinator warnings render exception categories without private values
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the five full-app failures as RED evidence.
2. Update actual-screen callback tests to the canonical `LibraryScreen` route.
3. Correct Loguru placeholder rendering in the coordinator and run the full module plus static checks.

ADR required: no
ADR path: N/A
Reason: This is a canonical-route and diagnostic bug fix within the existing runtime-policy boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Moved full-app runtime-policy evidence from the retired MediaScreen route to the canonical LibraryScreen owner while preserving precommit, CAS, commit/rebind, callback, and invalidation assertions. Corrected the coordinator's two Loguru placeholders so exception categories render and private exception values remain absent. Verification: the complete full-app runtime-policy module passed 16 tests; Ruff lint and py_compile passed; git diff --check passed. app.py retains its identical pre-task Ruff-format baseline.
<!-- SECTION:NOTES:END -->
