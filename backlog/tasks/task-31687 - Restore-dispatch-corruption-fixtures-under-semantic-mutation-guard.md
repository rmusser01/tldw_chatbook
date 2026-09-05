---
id: TASK-31687
title: Restore dispatch corruption fixtures under semantic mutation guard
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:27'
updated_date: '2026-09-05 18:37'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Exercise repository quarantine and compare-and-swap predicates with explicit corruption fixtures while preserving production mutation guards.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All four reported dispatch ownership corruption cases reach and verify repository rejection
- [x] #2 No semantic mutation guard or expected predicate is removed or weakened
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Test-only corruption fixture alignment with existing private authorization support.
1. Reproduce four guarded raw UPDATE failures and inspect established corruption helpers.
2. Reuse the bounded test corruption helper for only those injected corruptions.
3. Run repository and semantic guard regression tests, lint, and record evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reused the established one-statement _raw_semantic_corruption test helper for four deliberate invalid message-owner mutations. It restores the real authorization callback in finally; SQL guards and all quarantine/CAS predicates remain intact. No production changes or new ADR. Four RED cases reproduced; full dispatch repository plus semantic-guard migration tests:71 passed34.55s. Ruff lint, changed-block formatting and diff checks passed; self-reviewed against existing recovery fixture usage.
<!-- SECTION:NOTES:END -->
