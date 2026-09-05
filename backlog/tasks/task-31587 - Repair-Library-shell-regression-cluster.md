---
id: TASK-31587
title: Repair Library shell regression cluster
status: Done
assignee: []
created_date: '2026-09-05 05:22'
updated_date: '2026-09-05 08:10'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate and repair current Library shell focus, geometry, notes, and routing regressions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reproduced Library shell failures pass
- [x] #2 Library shell module passes in full
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the Library shell failures and group them by current UI/data contract. 2. Repair stale tests or production behavior with the smallest justified change. 3. Run focused regressions and the full Library shell module. ADR required: no. ADR path: N/A. Reason: this is localized regression maintenance for existing Library shell behavior.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Repaired the Library regression cluster after the Collections retained-reader and paged Notes tree cutovers. Updated stale route, geometry, focus, paging, deep-link, and test-harness synchronization expectations; added tolerant polling for retained-canvas recomposition and intentionally pending workers. Fixed production Notes tree consistency by patching loaded placement records after save and reconciling restored placements after undo. Verification: focused repaired cluster 37 passed; adjacent deep-link/coordinator slice 17 passed; final full Tests/UI/test_library_shell.py run 824 passed in 1451.25s; ruff passes for the modified test module; git diff --check passes. ADR required: no (localized regression maintenance within existing boundaries).
<!-- SECTION:NOTES:END -->
