---
id: TASK-34383
title: >-
  File Notes delete-safety completion: restore refusal with export fallback
status: To Do
assignee: []
created_date: '2026-10-04'
labels:
  - notes
  - library
  - file-notes
dependencies: []
priority: high
---

## Description

Reslice of the superseded TASK-399 B-phase under the SHIPPED ADR-029 design
(one SQLite replica, disk authority; see the 2026-10-04 TASK-399 arc
reconnaissance). Delete/restore landed in minimal form (two-press confirm, snapshot+tombstone, most-recent restore). Missing: refuse restore to occupied or missing-parent paths and offer exact-export fallback instead.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] Restore refuses an occupied destination path (no-replace) with a clear reason
- [ ] Restore refuses a missing parent directory with a clear reason
- [ ] Both refusals offer the existing exact-export as the fallback action
- [ ] Tests pin both refusal shapes and the fallback path
<!-- AC:END -->

## Implementation Plan (to be added when claimed)

