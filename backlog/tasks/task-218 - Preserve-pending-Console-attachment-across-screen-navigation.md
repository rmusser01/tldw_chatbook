---
id: TASK-218
title: Preserve pending Console attachment across screen navigation
status: To Do
assignee: []
created_date: '2026-07-13 09:30'
labels:
  - console
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 1 (PR #621) deliberately drops a staged-but-unsent attachment when the user navigates away from Console (screen-state serialization is metadata-only by spec; raw bytes never serialize). Decide and implement a preservation strategy that keeps the no-bytes-in-screen-state constraint (e.g. re-process from file_path on restore, or a bounded in-memory app-level stash).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Staging an attachment, visiting another destination, and returning to Console preserves the pending attachment (or shows an explicit, honest notice if the source file vanished)
- [ ] #2 Raw attachment bytes still never enter screen-state serialization
- [ ] #3 Behavior covered by a mounted navigation round-trip test
<!-- AC:END -->
