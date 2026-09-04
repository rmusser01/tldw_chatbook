---
id: TASK-31275
title: List returns stale after Restore from Trash and demands a manual Retry
status: To Do
assignee: []
created_date: '2026-09-04 13:54'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 P2: after restoring an item in Trash and pressing `‹ Media`, the list comes back with `Media changed; retry to load a current page.`, every row and action rendered `○`, and `Page boundary is unknown.` until the user presses Retry (B cap_104-110). The app itself made the change, so it knows the page is stale; the honest-stale gate exists for external changes, not for the app's own mutations.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After Restore or permanent delete in Trash, returning to Media shows a fresh list without pressing Retry
- [ ] #2 The stale-list gate still fires for changes the app did not make itself
- [ ] #3 Test plus live verification
<!-- AC:END -->
