---
id: TASK-414
title: Skills trust - show content changes in Review changes before Approve
status: To Do
assignee: []
created_date: '2026-07-21 15:18'
labels:
  - skills
  - ux
  - trust
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P1 from the 2026-07-21 Skills UX/NNG review (verified live). The trust flow's Review changes button renders only a comma-joined filename list - which the trust state line already shows - so Approve is effectively blind sign-off on unseen executable skill content. capture_review already returns current_files with full text; the UI just never renders it. Also approve-time snapshot_mismatch surfaces only as a generic warning toast. NNG heuristic 6 (recognition over recall) plus security-UX honesty.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After Review changes the user can see the content of changed files (at minimum a per-file preview; a before/after diff against the trusted baseline preferred) before Approve is enabled,Approve failure due to snapshot mismatch produces a specific actionable message instead of the generic trust-action warning,Review presentation handles multi-file and deleted-file cases without breaking layout,Covered by tests
<!-- AC:END -->
