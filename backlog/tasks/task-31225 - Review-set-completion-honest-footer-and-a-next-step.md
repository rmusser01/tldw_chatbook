---
id: TASK-31225
title: Review-set completion - honest footer and a next step
status: Done
assignee: []
created_date: '2026-09-03 22:31'
updated_date: '2026-09-04 00:31'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Re-critique P2: after 'All N reviewed' the footer keeps advertising '] next in set' (now a silent no-op, violating the task-28005 honest-footer rule) and offers no next step. Riders: the storage-unavailable notice ships error twice/warning once (normalize); investigate B's unattributed Space-in-empty-select-mode canvas blank at 100x30 (pane-grip hypothesis).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A completed set's footer stops advertising ] and keeps R (and m for un-marking)
- [x] #2 Completion offers a next step
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shipped in PR #2359. _review_footer_entries (extracted, unit-tested): a COMPLETE set shows 'm toggle reviewed | R finish review | All N reviewed' with no ]; un-marking restores the walk keys (live round trip). Severity rider documented in code. OPEN rider: Assessment B's Space-in-empty-select-mode canvas blank at 100x30 remains unconfirmed (pane-grip hypothesis).
<!-- SECTION:NOTES:END -->
