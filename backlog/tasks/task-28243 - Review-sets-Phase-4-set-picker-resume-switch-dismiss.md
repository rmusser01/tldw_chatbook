---
id: TASK-28243
title: 'Review sets - Phase 4: set picker (resume / switch / dismiss)'
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-02 22:29'
updated_date: '2026-09-03 04:21'
labels:
  - library
  - media-ux
dependencies:
  - TASK-28240
  - TASK-28241
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A lightweight picker to resume, switch between, or dismiss saved review sets (design: backlog/docs/design-library-review-sets.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A picker opened from the media list lists saved sets with name + progress (X of M, reviewed N); selecting one activates it and loads at its cursor
- [ ] #2 The picker can dismiss (soft-delete) a set and reopen a completed one; activating a set deactivates the previously active one (one-active invariant)
- [ ] #3 Reuses the Library choice-strip / picker idioms; no new rail row required for v1
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pure build_picker_rows in review_set_state (rows = id/name/live-progress/active) - TDD
2. Dumb modal LibraryReviewSetPickerDialog (SafeModalDismissMixin, decision tuple open/dismiss/None) - TDD via ModalHarness
3. Screen worker: collect rows off-loop (one union liveness resolve), push_screen_wait, apply decision (reopen-if-completed + activate + land at resolved cursor; dismiss + notice; empty set = notice, never activated)
4. 'Sets' toolbar button on the media list (no rail row, hidden in select mode)
5. Live tmux verify + docs stamp
<!-- SECTION:PLAN:END -->
