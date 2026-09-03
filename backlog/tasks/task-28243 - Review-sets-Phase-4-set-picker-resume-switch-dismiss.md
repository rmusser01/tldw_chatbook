---
id: TASK-28243
title: 'Review sets - Phase 4: set picker (resume / switch / dismiss)'
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-02 22:29'
updated_date: '2026-09-03 04:48'
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented on feat/review-sets-p4 (stacked on p3 #2335). build_picker_rows pure rows (review_set_state) over ONE union liveness snapshot; dumb LibraryReviewSetPickerDialog (SafeModalDismissMixin, decision tuple, literal-Text labels, esc/Close cancel); screen worker collects rows off-loop, push_screen_wait, applies decision off-loop (open = reopen-if-completed + activate + land at resolved cursor; all-tombstoned never activated; dismiss = soft-delete + notice + viewer sync). 'Sets' opener on the media list TITLE row - the action toolbar overflows the narrow Items pane (28025) and one more button crashed rich chop_cells at width 2 (live-verified); not composed on the fresh-empty page (pins ONE recovery action). Live tmux end-to-end pass: create -> walk -> exit -> picker resume at cursor -> dismiss drops footer chrome. Docs: media-and-conversations.md Review sets section. Traps: byte-vs-cell tmux column arithmetic (BSD awk/cut count bytes) gave false negative clicks; Static defaults to 1fr in Horizontal and swallowed the title row.
<!-- SECTION:NOTES:END -->
