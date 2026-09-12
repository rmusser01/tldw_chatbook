---
id: TASK-31238
title: Review-set replacement notice and distinguishable picker rows
status: Done
assignee: []
created_date: '2026-09-04 01:50'
updated_date: '2026-09-04 03:05'
labels:
  - library
  - media-ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 P3: creating any review set silently deactivates the currently active one (one-active invariant) without a word — a walk in progress just stops being resumable-by-] with no acknowledgment. And auto-names ("2 selected items") make picker rows indistinguishable later: two selection-sets render identical rows. Fix: toast the pause ("Paused 'Read later' at 1 of 2") when a create deactivates an in-progress set, and add a distinguishing detail (created date or first-item title) to picker rows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Creating a set while another is active surfaces a notice naming the paused set and its progress
- [x] #2 Picker rows carry a distinguishing detail beyond the auto-name so identical-name sets can be told apart
- [x] #3 No notice fires when no active set was displaced
<!-- AC:END -->

## Implementation Plan

1. RED: build_picker_rows date suffix (incl. same-name distinguishability + malformed timestamp) and the create-pause notice trio
2. GREEN: _with_created_date in review_set_state; displaced-set capture + Paused notice in _create_and_open_review_set
3. Live tmux verify (fast capture for the toast)

## Implementation Notes

Picker rows: build_picker_rows suffixes the detail label with the created DATE (" · YYYY-MM-DD", omitted for malformed timestamps) — tuple shape unchanged, so the picker widget needed zero changes. Pause notice: _create_and_open_review_set captures get_active_review_set() BEFORE creating; when the displaced set is incomplete (completed_at is None) it toasts "Paused '<name>' at <live progress>. Resume from Sets." — name markup-escaped, progress computed live via _review_set_live_ids so the toast can never disagree with the picker. No notice when nothing was displaced or the displaced set was complete. Live-verified: toast captured at +1s; picker rows show the date (wraps at narrow widths but renders).
