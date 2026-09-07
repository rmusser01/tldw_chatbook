---
id: TASK-31960
title: >-
  Library media - Review these is the only list-wide action outside the
  failed-load gate
status: Done
assignee: []
created_date: '2026-09-07 08:27'
updated_date: '2026-09-07 20:34'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
J final review M1: every list-wide action routes through the failed-load gate except 'Review these', which keeps _gate_stale_action alone, so on a failed first page it stands live beside a dimmed Export. It is defensible - its worker re-fetches and notifies on failure - but the asymmetry is unexplained at the surface and is one line to remove if symmetry is what we want.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The list-wide actions present one consistent enabled state on a failed first page, or the exception is documented where the gate is defined
- [x] #2 The decision is recorded either way
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pin: on a failed first page `○ Review these` is disabled with the same reason Export shows, at 235×52 and 100×30; live on retained rows and on a healthy list. 2. One line of symmetry at the gate if the exception is not deliberate; otherwise document it.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Symmetry, one line: `_gate_failed_action(review_btn, "Review these")` runs before the stale gate in the same order Export uses, so a whole-list action never targets a list that failed to load; `#library-media-review` has one press handler, so gating the button gates the action. Trash and the callout's own Retry stay live (Trash acts on retained rows; Retry is the recovery). The decision and what stays ungated are recorded in `_gate_failed_action`'s docstring and the User Guide. Pinned at both widths, live over retained rows and on a healthy list.
<!-- SECTION:NOTES:END -->
