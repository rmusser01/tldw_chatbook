---
id: TASK-31957
title: >-
  Library media - the preview pane does not show the analysis marker the row
  does
status: Done
assignee: []
created_date: '2026-09-07 08:26'
updated_date: '2026-09-07 20:20'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
I final review M6: task-28008's description asked for the analysis marker in the row secondary line and in the preview pane; only the row line shipped, a deliberate v1 omission because the acceptance criteria did not require the pane. The preview pane (library_media_state.py ~1090) still shows Title / Type / Updated, so the two surfaces disagree about the same item.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The preview pane reports the same analysis state as the row's secondary line
- [x] #2 A painted pin covers an analysed and an un-analysed item
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Painted pins: an analysed item's preview pane says so; an un-analysed one does not. 2. Add the analysis state to the browse projection's preview lines from the same `has_analysis` the row uses.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The browse projection's `preview_lines` now carry `Analysed: yes|no` from the same `has_analysis` the row's secondary line uses. Closed on the state contract: the preview pane is DEAD in production — every Media canvas path passes `show_preview=False` since d99fb4a9c (the Reader is the detail half) — so the change is invisible; the painted pin flips the flag on over the real screen and says so. The legacy `build_library_media_state` preview keeps its three lines. Rider: delete the dead pane (builder, canvas branch, two CSS tiers) or re-enable it for the compact layout — a product call.
<!-- SECTION:NOTES:END -->
