---
id: TASK-31237
title: Reader uses its vertical space
status: Done
assignee: []
created_date: '2026-09-04 01:50'
updated_date: '2026-09-04 04:21'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 P2: at 52 terminal rows the Reader's content box ends near row 39 (#library-media-viewer-content max-height: 75vh, _agentic_terminal.tcss) leaving ~10 blank rows below while long documents scroll inside a smaller box; the default-open "Search content…" find input spends 3 more rows on every fresh item (duplicating the Find action); a single-page document still renders two dead "○ Previous ○ Next" pager controls. This is the reading surface of a reading workflow idling a third of its pane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Reader content area grows to fill the remaining pane height at tall terminal sizes (no stranded band below the box)
- [x] #2 The content find input is collapsed until Find is invoked, and Escape re-collapses it
- [x] #3 The pager row is hidden when there is only one page
- [x] #4 The task-31222 regression (fixed 18-row cap under an unstyled 1fr band) stays fixed at small sizes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: fill contract (content box ends at pane bottom, virtual height == container), Find round trip (collapsed → open → Escape collapses), single-page pager absence + multi-page negative control
2. GREEN: #library-media-viewer-content 75vh cap → height:1fr; find_open state + one _close_library_media_find() seam; _compose_pager drops controls on single_page
3. Convert test helpers that typed into the always-open bar to press Find first
4. Live tmux verify at 200x52 and confirm 31222's small-size fix holds
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Three parts. Fill: #library-media-viewer-content goes from the 75vh cap to height:1fr — at 200x52 with a 300-line document the box ends at the pane bottom and the viewer's virtual height equals its container (no stranded band, no double scroll). The old '1fr balloons the pane' CSS note predated task-31222's height:auto wrappers; with them in place 1fr resolves against the viewer remainder, and the 31222 small-size fix still holds. Find collapse: the 'Search content…' bar no longer renders permanently; new find_open state with one _close_library_media_find() seam replacing all 8 query-reset pairs (item open, external open, viewer exit, rail switch, delete, mode change, Escape, entry guard), threaded through the viewer constructor AND the in-place _sync_library_media_viewer_state compare/assign. Focus is taken by the bar's own post-mount hook (three screen-level call_after_refresh variants lost the race against the nested recompose-mount). Qodo round: an open Find bar is a reader substate — Escape closes it first from any reader-region focus incl. after F6, and the footer says 'esc close find'. Dead pager: a single-page list renders no Previous/Next controls (supersedes task-28016's keep-disabled-controls choice per the critique ruling); the range Static stays; controls return with a second page; Retry still renders on a failed single-page fetch. Test helpers now press Find before typing. Live-verified at 200x52. Shipped in PR #2367 with task-31235.
<!-- SECTION:NOTES:END -->
