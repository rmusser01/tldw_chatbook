---
id: TASK-32044
title: >-
  Library media: a regional-indicator flag keyword miscounts by two cells and
  overpaints the row frame
status: Done
assignee: []
created_date: '2026-09-08 14:36'
updated_date: '2026-09-08 15:50'
labels:
  - library
  - media
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #7 P2 (the visible consequence of task-31955's stated flag-pair ceiling). A keyword containing a regional-indicator flag emoji (e.g. the pair rendered as one flag) is width-miscounted: the row frame drifts +2 cells (border at column 237 vs 235) and the reader header overpaints (e.g. 'NoNo analysis yet.'). task-31955 cut keyword reasons by display cells with rich.cells but documented that regional-indicator pairs are not fully handled; this is that ceiling showing a real frame break.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A keyword reason containing a regional-indicator flag emoji does not drift the row frame width or overpaint neighbouring panes at 235x52 or 100x30
- [x] #2 The cut is flag-pair-aware (a UAX #29 segmenter, or the reason refuses to paint a half-flag / cuts before the pair) rather than mis-measuring the pair
- [x] #3 A pin paints a flag-keyword reason and asserts the row frame stays aligned
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause (mechanism a): `chop_cells(keyword, 10)` fills the 10-cell budget to the tail and splits a regional-indicator FLAG pair, leaving a LONE indicator (odd trailing RI run); rich's cell_len counts a lone RI as 1 cell but the terminal boxes it as 2 cells → the row frame drifts +2 (border at col 237 vs 235) and the reader header overpaints. Fix (dependency-free): `_trailing_regional_indicators()` counts the trailing U+1F1E6..U+1F1FF run over code points; when it is ODD, drop one dangling indicator (head[:-1], sets cut=True) so a half-flag never paints. A whole pair in budget is kept; a keyword with no flags / ending in ascii / ending in a full flag is untouched; the ZWJ/CJK/zero-width handling (task-31955) and the empty-head/ellipsis guard are unchanged. The reviewer confirmed 'no lone RI survives' is the necessary+sufficient condition for no frame drift (once no half-flag remains, rich and the terminal agree at 2 cells for whole flags). The task-31955 ceiling pin was renamed + inverted (was: accepts a half-flag; now: `test_flag_pair_keyword_never_paints_a_half_flag` asserts the trailing RI run is even). Painted pin `test_flag_keyword_reason_paints_no_half_flag`; caveat: the headless compositor measures cells like rich so it pins the no-lone-RI INVARIANT, not the terminal-font drift (documented ceiling). Fold-in: the test now imports the production `_trailing_regional_indicators` instead of a local copy. Files: library_media_state.py, Tests/Library/test_library_media_state.py, Tests/UI/test_library_media_render_fixes.py, Docs/User_Guide/library/media-and-conversations.md.
<!-- SECTION:NOTES:END -->
