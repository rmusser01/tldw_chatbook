---
id: TASK-31955
title: >-
  Library media - keyword-reason suffix is cut by code points and has no
  in-place toggle pin
status: Done
assignee: []
created_date: '2026-09-07 08:26'
updated_date: '2026-09-07 20:20'
labels:
  - library
  - media-ux
  - test-debt
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
I Task 3 review: _KEYWORD_REASON_CHARS truncates the keyword reason by code points rather than cells or graphemes, so ten CJK characters occupy twenty cells, an emoji ZWJ cluster can be cut mid-sequence, and interior spaces survive the cut. Separately, neither the analysed marker nor the keyword: suffix has a pin that it survives a density or select-mode in-place toggle - traced by hand to hold, so this is a parity gap rather than a known break.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The reason suffix is cut on a cell-width-aware boundary and never splits a ZWJ/modifier grapheme cluster (regional-indicator flag pairs are a stated, pinned ceiling)
- [x] #2 Pins cover a wide/CJK reason and an emoji-cluster reason
- [x] #3 A pin asserts the analysed marker and the keyword suffix both survive a density toggle and a select-mode toggle in place
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pins: a CJK reason (10 cells, not 20), an emoji ZWJ-cluster reason, interior spaces; an in-place pin that `analysed` and `keyword:` survive a density and a select-mode toggle. 2. Cut by cells with rich's `chop_cells`.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The suffix is cut on a cell boundary with `rich.cells.chop_cells` (`_KEYWORD_REASON_CELLS = 10`; 5 CJK characters fill the budget); ZWJ sequences, VS16/keycap, skin-tone modifiers and combining marks are span-merged by rich and never split. Ceiling, stated beside the cut and in a pin: rich's grapheme splitting is not full UAX #29 — regional-indicator (flag) pairs can halve at an odd offset; a UAX #29 segmenter would need a dependency. A reason whose cut head has no visible cells (zero-width or leading-space-only) drops the suffix instead of painting a dangling `keyword:`. In-place pin: both markers survive the density and select-mode toggles without a recompose. Live: `article · 3m · keyword: 会議記録一…`.
<!-- SECTION:NOTES:END -->
