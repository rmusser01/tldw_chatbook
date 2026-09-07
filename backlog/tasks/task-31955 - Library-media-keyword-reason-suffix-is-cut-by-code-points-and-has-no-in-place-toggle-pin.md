---
id: TASK-31955
title: >-
  Library media - keyword-reason suffix is cut by code points and has no
  in-place toggle pin
status: To Do
assignee: []
created_date: '2026-09-07 08:26'
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
- [ ] #1 The reason suffix is cut on a cell-width-aware boundary and never splits a grapheme cluster
- [ ] #2 Pins cover a wide/CJK reason and an emoji-cluster reason
- [ ] #3 A pin asserts the analysed marker and the keyword suffix both survive a density toggle and a select-mode toggle in place
<!-- AC:END -->
