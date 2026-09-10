---
id: TASK-32212
title: >-
  Library rail: the 'Search Library…' row is two cells too wide and clips the
  canvas frame on every canvas (clear-button regression)
status: To Do
assignee: []
created_date: '2026-09-10 14:53'
labels:
  - library
  - rail
  - layout
  - regression
  - critique-9
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The rail search line's box pipes sit at columns 1,3,41,43,44,235 instead of 1,3,41,42,233,235; the pane's right border at column 233 is absent, the canvas's left border shifts two cells right and its right border clips, on the landing, Import, Notes and at 100x30 (where the `x` button itself is cut). INFERRED: the clear button added by task-32069 widened the row past its pane. Wave regression. Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 9.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The rail search row (Input + clear button) fits the rail's inner width at 235, 100 and 60 columns; the canvas frame is intact
- [ ] #2 A painted-cell test pins the frame columns
<!-- AC:END -->
