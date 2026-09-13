---
id: TASK-32212
title: >-
  Library rail: the 'Search Library…' row is two cells too wide and clips the
  canvas frame on every canvas (clear-button regression)
status: Done
assignee: []
created_date: '2026-09-10 14:53'
updated_date: '2026-09-10 19:04'
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
- [x] #1 The rail search row (Input + clear button) fits the rail's inner width at 235, 100 and 60 columns; the canvas frame is intact
- [x] #2 A painted-cell test pins the frame columns
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Painted-frame test FIRST at 235x52/100x30/60x24 -- record the real regions before touching anything.
2. Fix only what the measurement justifies.
3. Rebuild the CSS bundle; re-measure at all three widths.
4. Live-verify the frame columns on both profiles.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
MEASURED BEFORE FIXING, and the critique's inferred cause was wrong. Every widget region already sat inside the rail on dev at all three widths (row.region.right <= rail.region.right, clear.region.right <= rail.region.right, box.region.right <= clear.region.x all passed). What overflowed was the clear button's own CONTENT: Textual's `Button` carries `line-pad: 1`, which flanks each rendered line with a cell, so "x" painted as border(2) + " x "(3) = 5 cells inside its 3-cell box -- and Textual does not clip that. The search row's middle line therefore painted two cells long, moving the rail's right border from column 41 to 43 and the canvas's left border with it, exactly the pipe columns the critique recorded (0-based [0,2,42,43,234,236] instead of [0,2,40,41,232,234]).

Fix: zero the button's `line_pad` inline in `LibraryRail.compose`, next to the styles it already sets there (`LibraryNavigationRailHandle.compose` sets `line_pad = 0` the same way). It cannot be written in TCSS: Textual's `_process_integer` rejects a literal 0 for `line-pad`, so the stylesheet fails to parse. The TCSS block carries a comment saying so.

The new test pins the PAINTED frame column, which is what actually reproduced the defect -- region assertions alone were green on dev. The canvas assertion is skipped below the compact breakpoint where the canvas is deliberately not mounted (task-32066).

Live: pipe columns constant at [0,2,40,41,232,234] (235x52), [0,28] (100x30), [0,59] (60x24) on both profiles.

Files: tldw_chatbook/Widgets/Library/library_rail.py, tldw_chatbook/css/components/_agentic_terminal.tcss (+ regenerated bundle), Tests/UI/test_library_crit9_rail.py
<!-- SECTION:NOTES:END -->
