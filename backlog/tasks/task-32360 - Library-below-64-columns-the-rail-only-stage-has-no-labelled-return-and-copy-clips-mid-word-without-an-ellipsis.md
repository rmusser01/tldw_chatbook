---
id: TASK-32360
title: >-
  Library below 64 columns: the rail-only stage has no labelled return and copy
  clips mid-word without an ellipsis
status: To Do
assignee: []
created_date: '2026-09-11 06:18'
labels:
  - library
  - layout
  - critique-10
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
At 60x24 on the rail-only stage Escape is the only return and the footer never says so; strings clip without an ellipsis; reader-shell grips overpaint list rows (B D8 caps 56-59; PROVEN library_adaptive_reader_shell.py:151-156). Evidence: critique #10 snapshot (.impeccable/critique/2026-09-11T06-13-11Z__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 1f3184655b, captures under the session scratchpad crit10/A and crit10/B).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The footer names the return on every single-stage surface
- [x] #2 Clipped copy ends in an ellipsis
- [ ] #3 Grips never overpaint content
<!-- AC:END -->

## Implementation Notes (AC#2 only)

Handed to the `notes-details` branch by the coordinator because the two
measured losses are both composed in `library_notes_canvas.py`. AC#1 and AC#3
belong to the layout branch and are untouched here.

**The handed-over diagnosis did not survive measurement.** The compact gate
(`legacy_compact` in `library_notes_controller.py`) is NOT the defect: at 60x24
on the Notes route `_notes_state.compact` is `True`, `.library-notes-route`
matches, and `#library-shell-grid` carries `library-notes-compact` --
the compact rules are in effect. What they do at that width is the problem,
and one input never arrives:

1. **The status line** (`#library-notes-authority`) is capped at two rows by
   the compact sheet, and the full line needs three in the 32-cell pane a
   60-column terminal resolves, so "files." was cut off the bottom. Compact
   now drops the `Library notes · ` prefix, which the source strip directly
   above already carries; what is left fits both rows whole.
2. **The browse toolbar** cropped "Select" to "Sel" because three actions
   need 33 cells and the pane has 32, under `overflow-x: hidden`. The action
   that does not fit moves to a row of its own -- decided from the labels
   about to be rendered (a disabled action grows a "○ " marker two cells
   wide, which a width constant cannot see).
3. **Root cause of (2):** the canvas's `pane_width` -- task-32127's
   width-aware toolbar input, resolved from the reader layout -- arrived as
   `0` ("not measured yet") on this route and no later sync ever carried the
   real one, so the toolbar was permanently on its widest shape. The canvas
   now falls back to its own rendered width (`on_resize`, the
   `LibraryRailRowButton` precedent) whenever the screen's number is absent,
   and never writes `pane_width` itself -- that attribute stays the screen's
   contract, which its own pin asserts against the resolved layout.

Nothing ends in an ellipsis because nothing is clipped any more: `text-overflow`
was measured against both losses by the layout branch and has no effect on
either (one is a height clip, the other a container crop).

**Still open, and NOT this branch's file:** at 60 columns the Notes list pane
resolves to 32 cells because `_sync_library_notes_reader_layout_from_shell`
(`library_screen.py:6248-6258`) passes `priority="items"` whenever the list
owns the workflow, and the resolver answers that with a 32/18 split -- 18
cells spent on a work pane holding only "Select a note to edit it here.".
Measured: `resolve_adaptive_reader_layout(60, …, priority=None)` closes the
Items pane instead, and `priority="items"` opens it at 32. Whether the list
should simply take the width at that size is a layout decision.

Pinned by `Tests/UI/test_library_crit10_notes_details.py::
test_notes_copy_is_not_clipped_mid_word_below_64_columns` (red first against
the unpatched canvas), and verified live at 60x24 on a 7-note profile
(`crit10/wave/notes-details/caps/cap-13-power-notes-60x24-no-midword-clip.txt`).
