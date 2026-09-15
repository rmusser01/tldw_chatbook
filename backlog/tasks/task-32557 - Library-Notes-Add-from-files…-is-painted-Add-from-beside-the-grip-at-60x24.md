---
id: TASK-32557
title: >-
  Library Notes: "Add from files…" is painted "Add from" beside the grip at
  60x24
status: Done
assignee: []
created_date: '2026-09-13 06:48'
updated_date: '2026-09-14 18:35'
labels:
  - library
  - notes
  - critique-3
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 (dev 5fd502dbac), assessor B, compact terminal. D14. Task-32360 fixed mid-word clipping on the rail-only stage below 64 columns; task-32127's whole-word pin is at 235 wide.

**What happened.** At 60x24 the Notes list toolbar's first row paints `New  Sort: Newest  Select  Add from   s` — "Add from files…" cut to "Add from" against the grip (B 52 line 10). The footer compacts correctly and the status drops its prefix as documented. Captures: B 52.

**Cause.** INFERRED. Docs contradicted: "Nothing is ever painted as half a word."
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 At 60x24 every Notes toolbar label paints whole or is elided with an ellipsis
- [x] #2 A test at 60 columns pins the toolbar labels
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce: fresh 60x24 is clean; resizing a merged 235-wide list down to 60 paints 'New  Sort: Newest  Select  Add from   s' (the critique's exact line).
2. Trace: _effective_pane_width prefers the screen-contract pane_width, which lags one resize behind the resolved layout, so on_resize re-decides from the stale wide value and keeps the merged shape.
3. Prefer the canvas's own measured width; pane_width stays the pre-measurement fallback.
4. Pin the resize round trip through the production canvas.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Cause PROVEN, and the reported state is only reachable by a resize.

A fresh launch at 60x24 paints this toolbar correctly on dev (two rows, 'Add from files…' whole) -- task-32360's work holds. What reproduces the critique's line exactly is narrowing an already-merged wide list: at 235 the list pane is 138 cells and the two action groups share one row; resize to 60 and the canvas kept painting that merged row inside a 50-cell pane, cutting 'Add from files…' to 'Add from' against the grip. Captured verbatim, grip included: 'New     Sort: Newest     Select       Add from   s' (wave4-caps/layout/layout-03-add-from).

**Root cause and where it was fixed.** The canvas learned its pane width only from a state sync (`_build_library_notes_state` passes it at compose); a resize re-resolves the layout on the screen but never told the canvas, so `on_resize` re-decided the toolbar's shape from the width the PREVIOUS compose had. Fixed at that source: `_sync_library_notes_reader_layout_from_shell` now hands the resolved Items width to the canvas through a new `LibraryNotesCanvas.apply_pane_width`, which re-shapes only when the width SHRANK.

**A wrong first fix, and what caught it.** The first attempt inverted `_effective_pane_width` to prefer the canvas's own measured width. That decides the toolbar from MID-LAYOUT numbers: one 60 -> 170 -> 60 round trip delivers this canvas widths of 110, 106, 46, 48, 68, 40, 1 and 72 (logged, not guessed), and the transients under the 48-cell stack threshold recomposed it -- costing the in-place breakpoint path its widget identity and turning `test_library_note_compact_labels_round_trip_without_recompose` red, a test this branch never touched and which is green on dev. Found by the FAILED-name SET comparison against a detached dev baseline. The shrink-only asymmetry is deliberate and documented at the seam: a shape too wide for its pane paints half words and must be answered this frame; a shape too narrow only leaves space and can ride the next compose.

**A regression the review caught in that asymmetry.** The first version did not just skip the re-shape on growth, it dropped the grown width entirely -- and `_effective_pane_width` gives `pane_width` priority over the measured width, so 235 -> 60 -> 235 left the canvas composing the 50-cell shape into a 138-cell pane until an unrelated state sync re-stamped it. That is NOT dev's behaviour (dev's `pane_width` never moves off its compose value, so dev ends that round trip correctly wide), so the first write-up's "nothing regresses" was wrong. A grown width is now RECORDED without refreshing -- one line, no recompose, so the widget-identity pin stays green -- and the round trip is pinned by `test_a_pane_that_widens_again_records_the_width_it_was_given` (RED without it: 'the widened pane was dropped: canvas still holds 50 after 134 -> 50 -> 134').

The pin is through the real screen and a real resize, including the state sync that stamps the contract width -- without that sync the harness canvas has never been told a width at all and falls back to measuring itself, which is not the state the defect lives in. RED without the push: 'actions painted off the 50-column canvas: add-from-files Region(x=44, width=19), export Region(x=64, width=10)'. GREEN with it.

Live after the fix: the identical resize sequence paints 'New  Sort: Newest  Select' / 'Add from files…  Export' / 'New folder', and the status line drops its authority prefix as documented (layout-19-resize-60-after, re-verified at layout-28-resize-60-after-v2).

Files: Widgets/Library/library_notes_canvas.py, UI/Screens/library_screen.py, Tests/UI/test_library_notes_w4_layout.py.
<!-- SECTION:NOTES:END -->
