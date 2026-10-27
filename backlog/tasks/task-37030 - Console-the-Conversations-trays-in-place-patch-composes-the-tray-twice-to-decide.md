---
id: TASK-37030
title: >-
  Console: the Conversations tray's in-place patch composes the tray twice to
  decide
status: Done
assignee:
  - '@claude'
created_date: '2026-10-26 14:32'
updated_date: '2026-10-27 21:48'
labels:
  - console
  - performance
dependencies:
  - TASK-36012
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-36012 lets the Conversations tray repaint a run's start and end in place: on a delta confined to run markers, attention facts, icons and colours it composes the tray twice off the DOM (the state shown and the next one), diffs the two, and patches the mounted rows when no box moves. Those two composes cost about what the rebuild they replace did (with the app's stylesheets, a running-marker sync measured 81 ms of CPU against dev's 81 ms at 12 rows and 211 ms against 205 ms at 40 rows), so the patch saves the whole-screen reflows a rebuild triggers but no CPU inside the tray, and all of the cost now blocks inside sync_state. The TASK-36012 review measured most of a compose's cost in the stylesheet applies that add_class and set_class run on widgets that are not yet mounted. Recording each patchable node's compose-time values when the tray composes would let the patch skip the compose of the state already shown, roughly halving the work on every send's start and end.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A run marker that the tray patches in place runs at most one compose of the tray
- [x] #2 The patch still lands the same geometry, paint, classes and tooltips as a rebuild (the TASK-36012 A/B stays green)
- [x] #3 CPU for a patched running-marker sync, measured with process_time at 12 and 40 rows, falls below the dev baseline
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Done in TASK-33620.15.20's branch (2026-10-27): the patch now runs no compose at all. It decides from the row data, as this description proposed, and goes one step further, since the record also replaces the compose of the next state.

- Compose records, per browser row, the values a run can change (`_BrowserRowPaint`: the row tooltip, the action glyph with its colour, the attention class and the action tooltip, all from one `_conversation_row_paint` that compose and patch both call) and the marker each section and group header shows. The patch compares the record with what the incoming rows give, checks that the mounted buttons still show the record, and writes only what differs. A header marker change, a glyph-mode flip, or a row that drifted from its record rebuilds.
- `plan_in_place_delta` (the compose diff) and the interim `detached_styles_skipped` (which swapped `App.update_styles` while the diff composes ran) are removed.
- AC#1: `Tests/UI/test_console_tray_patch_styles.py::test_running_marker_patch_builds_and_styles_no_throwaway_node` pins 0 composes, 0 widgets built and 0 off-DOM restyles for a patched running marker (red on origin/dev 0923ec0ca2 and on the interim head: 2 composes).
- AC#2: `test_running_marker_patch_matches_a_recompose` stays green. `test_browser_paint_deltas_match_a_rebuild` adds an A/B per patchable field (row attention, icon, colour, run marker; section and group markers) that also compares label spans and painted segment styles. It found that TASK-36012's patch dropped a colour-only change: Textual's reactive compares `Content` by text, so `button.label = ...` was a no-op. The label is now written with `set_reactive` + `mutate_reactive`.
- AC#3, headless (app stylesheets, `process_time` around `sync_state`, median of 9 alternating idle/running syncs, 4 interleaved runs per arm, host load 27-42): at 12 rows, dev 87-93 ms, the interim head 11-13 ms, this change 1.3-2.7 ms; at 40 rows, 243-266, 24-26 and 2.0-3.6 ms. Widgets built per sync: 96 / 96 / 0 at 12 rows, 264 / 264 / 0 at 40.
<!-- SECTION:NOTES:END -->
