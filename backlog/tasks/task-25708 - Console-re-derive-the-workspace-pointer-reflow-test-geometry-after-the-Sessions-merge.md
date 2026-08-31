---
id: TASK-25708
title: >-
  Console: re-derive the workspace-pointer reflow test geometry after the
  Sessions merge
status: Done
assignee: []
created_date: '2026-08-30 19:40'
updated_date: '2026-08-31 04:55'
labels:
  - console
  - tests
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
test_production_workspace_pointer_keeps_pressed_key_across_outer_reflow scrolls to workspace_header_y - 3 to put the Workspaces header above the fold. TASK-23199 retired the Sessions section, so Workspaces now leads the rail and that expression clamps to 0 - the outer never scrolls and the reflow the test asserts cannot occur. The behaviour under test is unchanged; only the setup's coordinates need re-deriving. Marked xfail meanwhile.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test creates a real outer reflow without depending on a section above Workspaces
- [ ] #2 The xfail marker is removed
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Done. Fixed by probing the live geometry rather than guessing coordinates.

The probe showed the constraint the old fixed offsets had been satisfying by accident: revealing Workspaces scrolls the outer back to the top, moving content DOWN by exactly the current offset, while the pointer deliberately stays still through the reflow. The pressed row must therefore be on the tree BOTH before and after the shift - at least reveal_shift rows into the tree (or the stationary pointer ends up above it) and reveal_shift rows clear of the clip bottom (or it leaves the other side). Encoding both bounds is what made it deterministic.

Two supporting findings from the same probe: the offset must be small (two rows, not five) because each workspace row is followed by four conversation rows, so a large shift can leave NO workspace row satisfying both bounds; and the tree's own pre-scroll had to go - it existed to bring the hardcoded workspace-1 into view and was shifting the line-to-row mapping out from under the search.

The test now reads the layout instead of assuming one: it searches the visible band for a workspace row meeting the bounds and derives the expected activation id from that node. It should survive future changes to the rail's section set, which is what broke it in the first place.

Passes three consecutive runs; test_console_rail_reconciliation.py has no xfails left (54 passed). preflight green.
<!-- SECTION:NOTES:END -->
