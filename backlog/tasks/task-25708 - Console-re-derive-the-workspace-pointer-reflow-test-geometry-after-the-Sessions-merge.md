---
id: TASK-25708
title: >-
  Console: re-derive the workspace-pointer reflow test geometry after the
  Sessions merge
status: To Do
assignee: []
created_date: '2026-08-30 19:40'
updated_date: '2026-08-30 22:37'
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
Partially re-derived on the TASK-23199 branch; NOT finished.

Done: the press target is now chosen from the band that is actually visible (intersection of the tree's content region with the outer's clip), scanning outward from the midpoint for a WORKSPACE row, with the expected activation id derived from the node rather than hardcoded to 'workspace-1'. The outer is scrolled away from the top so the reveal has somewhere to move from. With that, the first half passes against the new layout: _pressed_node_key, the active section flipping to 'workspace', outer.scroll_y changing, and tree.content_region.y changing.

Remaining: the trailing double-click phase. After the reveal the pressed row lands on row 24 while the outer's content_region.bottom is exactly 24 -- one cell outside the clip -- and centring the row in the tree's own viewport does not move it off that boundary, which suggests the tree extends past the outer clip in a way the tree-relative offset does not account for. pilot.click then refuses the offset.

Left xfail rather than tuned to green: I had it passing at one point by hand-picking a scroll offset, then the click coordinate stopped landing, which is the signature of calibrating numbers until the bar turns green. The behaviour under test is unchanged; what is needed is understanding the tree/outer clip interaction, not another guess.

Sibling coverage note for whoever picks this up: Tests/UI/test_console_workspace_tree.py covers _pressed_node_key at the WIDGET level (54 tests). What this test uniquely covers is the press surviving a RAIL REFLOW, so the gap while xfailed is real.
<!-- SECTION:NOTES:END -->
