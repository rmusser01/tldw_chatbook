---
id: TASK-25708
title: >-
  Console: re-derive the workspace-pointer reflow test geometry after the
  Sessions merge
status: To Do
assignee: []
created_date: '2026-08-30 19:40'
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

## Renumbering provenance

Created as TASK-25706 at 2026-08-30 19:40. `dev` already carried a
TASK-25706 ("Make submitted-log regression coverage truthful on Windows",
created 17:52), which is the older arrival and keeps the id per the
2026-08-21 owner rule (TASK-19601). This task renumbered to TASK-25708 on
rebase; the xfail marker in
`Tests/UI/test_console_rail_reconciliation.py` was updated to match.
