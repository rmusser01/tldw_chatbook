---
id: TASK-23148
title: Workbench geometry tests still assume two-edge rail handles
status: To Do
assignee: []
created_date: '2026-08-28'
labels:
  - tests
  - console
priority: medium
dependencies: []
---

## Description

11 assertions in `Tests/UI/test_workbench_visual_snapshots.py` still compute a rail handle's
content width as `region - 2`. Rail handles are now framed on **one** vertical edge, so the correct
arithmetic is `region - 1`. The file has been red since 2026-08-23.

Someone already hit this and patched **exactly one** of the stale numbers, leaving the rest — which
is the argument for pinning the contract rather than the arithmetic.

## Acceptance Criteria

- [ ] The stale expected values are corrected (`9`->`10` at line 287, `content_width=11`->`12` at
  line 485, and the one `expected_rail_widths` entry `34`->`35`)
- [ ] A test asserts each handle carries **exactly one** vertical border, so the next edge-count
  change fails on the contract instead of on six unexplained integers

## Evidence

`tldw_chatbook/UI/Screens/chat_screen.py:10368` `_frame_console_region(left_handle,
edges=("right",))` and `:10617` `(right_handle, edges=("left",))`; rationale in
`tldw_chatbook/UI/Console_Modules/frame.py:8-10` (TASK-20937.3, "one edge-owned surface").

Introduced by `a581f28e0a` (2026-08-23) "Redesign Console edge rails and workspace tree (#2034)",
whose diff is literally `-...(left_handle, bottom=False)` -> `+...(left_handle, edges=("right",))`
and which did not touch this test file. `04e29673a2` later fixed one number and left the others.
