---
id: TASK-42347
title: "Roleplay compact split overflows the screen at 61-73 columns"
status: To Do
assignee: []
created_date: 2026-11-22 16:37
updated_date: 2026-11-22 16:56
labels:
- follow-up
- roleplay
- bug
priority: low
references:
- tldw_chatbook/UI/Screens/personas_screen.py
- tldw_chatbook/css/components/_agentic_terminal.tcss
- Tests/UI/test_personas_workbench.py
- Docs/User_Guide/roleplay-chat-dictionaries.md
dependencies:
- TASK-33910.49
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the TASK-33910.49 port (PR #3431, round 20, fu/r20-roleplay) and confirmed by its review. Measured on that branch with `run_test(size=...)`; the code path at these widths is dev 8a55555798's.

**Why it matters.** Between 61 and 73 columns the Roleplay workbench does not fit the screen: the Inspector is off-screen and the Library pane is wider than the space it has.

**Evidence.** At 61-97 columns the workbench uses the compact split (`PERSONAS_COMPACT_WORKBENCH_MAX_WIDTH`, `personas_screen.py:623`; the one-pane narrow layout starts at 60, `:2054`). The compact minimums are 16 + 34 + 22 columns (`$ds-size-16/-34/-22`, `css/components/_agentic_terminal.tcss:746-762`) plus 2 border cells, which is 74. At 61-73 the Library pane measures 59-71 columns wide and the Inspector is off-screen. With the Library rail collapsed the overflow is at 61-70; with the Inspector collapsed, at 61-62.

The fix needs another layout tier or smaller compact minimums. TASK-33910.49 pinned the 98-column cut-off to the CSS minimums with a test; the new cut-off needs the same.

**Overlap with TASK-33910.24** (To Do, "True single stage below 64 columns for Roleplay"). That task would show one pane at a time below 64 columns, which covers 61-63 of this range. This task owns 64-73 either way. If TASK-33910.24 lands first, this task starts at 64 and keeps its cut-off; if this task lands first, its layout for 61-63 is what TASK-33910.24 later replaces, and the two tests should not pin conflicting cut-offs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At every width from 61 to 73 columns, with both rails open, the Library rail collapsed, and the Inspector collapsed, every visible pane sits inside the screen and its controls stay whole
- [ ] #2 Layouts at 60 columns and below, and from 74 columns up, are unchanged
- [ ] #3 A test pins any new cut-off to the panes' CSS minimum widths, so a change to either fails until the other follows
- [ ] #4 The Roleplay User Guide page states which widths show which panes
<!-- AC:END -->
