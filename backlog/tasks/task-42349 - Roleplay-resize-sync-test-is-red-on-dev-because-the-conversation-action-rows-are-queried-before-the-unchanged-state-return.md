---
id: TASK-42349
title: "Roleplay resize-sync test is red on dev because the conversation action rows are queried before the unchanged-state return"
status: To Do
assignee: []
created_date: 2026-11-22 16:37
updated_date: 2026-11-22 16:37
labels:
- follow-up
- tests
- roleplay
- performance
priority: medium
references:
- Tests/UI/test_personas_workbench.py
- tldw_chatbook/UI/Screens/personas_screen.py
- scripts/ui_pr_gate_census.txt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the TASK-33910.49 port (round 20, fu/r20-roleplay), which reported it red on dev 8a55555798. Re-run at 59cc99e61b through the scratch wrapper: red. TASK-41800 had already noted it red on dev 5266c33936 and on no lane, and filed nothing for it.

**Why it matters.** `TestWorkbenchShell::test_resize_sync_skips_work_when_compact_state_is_unchanged` guards the rule that a resize which does not change the compact or narrow state does no pane work. It fails with "unchanged compact state should not query panes", so every resize currently does that work and no test notices. The file is censused only by node, and this node is on no PR lane.

**Evidence.** `_sync_responsive_workbench` (`personas_screen.py:2052-2091`) queries `#personas-conversation-actions`, `#personas-conversation-navigation-actions` and the conversation rows, and restyles them, before its `if self._workbench_compact == compact and self._workbench_narrow == narrow: return` early return. git blame puts that block in 12836ce11c ("add trusted character conversation navigation", 2026-09-05). The test fails any `#personas-*` query on an unchanged state.

Either the conversation rows' sizing should also be skipped when nothing changed, or it must run on every resize and the test should allow those selectors. The fix should say which and why.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The implementation notes say whether the conversation rows' sizing must run on every resize, with the reason
- [ ] #2 A resize that changes neither the compact nor the narrow state does no work beyond what #1 requires, and the test asserts exactly that
- [ ] #3 The test passes through the scratch wrapper on a branch carrying the fix, and the conversation rows still size correctly at 52, 60, 61 and 98 columns
- [ ] #4 The node runs on a PR lane
<!-- AC:END -->
