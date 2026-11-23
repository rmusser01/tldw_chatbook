---
id: TASK-42476
title: "Measure how many Inspect rail owner passes one real launch staging costs"
status: To Do
assignee: []
created_date: 2026-11-23 09:42
updated_date: 2026-11-23 09:42
labels:
- follow-up
- console
- performance
priority: low
references:
- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/UI/Console_Modules/right_rail.py
- tldw_chatbook/UI/Console_Modules/config_sync.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 22 of the follow-up burn-down (2026-11-23), by TASK-42474's review. Checked on `fu/r22-railrecon` (the round-22 stack a999508ae7 plus TASK-42474); paths are under `tldw_chatbook/`.

**Why.** Staging a launch refreshes every Inspector reader of the pending launch through `_sync_console_pending_launch_surfaces` (`UI/Screens/chat_screen.py:14363-14411`). That syncs the staged-context tray and the settings summary at once, in the same call, so their reconcile requests can share one outer owner pass. The run inspector is synced later, through the deferred control-bar sync (`_request_console_control_bar_sync()`, a `call_after_refresh` into `_run_coalesced_control_bar_sync`), and the live-work card swap is queued with `call_later` and requests its own outer reconcile when it finishes. So one staging may cost two or more outer owner passes (`ConsoleInspectorRail._outer_owner_reconcile_count`), each a measure and refold of the rail.

TASK-42474's sibling test showed the shape: after it staged a launch and synced only the tray, a Console sync that landed later synced the settings summary and run inspector for that launch and added a second owner pass (`assert 3 == (1 + 1)`). Nobody has measured what a real staging costs, or whether the extra passes are worth folding.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The notes record how many outer owner passes, and how much UI-loop time, one real launch staging costs on a mounted Console with the Inspect rail open and closed, measured with the startup syncs settled
- [ ] #2 If a staging costs more than one owner pass, the notes say which surfaces cause the extra passes and either the product folds them into one (shown by a test that counts passes per staging) or the notes record why it should not
<!-- AC:END -->
