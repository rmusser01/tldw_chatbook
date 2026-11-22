---
id: TASK-42249
title: "Four hooks_v2 test nodes hang their teardown in legacy_hook_ownership.wait_closed, and one cannot exit when run alone"
status: To Do
assignee: []
created_date: 2026-11-22 00:17
updated_date: 2026-11-22 00:17
labels:
- follow-up
- tests
- console
- hooks
priority: medium
references:
- tldw_chatbook/Agents/legacy_hook_ownership.py
- tldw_chatbook/Agents/hooks_v2/engine.py
- Tests/Chat/test_hooks_v2_teardown.py
- Tests/Chat/test_hooks_v2_continuations.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the consoleb stream (TASK-42155 and TASK-42156, round 19). Four nodes leave an executor thread blocked in `legacy_hook_ownership.wait_closed` (`tldw_chatbook/Agents/legacy_hook_ownership.py:37`, called through `asyncio.to_thread` at `tldw_chatbook/Agents/hooks_v2/engine.py:1263`) with no timeout, so teardown hangs:
- `Tests/Chat/test_hooks_v2_teardown.py::test_mounted_stop_button_cancels_pending_stop_proposal[size1-False]` and `[size1-True]`
- `Tests/Chat/test_hooks_v2_continuations.py::test_consumption_before_foreground_or_uncertain_failure_is_not_reopened[stop]`
- `Tests/Chat/test_hooks_v2_continuations.py::test_lost_scheduler_response_never_returns_machine_work_to_queue`

One of them cannot exit when run alone. This is the same mechanism as TASK-41028, which is still open (To Do) at e5626ea043; the two are filed separately and neither depends on the other.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause of the blocked `wait_closed` is named for each of the four nodes
- [ ] #2 Each node exits on its own and in a `-n 4` run of its file, within a stated bound, and no executor thread is left in `wait_closed` after the run
- [ ] #3 A red-before-fix assertion or lane pins the hang
<!-- AC:END -->
