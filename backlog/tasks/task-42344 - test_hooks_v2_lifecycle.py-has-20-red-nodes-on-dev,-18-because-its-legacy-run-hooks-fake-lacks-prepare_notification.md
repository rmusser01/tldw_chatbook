---
id: TASK-42344
title: "test_hooks_v2_lifecycle.py has 20 red nodes on dev, 18 because its legacy run-hooks fake lacks prepare_notification"
status: To Do
assignee: []
created_date: 2026-11-22 16:37
updated_date: 2026-11-22 16:37
labels:
- follow-up
- tests
- console
- hooks
priority: medium
references:
- Tests/Chat/test_hooks_v2_lifecycle.py
- tldw_chatbook/Chat/console_chat_controller.py
- tldw_chatbook/Agents/run_hooks.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the TASK-34975 port (round 20, fu/r20-ctxclose), which reported it as `Tests/UI/test_hooks_v2_lifecycle.py`; the file is `Tests/Chat/test_hooks_v2_lifecycle.py`. Re-run at 59cc99e61b (dev 8a55555798 plus the round-20 stack, which does not touch this file or the code below) through the scratch wrapper at `-n 4`: 11 failed and 9 errors. The file is in no PR lane (`scripts/ui_pr_gate_census.txt` and `.github/workflows/` do not name it), so the Hooks v2 lifecycle checks it holds run nowhere that blocks a merge.

- **18 nodes (9 failures, 9 setup errors):** `AttributeError: 'LegacyRecorder' object has no attribute 'prepare_notification'`, raised at `console_chat_controller.py:10205`. 11d71e5db5 ("Integrate ordinary Console physical ownership", 2026-11-02) made the ordinary-tree path call `engine.prepare_notification("Stop", session.id)` on the legacy run-hooks engine (`Agents/run_hooks.py:1344`). The file's `LegacyRecorder` fake (`test_hooks_v2_lifecycle.py:27-33`) has only `fire_async` and `notify`.
- **2 more nodes:** `test_scheduled_wake_initializes_without_user_prompt_submit` (`assert False` at `:568`: the woken turn never reached the fake gateway) and `test_actual_console_stop_mcp_notification_keeps_one_second_guard_deadline[success]` (`assert '1' == '2'` at `:1403`). These may share the cause or be load-sensitive; they were not traced.

`Agents/run_hooks.py` is in another session's open work (tldw-chatbook-30). If the fix needs that file rather than the test fake, coordinate first.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The 18 `prepare_notification` nodes pass, with the test fake matching the engine interface the controller now calls
- [ ] #2 The other two red nodes are traced: each is fixed, or the implementation notes show it is a separate cause and name a task for it
- [ ] #3 The whole file passes through the scratch wrapper at `-n 4` on a branch carrying the fix, with the load average recorded
- [ ] #4 The file runs on a PR lane
<!-- AC:END -->
