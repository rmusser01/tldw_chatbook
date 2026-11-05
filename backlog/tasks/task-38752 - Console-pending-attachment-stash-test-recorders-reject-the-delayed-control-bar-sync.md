---
id: TASK-38752
title: "Console pending-attachment stash test recorders reject the delayed control-bar sync"
status: To Do
assignee: []
created_date: 2026-11-05 07:11
labels:
- follow-up
- tests
- console
- flaky
priority: medium
references:
- Tests/UI/test_console_pending_attachment_stash.py
- tldw_chatbook/UI/Screens/chat_screen.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 7 of the 2026-10-23 follow-up burn-down. Checked on dev bb80905eb3, 2026-11-04.

`Tests/UI/test_console_pending_attachment_stash.py::test_fresh_mounted_screen_settles_late_h3_outcome_in_dom_and_controls` (:663) replaces `ChatScreen._request_console_control_bar_sync` with two local recorders: `_record_control_sync` (:766) and `_count_control_sync` (:866). Neither takes any argument.

Production calls that method with `delayed=True` whenever a sync defers. Examples: `UI/Console_Modules/config_sync.py:156` and :163, `inspector_data.py:711`, `context_spend.py:916`, `UI/Screens/chat_screen.py:17860`. So any deferral during the test raises `TypeError: ..._record_control_sync() got an unexpected keyword argument 'delayed'` in whichever worker made the call, and that worker dies. Every other test double of this method under `Tests/UI` accepts `**_`.

TASK-38696's notes (PR #3404) record the node failing in 2-3 of 3 runs at base, always with this TypeError. On this base, one `-n 4` run failed both parametrized cases (`[success]` and `[failure]`) with it. #3404 widens the method's signature to `delayed: bool | float`, so a fix should forward whatever it is given.

The file runs on no PR lane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both recorders accept the method's real arguments and pass them on to the original, so a deferred sync still reaches it
- [ ] #2 The node passes 10 of 10 `-n 4` runs, with the load average recorded in the notes; if a run fails, it fails on the test's own assertion, never with a TypeError
<!-- AC:END -->
