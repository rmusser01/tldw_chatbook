---
id: TASK-38752
title: "Console pending-attachment stash test recorders reject the delayed control-bar sync"
status: Done
assignee:
  - '@claude'
created_date: 2026-11-05 07:11
updated_date: 2026-11-05 22:03
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
- [x] #1 Both recorders accept the method's real arguments and pass them on to the original, so a deferred sync still reaches it
- [x] #2 The node passes 10 of 10 `-n 4` runs, with the load average recorded in the notes; if a run fails, it fails on the test's own assertion, never with a TypeError
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Make both recorders take `*args, **kwargs` and forward them to the original.
2. Record only the argument-free requests (the settlement's own call), so a deferred retry from elsewhere cannot reorder or inflate the events the test asserts on.
3. Show the TypeError red at base with a forced delayed request, then 10 `-n 4` runs with the load recorded.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Change.** In `Tests/UI/test_console_pending_attachment_stash.py::test_fresh_mounted_screen_settles_late_h3_outcome_in_dom_and_controls`, `_record_control_sync` and `_count_control_sync` now take `*args, **kwargs` and forward them to the original, so a deferred `delayed=` request reaches the real method (AC#1). They count only argument-free requests, the settlement's own call shape (`image.py` `_settle_current_h3_outcome`). A forwarded deferral retry is not the settlement's request, so it cannot inflate `control_syncs` or reorder the events.

**Second fix, needed for AC#2.** With only the forwarding, the node failed 2 of 3 `-n 4` runs (load 43-49) on its own ordering assertion. The first event list was `['transcript', 'transcript', 'reconcile', 'reconcile', 'transcript', 'controls']`: the Console's own ticks call `_sync_native_console_chat_ui`, which also reconciles (`chat_screen.py:17767`), before and during the settlement. At base this was hidden, because the TypeError failed the node first. The test now requires the settlement's order as a subsequence (`reconcile`, then `transcript`, then `controls`), both in its wait predicate and in the final assertion, instead of comparing the first occurrence of each event.

**Red/green.** With a temporary `fresh._request_console_control_bar_sync(delayed=True)` right after the recorders are installed, base failed all 3 cases with `TypeError: ..._record_control_sync() got an unexpected keyword argument 'delayed'`. The fixed test passed all 3 cases (load 24-26).

**AC#2 runs.** The node passed 10 of 10 `-n 4` runs, 3 of 3 cases each, at load 27.8-40.7 (1-minute average), with no TypeError and no failure.

**Whole file** (`-n 4`): 8 other plain nodes fail with `RecoveryRequired: raw_source_selection_changed` on the branch and at base with the same command. That is TASK-38111's trap. At base the same run also failed this node's `[success]` and `[failure]` cases.
<!-- SECTION:NOTES:END -->
