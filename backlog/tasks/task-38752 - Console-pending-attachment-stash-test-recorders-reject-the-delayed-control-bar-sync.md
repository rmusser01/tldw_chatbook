---
id: TASK-38752
title: "Console pending-attachment stash test recorders reject the delayed control-bar sync"
status: Done
assignee:
  - '@claude'
created_date: 2026-11-05 07:11
updated_date: 2026-11-05 23:18
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
**Change (AC#1).** In `Tests/UI/test_console_pending_attachment_stash.py::test_fresh_mounted_screen_settles_late_h3_outcome_in_dom_and_controls`, `_record_control_sync` and `_count_control_sync` take `*args, **kwargs` and forward them, so a deferred `delayed=` request still reaches the real method.

**Attribution (review r8c-rev1).** The first version recorded every argument-free control request and checked the order only as a subsequence. Neither pinned anything:
- Console ticks append `transcript` and `reconcile`.
- The settlement's own nested native sync adds an inner `reconcile`.
- Other code makes argument-free control requests: the readiness `_retry`, `cut_replay_backoff`, and three `chat_screen.py` sites.

The reviewer's in-memory controls left `[success]` and `[failure]` green.

Each recorder (and both `[cancel]` counters) now counts a call only when `_settle_current_h3_outcome` is the nearest owner on its caller stack, ahead of `ChatScreen._sync_native_console_chat_ui`. Ticks and the nested sync are both ignored. The wait predicate and the final assertion require exactly `["reconcile", "transcript", "controls"]`.

**Two other load flakes in this node, fixed after the attribution exposed them:**
- **The settled row's text.** The predicate now also waits for the row's rendered text. It treats `NoMatches` as not settled, because the row mounts before its Markdown body. Without this the node failed 1 of 10 at `assert '[image] settle terminal' in 'Assistant ... image/png'`, and once with `NoMatches: 'Markdown'`.
- **The pop timeout.** The setup's `wait_for(host.pop_screen(), timeout=0.5)` timed out at load ~30-36 in 2 of 30 runs (`TimeoutError` at the old :736), once in all 3 cases. It is now 5 s. That still fails long before the held edit's 10 s `release.wait`.

**Red evidence.**
- With a temporary `fresh._request_console_control_bar_sync(delayed=True)` after the recorders are installed, base failed all 3 cases with `TypeError: ..._record_control_sync() got an unexpected keyword argument 'delayed'`. The fixed test passed all 3.
- The reviewer's three controls, each a temporary edit of `_settle_current_h3_outcome` in `UI/Console_Modules/image.py`, were each red in all 3 cases (`Timed out waiting for late H3 <kind> transcript/control settlement`), then reverted. All three were run before and after the two flake fixes (load 14-31):
  - reconcile moved after the transcript sync;
  - the settlement's reconcile dropped;
  - its control-bar request dropped.
- A temporary edit that lets a stale generation settle failed `[cancel]` at the attributed counter (`assert 1 == 0`).

**AC#2.** The final node passed 30 of 30 `-n 4` runs, 3 of 3 cases each: 10 at load 10.4-17.3, then two batches of 10 in parallel at load 20.9-47.9 (1-minute average). No TypeError and no failure. One `[cancel]` settlement timeout in an intermediate run (load 20.5) did not recur in the 82 later runs; the other failures in those runs were the pop timeout and the row text above. Its events were not captured.

**Whole file** (`-n 4`): 8 other plain nodes fail with `RecoveryRequired: raw_source_selection_changed` on the branch and at base with the same command. That is TASK-38111's trap. At base the same run also failed this node's `[success]` and `[failure]` cases.
<!-- SECTION:NOTES:END -->
