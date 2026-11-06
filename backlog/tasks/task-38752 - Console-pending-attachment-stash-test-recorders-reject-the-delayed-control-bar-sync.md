---
id: TASK-38752
title: "Console pending-attachment stash test recorders reject the delayed control-bar sync"
status: Done
assignee:
  - '@claude'
created_date: 2026-11-05 07:11
updated_date: 2026-11-06 00:00
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

**Re-review r8c-rev2: the Stop precondition, a second racy Stop read, and the wait deadline.**
- **The precondition.** After the adopted screen's `await fresh._sync_native_console_chat_ui()`, the test asserted `stop.styles.display != "none"` once. It failed 1 in 10 runs on the branch and once in 10 at base (`assert 'none' != 'none'`). A temporary probe over 60 cases (load 9-15) found that the awaited sync returned False in 39 of them without reaching `prepare_console_presentation_sync`. In 32 of those the whole-sync replay was pending after the call; the other 7 had coalesced into a sync already in flight. So the awaited sync usually paints nothing, and Stop shows only once some earlier or later sync paints the control bar. Holding the fresh screen's `_sync_console_control_bar_under_config` for 0.3 s after the awaited sync reproduced the failure in all 3 cases on the old code, twice (load ~10). The node now waits, with `_wait_for_h3_state`, until Stop is displayed and enabled; it passed both runs under the same hold. The run-status copy check stays a one-shot assert: it was right in 60 of 60 cases.
- **`[cancel]`'s last Stop check.** It read Stop after the newer edit had ended (`remove_active` runs when its task finishes). By then any repaint may hide Stop legitimately. One forced control-bar repaint there failed it (`assert 'none' != 'none'`), and a delayed-read control had hit the same assertion once. The check is removed; the attributed counters pin that the stale settlement requested no sync of its own. The Stop check made while the newer edit is still active stays.
- **`_wait_for_h3_state`'s deadline is now 10 s, up from 2 s.** Timed per wait across 15 runs at load 16-24: late `[success]` settlements took up to 1.163 s (median 0.914 s) and `[failure]` up to 1.143 s (median 0.831 s). At load 8-15 the maxima were 0.99 and 0.76 s, so the time grows with load. The helper returns as soon as its predicate holds, so only a failing wait takes longer. This may explain the one-off `[cancel]` timeout above, which was never caught with its events.

**AC#2.**
- After the Stop wait and deadline change (aa446e54fd plus this commit): 20 of 20 `-n 4` runs, two parallel batches of 10, 3 of 3 cases each, load 15.1-24.0 (1-minute average).
- The previous version passed 30 of 30 (load 10.4-47.9).
- No TypeError and no failure.
- The reorder control, re-run on the final node, was still red in all 3 cases.

**Whole file** (`-n 4`): 8 other plain nodes fail with `RecoveryRequired: raw_source_selection_changed` on the branch and at base with the same command. That is TASK-38111's trap. At base the same run also failed this node's `[success]` and `[failure]` cases.
<!-- SECTION:NOTES:END -->
