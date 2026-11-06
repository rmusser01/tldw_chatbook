---
id: TASK-39058
title: "Two Watchlists Artifacts tests assert idle before the reload that a retry press or a scope change queued has finished, so they read 'refreshing'"
status: Done
assignee:
- '@claude'
created_date: 2026-11-06 10:09
updated_date: 2026-11-06 17:59
labels:
- follow-up
- tests
- watchlists
- flaky
priority: medium
references:
- Tests/Watchlists/test_watchlists_artifacts_refresh_states.py
- tldw_chatbook/UI/Watchlists_Modules/artifacts_pane.py
dependencies:
- TASK-38990
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 10 of the 2026-10-23 burn-down; checked on dev b2c618185f, 2026-11-06, and on the Architecture branch that lands TASK-38990, now PR #3436 (head b7c8e809b3); the runs used an earlier commit of that branch. TASK-38990's notes name both as pre-existing flakes. Line numbers are dev's, with #3436's in brackets.

**After a retry press.** In `test_accepted_missing_receipt_retries_its_exact_id` (`:480` [#3436 :491]), the test does `press()`, then `await host.workers.wait_for_complete()`, then `await pilot.pause()`, then `assert pane.view_state == "idle"` (`:550-554` [#3436 :562-566]).
1. `press()` posts a message.
2. `ArtifactsPane.on_button_pressed` then posts `RefreshBriefingsRequested` (`UI/Watchlists_Modules/artifacts_pane.py:2283-2284`).
3. Only then does the screen schedule the `wl-briefings-load` worker.

`wait_for_complete()` can return before that worker exists, and the test reads `assert 'refreshing' == 'idle'`.

**After a scope change.** In `test_generation_completion_does_not_publish_into_a_new_watchlist_scope[False]` (`:616` [#3436 :682]), the test asserts `screen._artifacts_view_state == "idle"` after `await task; await pilot.pause()` (`:703` [#3436 :769]). The reload that `_request_tree_scope` queued is still running, again `'refreshing' == 'idle'`.

**Runs.** All 6 nodes of the two tests, 10 runs at `-n 4 -p no:randomly`. The runs used a scratch plugin that adds the `bootstrap_profile` marker; without it every node fails locally with RecoveryRequired (TASK-38111).

| Tree | Runs red | Failing node runs | Load |
|---|---|---|---|
| Dev | 10 of 10 | 8 at `:554`; 5 at `:541` (`'idle' == 'storage_mismatch'`); 2 `WorkerCancelled` | 21-26.5 |
| After TASK-38990 | 6 of 10 | 9 at [#3436 :566]; 1 at [#3436 :769] | 10.5-26.4 |

- TASK-38990's `_settle_briefings_load` fixes the `:541` and `WorkerCancelled` failures; the after-press race remains.
- Run once alone after TASK-38990, at load 11.8: 1 failed (`[True-size1]` at [#3436 :566]), 5 passed.
- No PR lane runs the file.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both assertions wait for the reload they depend on (the worker the press or the scope change queued), not for a fixed pause; no assertion is weakened
- [x] #2 10 runs at `-n 4` of both tests give 0 red, with `uptime` recorded; before the fix, 6 of 10 runs are red after TASK-38990
<!-- AC:END -->

## Implementation Plan

1. Reproduce: 10 runs at `-n 4` of both tests, under a temporary `bootstrap_profile` mark (the filer's method; without it every node fails locally with RecoveryRequired, TASK-38111).
2. Trace what each assertion waits on. The press reaches the loader through two posted messages, and a load that moves the selection posts `BriefingSelected`, whose handler queues one more load. So both tests must wait for every load their action queued, follow-ups included.
3. Add a drain helper: record the pane's load-requesting messages, then wait until each was handled (its handler calls `stop()` first) and no `wl-briefings-load` worker is left. Use it after the press and after the scope change. No assertion changes.
4. Prove it: a temporary 0.5 s delay at the start of the queued load makes the old waits red and the new ones green; then 10 runs at `-n 4` with `uptime`.

## Implementation Notes

**Status: Done.**

**What the assertions really wait on.**
- The retry press reaches the loader through two posted messages: the button's `Pressed`, then the pane's `RefreshBriefingsRequested`.
- Every load that moves the selection assigns `pane.selected_briefing`. The pane's watcher then posts `BriefingSelected`, and the screen's handler queues one more full load. The scope-change test meets the same chain through its direct `_load_briefings(select_briefing_id=...)` call.
- So the filed "wait for the worker the press queued" is not enough. A first fix that did exactly that (a spy on `_request_briefings_refresh`, then the loader group's `wait()`) still failed the scope test in 2 of 10 runs under the delay control below.

**Fix (test only).**
- `_track_load_requests(pane)` records each `RefreshBriefingsRequested`, `BriefingSelected` and `ScriptSelected` the pane posts.
- `_drain_briefings_loads(host, pilot, posted, after=...)` loops until every recorded message has been handled (each screen handler calls `event.stop()` first) and no `wl-briefings-load` worker is unfinished. After a press it also requires the `RefreshBriefingsRequested` to have been posted. Its 30 s deadline only catches a hang; it is not a settle.
- The press test and the scope test use it in place of `wait_for_complete()` + `pause()` and `pause()`. No assertion changed.

**Evidence.** These tests are plain, and locally every node fails with RecoveryRequired (TASK-38111). So all runs used a temporary module-level `pytest.mark.bootstrap_profile`, the filer's method, so that the before numbers compare with the filed ones; it was removed after each run. I used it rather than `@private_profile_test`, which runs each case in a child process and so would change the timing being measured.
- AC#2, before the fix: 10 runs at `-n 4`, 1 of 10 red (`[False-size0]` at `:566`, `'refreshing' == 'idle'`), load 38-52. The filer saw 6 of 10.
- AC#2, after the fix: 10 runs at `-n 4`, 0 of 10 red, load 25-34.
- Deterministic control: a temporary `await asyncio.sleep(0.5)` at the start of the queued load in `_request_briefings_refresh`, reverted afterwards.
  - Old waits: red in every run. All four press nodes fail at `:566` with `'storage_mismatch' == 'idle'`, and the scope nodes fail in 3 of 8 runs with `'refreshing' == 'idle'`.
  - New waits: green in 8 of 8 runs.
- Four other tests in the same file have the same race and are red at HEAD under load too (Proposed follow-ups in the report): `test_generation_failure_preserves_last_good_content_and_has_own_state`, `test_refresh_failure_preserves_screen_owned_last_good_content`, `test_durable_receipt_mismatch_retains_last_good_content` and `test_citation_resolution_failure_retains_last_good_citations`.

Lesson added to `backlog/docs/lessons-testing-evidence.md` (the follow-up load chain).

File: `Tests/Watchlists/test_watchlists_artifacts_refresh_states.py`.
