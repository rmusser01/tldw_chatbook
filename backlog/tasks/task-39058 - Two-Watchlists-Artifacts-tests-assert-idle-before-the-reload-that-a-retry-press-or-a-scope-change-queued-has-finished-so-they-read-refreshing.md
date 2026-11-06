---
id: TASK-39058
title: "Two Watchlists Artifacts tests assert idle before the reload that a retry press or a scope change queued has finished, so they read 'refreshing'"
status: To Do
created_date: 2026-11-06 10:09
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
Found in round 10 of the 2026-10-23 burn-down; checked on dev b2c618185f, 2026-11-06, and on the Architecture branch that lands TASK-38990 (the state after it merges). TASK-38990's notes name both as pre-existing flakes. Line numbers are from dev, with the branch's in brackets.

**After a retry press.** In `test_accepted_missing_receipt_retries_its_exact_id` (`:480`), the test does `press()`, then `await host.workers.wait_for_complete()`, then `await pilot.pause()`, then `assert pane.view_state == "idle"` (`:550-554` [`:562-566`]).
1. `press()` posts a message.
2. `ArtifactsPane.on_button_pressed` then posts `RefreshBriefingsRequested` (`UI/Watchlists_Modules/artifacts_pane.py:2283-2284`).
3. Only then does the screen schedule the `wl-briefings-load` worker.

`wait_for_complete()` can return before that worker exists, and the test reads `assert 'refreshing' == 'idle'`.

**After a scope change.** In `test_generation_completion_does_not_publish_into_a_new_watchlist_scope[False]` (`:616`), the test asserts `screen._artifacts_view_state == "idle"` after `await task; await pilot.pause()` (`:703` [`:727`]). The reload that `_request_tree_scope` queued is still running, again `'refreshing' == 'idle'`.

**Runs.** All 6 nodes of the two tests, 10 runs at `-n 4 -p no:randomly`. The runs used a scratch plugin that adds the `bootstrap_profile` marker; without it every node fails locally with RecoveryRequired (TASK-38111).

| Tree | Runs red | Failing node runs | Load |
|---|---|---|---|
| Dev | 10 of 10 | 8 at `:554`; 5 at `:541` (`'idle' == 'storage_mismatch'`); 2 `WorkerCancelled` | 21-26.5 |
| After TASK-38990 | 6 of 10 | 9 at `:566`; 1 at `:727` | 10.5-26.4 |

- TASK-38990's `_settle_briefings_load` fixes the `:541` and `WorkerCancelled` failures; the after-press race remains.
- Run once alone after TASK-38990, at load 11.8: 1 failed (`[True-size1]` at `:566`), 5 passed.
- No PR lane runs the file.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both assertions wait for the reload they depend on (the worker the press or the scope change queued), not for a fixed pause; no assertion is weakened
- [ ] #2 10 runs at `-n 4` of both tests give 0 red, with `uptime` recorded; before the fix, 6 of 10 runs are red after TASK-38990
<!-- AC:END -->
