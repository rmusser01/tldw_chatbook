---
id: TASK-36389
title: Audit tests that assert what a reopened cached Console's attach arms right after the reconcile flag
status: Done
assignee:
- '@claude'
created_date: 2026-10-22 23:58
updated_date: 2026-10-30 05:16
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/UI/pump_probe.py
- backlog/docs/lessons-testing-evidence.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-35833 (PR #3212, merged 2026-10-22). On a reopened, cached Console, `_console_attach_reconciled` can be True before the work the attach queued has run. The reconcile skips the view start, and the transcript poll is armed only through the wake hook's deferred `call_later`. TASK-35833 fixed the two tests in `test_console_runtime_ownership.py` that asserted the poll right after the flag, using `pump_runs(screen, 5.0)` as a same-screen FIFO barrier. TASK-35574's switcher probe test is the same attach-reconcile race in the other direction. Other files were not audited.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every test that reopens a cached Console and then asserts a timer, poll or other deferred effect of the attach is listed in the task notes
- [x] #2 Each listed test waits for the reconcile to end and for the screen's queue to drain before that assertion, or the notes record why it cannot race
<!-- AC:END -->

## Implementation Plan

1. Find every test that reopens a cached Console: an instrumented sweep (a scratch pytest plugin that logs each attach reconcile completing with `_console_attach_view_started` already True) over every non-Architecture test file that names `ChatScreen`, `NavigateToScreen` or `_console_attach`, plus a grep for reopen paths (navigation back to chat, a modal dismissed over the Console, direct suspend/resume) in files the sweep cannot run locally. Add TASK-35574's switcher probe test.
2. For each hit, read what the test asserts after the reopen and which mechanism arms it (synchronous in the reconcile, the wake hook's `call_later`, `call_after_refresh`, `set_timer`, a worker). A test that asserts an effect once, after a wait that does not imply it, is racy; a test that polls the effect itself, or asserts something the reconcile sets synchronously, is not.
3. For each racy test, add the TASK-35833 wait (`_console_attach_reconciled and not _console_attach_reconcile_running`) plus `Tests/UI/pump_probe.py::pump_runs(screen, 5.0)`, reusing an existing helper where one exists. Show it racy at base and fixed at head under the one-window-idle probe plugin where feasible.
4. Record the full candidate list with a one-line verdict each in the notes.
5. Run each touched file with `-n 4`, the changed nodes serially 10 times, and preflight.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**What counts.** A reopened cached Console is the same ChatScreen running its attach reconcile again with `_console_attach_view_started` already True. Three paths do this: navigation back to the reusable chat route, a screen or modal popped or dismissed over the Console, and a direct suspend/resume. A "deferred effect of the attach" is anything that reconcile, or the resume it runs, arms or changes. That covers the transcript poll, the environment refresh, discovery and handoff consumption. It also covers the send path, which reads state the attach restores. (Step 1 of the plan named a `-p` plugin. It reached no `@private_profile_test` child, so the probes moved to a `sitecustomize.py`; see the lesson below.)

**How the candidates were found.**
1. An instrumented sweep. A scratch `sitecustomize.py` on `PYTHONPATH`, loaded only through `run.sh`, wrapped `_reconcile_console_after_attach` and logged each reconcile that ended with the view already started. It ran over the 198 non-Architecture test files that mount a screen, name `ChatScreen`, `NavigateToScreen` or `_console_attach`, and take a reopen path: 9,074 tests. 136 nodes reopened a cached Console, plus the two Performance tours, whose `PYTEST_CURRENT_TEST` is overridden.
2. 3,150 of those tests cannot run locally (`RecoveryRequired`, TASK-36381). An AST scan of their bodies, for a reopen path together with an attach-effect name, gave 25 hits.
3. An AST scan of every test file, for a reopen path together with an assertion on an attach-armed effect (`_console_transcript_sync_timer`, the survivor, cost-TTL and credential timers, `_discovery_calls`, the reconcile flag, `poll`), gave 15 hits.
4. TASK-35574's test was added by hand.

**How each candidate was judged.** Every hit ran at base under three probes:
- `HA_DEFER`: the reconcile starts 0.5 or 1.0 s late, and the screen's pump stays free.
- `HA_POST`: the pump is held 0.5 s after the flag flips. This holds open the TASK-35833 gap.
- The reviewer's one-window idle, in which `pilot.pause()` sees one 20 ms idle window.

The probe reproduces TASK-35833. The pre-fix delivery test failed 3 of 3 with its original message (`reconciled=True`), and the fixed test passed 3 of 3. A probe that only slept inside the reconcile caught nothing, because each `pilot.pause()` waits for work running on the current screen's pump.

**AC#1: tests that reopen a cached Console and assert an attach effect.** Each entry gives its AC#2 verdict.

Fixed. Each now calls the new `Tests/UI/pump_probe.py::console_attach_ended(screen)`. It first drains the Console's queue with `pump_runs(screen, 5.0)`, then waits for `_console_attach_reconciled and not _console_attach_reconcile_running`, then drains again. If a stage times out it raises, naming the stage and the flags. Red and green were measured at base and head under the same probe.
- `test_console_screen_reuse.py::test_console_resume_restarts_sync_timer_for_active_run`. Two ctrl+2 returns, then the poll is asserted armed with a run in flight and unarmed when idle. The reconcile arms the poll, and the test waited only for the screen. Under `HA_DEFER=1.0` with one-window idle: base red 5 of 5 (`resume must restart the transcript-sync poll ...`), head green 5 of 5. One-window idle alone does not reproduce it, because `_press_until_screen` pauses six times. The file is plain, so these runs used a temporary `@private_profile_test` copy.
- `test_console_environment_wiring.py::test_suspended_environment_stays_quiet_and_refreshes_on_resume`. A cover is popped after one pause, then the environment refresh is asserted. It needed two waits. The first, before the cover, keeps the Console from being covered before its first attach ends; covered that early, the first return runs as the mount's resume and skips the refresh. The second waits after the return. Base was red 5 of 5 under one-window idle, 3 of 3 under `HA_DEFER=1.0`+`HA_POST`, and 1 of 2 with no probe. Head was green 5 of 5 and 3 of 3. Temporary private-profile copy, as above. **At the re-sync onto ff815f8559:** dev's TASK-33620.15.1.6 (`d520897970`) polls, with a failing `_until`, for both dispatches after the return. That covers the post-return race, so this task's post-return wait is gone. This task's pre-cover wait stays, right after dev's `_until(pilot, lambda: dispatches)`. That first dispatch does not wait for the first attach: the rail-open path's guard checks only for the active screen and a displayed rail. The re-sync first dropped this wait too, and the re-sync review (ha-rev3) caught it. Control: a probe defers only the first visit's reconcile, by 2.5 s, and the test records the flag at the push. Dev's version pushed with `_console_attach_reconciled=False` 3 of 3 and went red 3 of 3 (`timed out waiting for the Environment refresh to land`). Head pushed with it True and stayed green 3 of 3. With no probe, both are green 2 of 2. The earlier probes missed this: they deferred every visit, and the push never landed with the flag False.
- `test_console_hooks_review.py::test_next_send_cancel_keeps_draft_then_allow_resumes_once`. After each review modal closes, the test sends at once. The Console's reconcile runs on its own pump, and a Send started before it ends was refused (`Draft, chat or hooks changed; Send again.`). A recorded failing run showed `cancel_pending` from `on_screen_suspend`: the review modal's suspend was handled after the review had closed. The unmodified base was red in 1 of about 70 parametrized runs under `HA_DEFER=0.5`+`HA_POST`. A copy with a timing spy was red 9 of 16, and 0 of 16 with the waits. Head: 0 of 10.
- TASK-35574's test, the other direction, on a fresh mount: `test_console_switcher_local_probe.py::test_a_probe_settling_mid_poll_still_reaches_the_console`. The test runs the poll by hand as soon as the summary mounts, and the poll returns at its attach gate. With first-mount reconciles deferred 1.0 s plus one-window idle, base was red 5 of 5 (`'Ready · not tested' == 'Not ready · refused :9099'`, TASK-35574's symptom) and head green 5 of 5. TASK-35574's own AC#2 control is not addressed here.
- `test_console_local_server_discovery_card.py::test_a_card_resumed_with_nothing_detected_runs_discovery_again` (review round 1). Its last check is negative: a second return, after a server was found, runs no discovery. Ten pauses could pass before the return's attach ran, so a regression that reruns discovery on every return could stay green. It now waits for the attach before the pauses. Negative control: a temporary edit in `on_screen_resume` that reruns discovery on every return even after a server is found (`if rearm_discovery:`, since reverted). Under `HA_DEFER=1.0` with that edit, base stayed green 3 of 3 and head went red 3 of 3 (`assert 3 == 2`). With no probe both go red. Without the edit, head is green under every probe. It is a private-profile test, so no temporary marker was needed.

Already waited for (TASK-35833): `test_console_runtime_ownership.py::test_opening_console_during_a_headless_delivery_arms_the_poll` and `::test_public_startup_console_keeps_its_claim_across_navigation`. Both pass under every probe.

Cannot race:
- `test_console_local_server_discovery_card.py::test_a_card_covered_at_launch_runs_discovery_again_on_the_first_return`. It waits for the detected action to take focus before counting.
- `::test_a_card_unblocked_and_broken_while_away_runs_discovery_on_return`. It waits with `_until` on the count it asserts.
- `test_console_live_work_handoffs.py::test_console_stage_then_navigate_then_stage_again_displays_the_newest_launch`. It polls the handoff's settlement (3 s) before reading the card.
- `test_console_provider_return.py`, all 17 nodes. Each return waits with `_until` on what it then asserts.
- `test_console_switcher_local_probe.py::test_a_refused_active_llama_cpp_reaches_the_console_status_row`. `_console_settled` polls readiness and then waits 0.3 s for the repaint.
- `test_console_hooks_review.py::test_hooks_action_is_reachable_and_returns_focus`. It awaits `_refresh_console_hooks()` itself.
- `test_console_runtime_ownership.py::test_persistent_attach_sync_failure_has_bounded_backoff_and_resume_retry` and `test_console_attach_completion.py::test_mounted_suspend_resume_rearms_after_old_reconcile_retires`. Both call the reconcile and await it themselves.
- `Tests/Performance/test_console_keystroke_work_census.py`, its "visit" phase. The phase settles over 20 pauses and drains every worker it started. The `test_ui_latency_guardrails.py` tours time switches and assert no attach effect.

Every candidate above passed under `HA_DEFER=1.0`+`HA_POST=0.5` and under one-window idle, except the fixed tests at base (the discovery test's red needs the edit above) and the two Performance tests, which were judged by reading and not probed.

**The other candidates (no attach effect asserted).** The rest of the 136 sweep hits close a modal or screen over the Console and assert that modal's own result: a send outcome, saved settings, a draft, a quit prompt, a video confirm. They are in `test_console_hook_review_send_freeze` (19), `test_console_video_send_freeze` (16), `test_console_library_controls_workflow` (7), `test_console_hooks_review` (4 others), `test_home_console_resume` (6), `test_app_keep_alive_dead_screen` (5), `test_console_endpoint_discovery` (5), `test_console_local_server_discovery_card` (7 others), `test_console_session_tab_close` (4), and 22 more files with 1 to 3 hits each. All passed under `HA_DEFER` (0.5 and 1.0) with `HA_POST`. Under one-window idle, four failed at a point before any reopen, or still failed with the wait added: endpoint round trip at its first pick, delete outcomes at its first message action, saved defaults at its first Ctrl+O, and the vLLM lab's own pane. A fifth, the workspace rail, was flaky with or without the wait (1 of 5, 1 of 4); it reads the workspace switch's own refresh. `Tests/Chat/test_console_session_settings.py::test_settings_active_compaction_close_anyway_keeps_provider_work_running_and_reopens_fresh` is red at dev with no probe (3 of 3), and waiting for the attach does not change that.

The static and blind-scan hits not listed above fall into three groups. Some build a fresh ChatScreen; they assert "screens are never cached" or reach the return through a manually pushed Console: the turn-navigation, store-continuity, persisted-chain, headless-wake, approval-first-open and pending-attachment tests, and the acceptance reattach in `test_console_runtime_ownership.py`. Some pop the Console itself: response rules, dictation. The rest are not Console reopens: activity switcher, Library, Personas, Settings.

**Limits.** Whether a test takes the cached path can depend on timing. The switcher refused-status test reopens a cached Console when run alone but did not in the sweep. The blind and static scans read only test bodies, so a reopen hidden inside a helper is missed.

**Review round 1.**
- `console_attach_ended` could return before the reattach's reconcile had started. The previous visit's flag still reads as ended while that visit's suspend, the only reset, is queued on the Console's pump, and a drain posted after the predicate held lands behind the suspend and resume. It now drains before polling. On the reviewer's Textual harness (`ha-rev1-demo.py`) the round-0 helper returned with `reconciled=False`, `running=True` and no reattach reconcile run, 3 of 3. The new helper returned after the reconcile ran, 3 of 3.
- The discovery test moved from "cannot race" to "Fixed" (above).
- The helper raises with the stage and the flags, so the call sites no longer carry a message that named only one stage.

**Re-sync onto dev `ff815f8559` (2026-10-29).** Backup branch `tmp/fu/h-attachaudit-pre-resync` at `791f585d42`. Two conflicts.
- `Tests/UI/test_console_environment_wiring.py`: took dev's post-return `_until` and kept this task's pre-cover wait (see above). The first re-sync commit dropped that wait as well; the next commit restores it.
- `backlog/docs/lessons-testing-evidence.md`: kept dev's new entries in dev's order, with this task's entry where it was added, before TASK-35578's.

The flags `console_attach_ended` reads, `_console_attach_reconciled` and `_console_attach_reconcile_running`, and the paths the fixes rely on are unchanged on dev: the reconcile, the suspend reset, the discovery rearm and the hooks `review_open` check. After the re-sync, dev's copies versus head under the same probes, 3 runs each:
- reuse (`HA_DEFER=1.0` + one-window): dev red 2 of 3, head green 3 of 3;
- switcher (late first mount + one-window): dev red 3 of 3, head green 3 of 3;
- discovery with the temporary rerun-every-return edit (since reverted), 2 runs each: dev red 1 of 2, head red 2 of 2;
- hooks and discovery without the edit: head green 3 of 3.

**Verification** (round 1, before the re-sync).
- The changed files with `-n 4`: switcher, hooks, discovery card and `test_library_media_export_no_freeze`, plus the two changed plain nodes through temporary `@private_profile_test` copies → 75 passed. The plain files themselves (`test_console_screen_reuse`, `test_console_environment_wiring`) → 50 failed, all `RecoveryRequired`. Base copies of both files fail the same 50.
- Round 0, unchanged by this round: `test_console_runtime_ownership` + `test_library_surface_swap_refused_on_real_screen` → 89 passed, 1 xfailed, 1 failed. The failure, `test_the_export_open_awaited_on_the_canvas_pump_lands_by_fallback` (`assert 2 == 1`), is also red with dev's copy of the test and of `pump_probe.py`.
- `Tests/Architecture/test_surface_swap_guard.py`, serial → 15 passed.
- Head under the probes, 3 runs each, all green: reuse, environment (one-window idle; `HA_DEFER`+`HA_POST`), switcher, hooks, discovery (`HA_DEFER`+`HA_POST`; one-window idle).
- The six changed nodes serially 10 times, each run a separate pytest → 60 of 60 passed. The two plain tests ran through temporary private-profile copies, since deleted.
- Preflight passed. Ruff reports the same findings as base.

**Verification after the re-sync.** The changed files with `-n 4` (switcher, hooks, discovery card, `test_library_media_export_no_freeze`), plus the reuse node through a temporary private-profile copy → 74 passed. `test_console_screen_reuse.py` as written → 4 errors at setup (`RecoveryRequired`), the same 4 as dev's copy of the file. `Tests/Architecture/test_surface_swap_guard.py`, serial → 15 passed. Preflight passed. Environment wait restored: its node through a temporary private-profile copy with `-n 4` → passed; `test_console_environment_wiring.py` as written → 47 `RecoveryRequired`, the same 47 as dev's copy.

**Files.** `Tests/UI/pump_probe.py` (new `console_attach_ended`), five test files (reuse, environment, hooks, switcher, discovery card), `backlog/docs/lessons-testing-evidence.md` (a `-p` probe never reaches a private-profile child; a probe must delay the reconcile's start, not its run), and this task.
<!-- SECTION:NOTES:END -->
