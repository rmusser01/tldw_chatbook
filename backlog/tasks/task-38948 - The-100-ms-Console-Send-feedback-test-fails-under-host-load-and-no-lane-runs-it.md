---
id: TASK-38948
title: "The 100 ms Console Send-feedback test fails under host load, and no lane runs it"
status: In Progress
assignee:
- '@claude'
created_date: 2026-11-05 21:35
updated_date: 2026-11-12 18:49
labels:
- follow-up
- console
- tests
- performance
priority: low
references:
- Tests/UI/test_console_received_intent_feedback.py
- .github/workflows/derived-artifacts.yml
- scripts/ui_pr_gate_census.txt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05.

**The test.** `Tests/UI/test_console_received_intent_feedback.py::test_actual_send_paints_sending_and_accepts_input_within_100ms` has three 100 ms budgets: the Sending frame (`:537`), the input mutation (`:541`) and the input frame (`:542`).
- **No lane runs it.** Only `test_actual_send_receives_intent_before_original_configuration_read` from this file is on the console-p0-gate (`.github/workflows/derived-artifacts.yml:1041`), and the census lists nothing from the file.
- **Earlier misses.** TASK-38718 (Done) recorded 117 ms misses at load about 22, and TASK-33621.22 (Done) recorded a 161 ms miss.

**Runs, through the scratch-profile wrapper.**
- Alone, 10 runs of both cases at load 50-56: 19 of 20 failed. The Sending-frame median was about 145 ms (range 58-1605) and the input-frame median about 130 ms (11-462). Each run took 27-49 s.
- Whole file at `-n 4`, 10 runs at load 31-63: 5 of 20 failed, and every other test in the file passed every run. The misses were both cases in one run and a single case in three others, one of them a 104 ms input frame. Passing Sending frames took 33-89 ms. The file's wall time was 42-115 s.
- For comparison, TASK-38718 measured a 21-33 ms Sending median at load 7-12.

These alone runs were at higher load than most of the whole-file runs, so they cannot separate load from a first-mount cold cost.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 On a host loaded below its CPU count, the test passes 10 of 10 runs alone and in its file at -n 4, with the median and maximum of all three intervals recorded
- [x] #2 Under heavy load the test either still passes 10 of 10, or reports a miss together with the measured loop lag instead of failing as a product regression; the choice is recorded in the notes
- [x] #3 A temporary delay of about 150 ms before the Sending paint makes the test fail (negative control)
- [ ] #4 The test runs on a PR lane, with the measured lane time recorded (about 27-49 s for both cases serially at load 50 on dev)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure the three intervals under the current load.
2. AC#2 choice: keep the 100 ms wall budgets, but when a separate probe process shows the host was contended in the interval's window, report the miss (record_property plus a warning) with the measured stall. Hold the event loop thread's CPU time for the Sending interval to the budget at any load, so added loop work still fails.
3. Negative control: a temporary ~150 ms delay at the start of `request_visible_send`, both busy (loop CPU) and blocking-sleep, with the result and load recorded for each.
4. Run 10x alone and 10x in-file at -n 4, recording the median and max of each interval. AC#1 needs load below 18; record the load seen.
5. Add both cases as UI census tail entries, with the measured serial time.

Round 13 (2026-11-10), after the owner's ruling (see the notes):
6. Profile the loop thread between the Send action and the Sending paint (sampler plus per-callback CPU), and record the top costs.
7. Subscription status: after the first read, a lapsed TTL keeps the last observed state while it is re-read, so the Console's credential pre-gate stops flipping missing -> pending -> missing. Red/green unit test; update the test that pinned "pending".
8. Console skill discovery: take its trust build and skill read off the UI loop in the real app (stock eager task factory), and stop an older discovery from publishing over a newer one. Red/green tests for both.
9. Re-measure the Sending interval (loop CPU and wall, both routes, with load) before and after, alone and in-file.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Test-only change. Base 8b762b52fa, 2026-11-07. The host's load average never fell below its 18 CPUs during this work (29-190), so AC#1 is not met and the task stays In Progress.

**AC#2 choice: two checks, and wall time is relaxed under load.** `_assert_send_budget` checks each of the three intervals (Sending frame, input mutation, input frame) two ways.
1. **Loop CPU, at any load, never relaxed.** The event loop thread's CPU time in the interval must stay within 100 ms. It is stamped on the loop thread at both ends: `action_cpu`/`feedback_frame_cpu` for Sending; for input, `key_cpu` when the loop starts handling the typed key, and `mutated_cpu`/`frame_cpu` through property setters on `_TimedInputFrame`. Work added on the loop therefore fails even on a loaded runner, so this check cannot pass vacuously.
2. **Wall time, enforced only while the 1-minute load average is under `os.cpu_count()`.** At or over the CPU count, a wall-clock miss is **not failed**. It is reported as a `<label>_host_load_miss` property and a `UserWarning` ("... so wall time was not enforced; the loop ran X ms on the CPU and was off it Y ms").

**This is a deliberate weakening on a loaded host:** a blocking wait (sleep or I/O) added before the paint passes there with a warning. It is caught only on a host that is not oversubscribed. The UI lane runs this serially on a 4-vCPU runner, so the wall budget applies there unless that runner is itself oversubscribed. The warning says when it did not apply.

Two stricter excuses were tried first and dropped, with evidence (lesson in `lessons-testing-evidence.md`):
- Subtracting a separate probe process's scheduling stalls still failed at load 104-130 (406 ms, 181 ms of probe stall), and its 10%-stall contention signal missed a 181 ms miss at load 48.

Also changed: the held provider read waited only 0.5 s for the paints. At load 105 that released the read before a slow frame, and `frame_while_held` failed. The wait is the test's patience, not a budget (the budgets come from the frames' own timestamps). It is now `_HELD_READER_SECONDS = 5.0`, and the release and join waits after it scale with it.

**Negative controls (AC#3)**, temporary edits at the start of `send_acknowledgement.request_visible_send`:
- **150 ms of loop CPU** (spin until `thread_time` advances 150 ms), at real load 84-88: both routes failed on the CPU check (211/193 ms of loop CPU).
- **A blocking 150 ms sleep**, at real load 74: both routes passed with warnings (wall 261-699 ms reported). This is the weakening above. With the test's load reading forced to 0 (an unloaded host's verdict) at load 43-46, both routes failed (215/202 ms), and with no delay they passed.
- A wall-clock busy-wait is **not** a fixed amount of work. At load 85 a 150 ms one burned only 53-75 ms of CPU and passed with warnings, so the control that counts is the CPU-work one.

**Runs at `-n 4`, final code** (AC#1's measurements; every one at load over 18):
- Alone, 10 runs at load 83-102: 10 of 10 passed. Over 20 cases, median/max: Sending 296/750 ms wall, 42.8/78.1 ms loop CPU; input mutation 89.8/323 ms (CPU at most 34.9); input frame 179.6/417 ms (CPU at most 43.9).
- In its file, 10 runs at load 93-118: the test passed 10 of 10 (both cases every run). Sending loop CPU median/max 52.3/78.9 ms; 50 wall misses reported. Other tests in the file, untouched here, failed 1-6 per run at that load on their own 15 s bounded waits (`test_pressed_capture_survives_identical_retyping_before_promotion`, `test_enter_sends_pressed_body_after_clear_and_type_before_paint_dispatch`, `test_navigation_after_receipt_…`, `test_send_feedback_survives_preparation_and_provider_wait`). Side by side at load 67-112, base gave 2 failed (this test's two routes) / 15 passed, and head gave 17 passed.
- An earlier variant's in-file series at load 33-56: 10 of 10, 17 passed each. That variant had the wall gate but no CPU check, and the same intervals.
- Probe of loop CPU against GC over 40 more cases at load 57-99: Sending at most 73 ms, of which GC at most 5 ms; input at most 50 ms.

**Lane (AC#4).** UI census tail, one entry per route (`[enter]`, `[send-button]`). Serially in the lane's invocation at load about 130: 9.1 and 6.2 s of call, 25 s of session for both.

**Review r12t-rev1 fixes (2026-11-07).** The reviewer ruled the wall-time gate acceptable: the UI shards run serial pytest on 4-vCPU runners, so the load stays under the CPU count where it gates merges.
- **The loop-CPU check false-failed** in the reviewer's six-file `-n 4` run at load 40-60: 483 ms of CPU against 100 ms (1456 ms wall), late in a busy worker. My earlier 258 ms at load 31 was the same thing. The likely cause is a gen-2 GC over the worker's accumulated heap inside the interval, plus CPU-time inflation under oversubscription. Fix: `gc.collect()` just before `action_at`. Round 1 also froze and unfroze the heap; review r12t-rev2 dropped that pair, see below. Re-measured in 10 mixed runs at `-n 4` and load 25-48: prompts modal, composer restyle, Inspector section, response rules and send-admission files first, the 100 ms node last. 20 of 20 cases passed. Sending loop CPU was 32-75 ms (wall 53-248 ms), input-frame CPU 7-70 ms. The cause of the two outliers is inferred, not proven; AC#2 stays ticked on these 20 clean cases.
- **Sending held-read bound.** `feedback_while_held` is stamped at the Sending frame and asserted, after a bounded wait for a late Sending frame. So a Sending paint that lands after the held read is released fails at any load, wall gate or not. Shown red with a temporary edit that delays every Sending source by 1.0 s: the ack's immediate `sync_run_chip` behind a timer, `run_copy`, the validating copy and the composer's run feedback. In 2 of 2 runs (load 45-47) one route failed with "Sending painted only after the held read was released", and the other read `None` because Sending never painted in time.
- Touched file, 3 runs at `-n 4` (load 25-61): 17 of 17 each. The census checker and preflight pass.

**Review r12t-rev2 fix (2026-11-07).** `gc.unfreeze()` in the test's `finally` emptied the whole worker's frozen set. That set includes what the product freezes at UI-ready (`freeze_long_lived_heap`) and at screen pre-import, so later timing-sensitive tests in the same worker or serial shard would carry that heap in gen 2. The `gc.freeze()`/`gc.unfreeze()` pair is dropped; `gc.collect()` stays. On CPython 3.12 a full collection zeroes the long-lived pending count, so no automatic full pass can run within the short interval.
- Re-measured with the reviewer's six-file set (endpoint discovery, Inspector section, prompts modal, received-intent feedback, response rules, store continuity) at `-n 4`, 6 runs at load 31-67.
- The node passed 12 of 12 cases. Sending loop CPU was 36-76 ms (wall 74-369 ms), input-frame CPU 7-71 ms.
- The other failures in those runs were in other files: the 4 RecoveryRequired store-continuity nodes, TASK-35567's node, the response-rules load failures already seen on dev, and at load 49-61 single failures of seven prompts-modal tests outside this task.

**Files.** `Tests/UI/test_console_received_intent_feedback.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `backlog/docs/lessons-testing-evidence.md`.

### Round 13 (2026-11-10, base ae9f9e8d21)

**Owner decision (2026-11-10, AskUserQuestion), verbatim:** "Fix the two loop hogs (Recommended)": "Move skill discovery's file I/O to a worker thread and make the subscription status cache keep its last value while it refreshes. Both are real UI-jank bugs; paint order is unchanged. Re-measure, and come back only if it's still over." Option B (reorder the acknowledgement paint) and option C (let the screen settle before the test's interval) were not chosen.

**Profile of the Sending interval, before.** The profile was thread-scoped (a stack sampler plus `thread_time` per asyncio `Handle._run` on the loop thread). 39 cases at load 22-105, with the probe on (it adds 10-20%).
- **The send's own path: 33-53 ms (median 43).**
  - The first-message Sending frame: 15-36 ms. That is one reflow of 120 widgets (5-7 ms) plus a render of about 5,300 dirty cells (13-18 ms), mostly from removing the empty-state panel.
  - `paint_acknowledgement`: 10-16 ms. The row mount is 48% of it, the header's status classes 19% and the composer's acknowledged state 18%.
  - Key dispatch, including the first-send import of `send_acknowledgement`: 3-12 ms. GC: 3-5 ms. No DB reads on the loop.
- **Work the send did not cause: up to 57 ms more (median 12).** Most of it ran before the key reached `on_key`:
  - credential-readiness passes, 7-30 ms;
  - the post-mount whole sync, 20-35 ms;
  - the coalesced control-bar sync, 4-13 ms;
  - a cold skill discovery, 57-85 ms;
  - extra frames, 2-25 ms;
  - the 0.2 s draft-spend refresh from the test's `load_draft`, 4-14 ms.

**Fix 1: the subscription status keeps its last value while it is re-read** (`LLM_Calls/anthropic_subscription.py`).
- **The bug.** `_SubscriptionReadinessCache.status()` answered "pending" whenever its 5 s TTL lapsed. TASK-34956's credential pre-gate (status, revision, version) therefore went missing → pending → missing every 5 s. That is two full readiness rebuilds on the UI loop, 2-15 ms each, for an unchanged credential, and it defeated task-33081's revision gate. In the profile it landed in the interval in 2 of 6 cases on a settled screen.
- **The fix.** After the first completed read, a lapsed TTL returns the last observed state while one background re-read runs. "pending" now means only "never read yet".
- **What users see.** The Console settings modal, the send-disabled line, and Settings' Overview and Key check no longer flicker to "Checking Claude…" / "checking login" every 5 s. The User Guide describes only the first check, which is unchanged.
- **Red/green.**
  - New `Tests/Chat/test_subscription_background_readiness.py::test_lapsed_ttl_reread_of_unchanged_credential_never_moves_the_ui_gate`. Red on base: `[('pending', 1), ...] == [('missing', 1)] * 3` fails. Green after.
  - `test_unchanged_refresh_does_not_bump_revision` asserted "pending" for a stale TTL. That was the flicker itself, and the owner's ruling removes it. It now asserts the last state ("ready") while the re-read runs. Its revision checks stay unchanged: the same revision for an unchanged credential, a new one once the file is removed, then "missing".
  - Four UI tests pinned the flicker after a TTL lapse. In `Tests/UI/test_console_subscription_readiness.py` they are the modal and the send line; in `Tests/UI/test_settings_subscription_readiness.py`, the Overview and the Key check. They now wait for the re-read, hold it past a 0.25 s poll tick, and assert that the last state stays with no "checking" copy. Red on base: each showed "Checking Claude…" or "checking login". Green after.

**Fix 2: Console skill discovery runs off the UI loop in the real app** (`app_skill_wiring.py`, `UI/Console_Modules/skill.py`).
- **The bug.** `_capture_console_skill_trust_preparation` required `loop.get_task_factory() is None`. Textual's `App.run_async` installs `asyncio.eager_task_factory`, so every real run refused that preparation.
  - The context read then found no trust service and fell back to `await get_context()` on the loop. That built the trust service through the sync property and read every skill (file I/O, pydantic models) synchronously.
  - It ran eagerly inside `_start_console_view_after_reconciliation`, in the mount itself.
  - `run_test` installs no factory, so every mounted test took the off-loop path. A subprocess probe of a real `TldwCli` with the eager factory showed `LocalSkillsService.get_context` on `MainThread`.
- **The fix.** The guard now admits the stock eager factory, checked by identity; a custom factory still refuses. The existing off-loop path then runs:
  - the trust build in the App worker (`ensure_local_skill_trust_service` → `run_preparation_read`);
  - then the read (`_read_console_skill_context`).

  Custom (non-stock) collaborators keep their pinned loop affinity (`Tests/Performance/test_console_skill_context_owned.py`).
- **Superseded discoveries.** Reads now finish in any order. `_refresh_console_skill_candidates` takes a serial when it starts and publishes only while it is still the newest. An older read can no longer replace newer candidates, including a failed one that fails closed to no skills.
- **Red/green.**
  - `Tests/UI/test_console_skill_context_worker.py::test_mounted_discovery_builds_and_reads_off_the_loop_under_the_eager_factory` (a real `TldwCli` in a subprocess, eager factory installed). Red on base: "skill read on the UI loop". Green after.
  - `Tests/UI/test_console_skill_controller.py::test_a_superseded_refresh_never_publishes_over_a_newer_one[stale|failed]`. Red on base: `['stale'] == ['fresh']` and `[] == ['fresh']` both fail. Green after.
- **Limit.** In the 100 ms test's own harness the discovery still runs on the loop. `Tests/UI/app_factory.py` gives the app a `SimpleNamespace` runtime policy, which the reader classifies as a custom collaborator. So fix 2 removes a 57-85 ms cold block from the real app's mount, but does not change that test.

**Re-measured, before and after.** Paired runs on the same host minutes apart: base ae9f9e8d21 and fix 064f161b9a, alternated run by run. 10 runs alone and 10 in the file, each at `-n 4`, both routes. Load was 35-63 on 18 CPUs, so the wall budget was reported but not enforced. Each cell is median / max in ms over 10 cases.

| Run | Route | Sending loop CPU | Sending wall | Input mutation | Input frame |
|---|---|---|---|---|---|
| Alone, base | enter | 48 / 110 | 69 / 139 | 22 / 78 | 34 / 108 |
| Alone, base | send button | 44 / 50 | 81 / 164 | 18 / 111 | 31 / 125 |
| Alone, fix | enter | 43 / 57 | 82 / 115 | 38 / 129 | 47 / 136 |
| Alone, fix | send button | 39 / 58 | 63 / 99 | 4 / 73 | 9 / 80 |
| In file, base | enter | 58 / 69 | 132 / 296 | 5 / 196 | 15 / 205 |
| In file, base | send button | 38 / 72 | 122 / 247 | 33 / 302 | 62 / 335 |
| In file, fix | enter | 58 / 81 | 143 / 310 | 52 / 147 | 63 / 156 |
| In file, fix | send button | 42 / 58 | 150 / 289 | 38 / 88 | 44 / 121 |

- **Base:** 1 of 40 cases failed, an alone `[enter]` case with 110 ms of loop CPU (138 ms wall). The other 39 passed.
- **Fix:** 40 of 40 passed. Its Sending loop CPU peaked at 58 ms alone and 81 ms in the file.
- **These fixes do not move this test's interval measurably.** The Sending loop CPU medians are the same within noise (alone 46 → 40, in file 48 → 49).
  - In this harness the skill discovery stays on the loop because of the `SimpleNamespace` runtime policy (see Limit above).
  - A credential TTL flip lands in the window only now and then.
  - What remains is the send's own 33-53 ms plus whatever mount-settling work happens to coincide.
- **For comparison:** the coordinator's base runs at load 6-12 (dev ae9f9e8d21, 20 cases each) read Sending loop CPU median 55 / max 107 alone and 65 / 105 in the file, with 4 and 3 failures.

**Covering runs (`-n 4`, load 23-46).**
- **Subscription set (20 files): 936 passed, 16 failed.** 12 of the 16 fail the same way on base: 8 in the Personas subscription-readiness file (some with RecoveryRequired) and 4 in the first-run wizard file. The other 4 were the flicker pins above, now updated and green: Console and Settings readiness, 13 of 13.
- **Skill set (15 files): 215 passed, 18 failed, 1 xfailed.** 17 of the 18 fail on base too: command popup ×9, visit dispatch ×2, controller wiring ×1, held pointer `[skill]`, managed received acceptance, and stock-owned `[send_enter]` / `[send_button]`.
  - The 18th is `Tests/Performance/test_console_skill_trust_stock_owned.py::…[app_closing]`. It hangs to its 240 s subprocess timeout now and then, on base and fix alike, under concurrent load. Base: 2 of 16 four-way-parallel runs and 0 of 19 serial. Fix: 4 of 59 overall, 0 of them in 16 four-way-parallel runs.
  - An asyncio task dump showed the script's outer `wait_for` pending with the loop idle. That is pre-existing; see the report's follow-ups.
- **Serial Architecture ratchets: 269 passed, 1 skipped.** Module size, screen size, wave-6 inventory, controller self-attributes, chat code identity.
- **New and updated test files on the final commit:** 8 files at `-n 4`, load 42-46. The first run had 2 Settings failures:
  - the first-read "being checked" wait, which is unchanged by this round and has a 3 s limit;
  - a `NoMatches` in `test_settings_missing_subscription_reports_owner_recovery_after_completion`, a test this round does not touch.

  Two reruns of all 8 files then passed 86 of 86 each, and the Settings file alone passed 8 of 8 three times.
- **Preflight passes.** The UI census gains two tail entries (the two new skill nodes) and the floor is 1915. The new Tests/Chat node is on no PR lane.

**Files (round 13).**
- Product: `tldw_chatbook/LLM_Calls/anthropic_subscription.py`, `tldw_chatbook/app_skill_wiring.py`, `tldw_chatbook/UI/Console_Modules/skill.py`.
- Tests: `Tests/Chat/test_subscription_background_readiness.py`, `Tests/UI/test_console_subscription_readiness.py`, `Tests/UI/test_settings_subscription_readiness.py`, `Tests/UI/test_console_skill_context_worker.py`, `Tests/UI/test_console_skill_controller.py`.
- Census: `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.
- Lessons: `backlog/docs/lessons-testing-evidence.md`, two entries.

**AC#1 is still not met.** Every run in this round was at load 23-46 on 18 CPUs.

**Review r13n-rev1 fixes (2026-11-10).**
- **I1: one more test pinned the flicker.** `Tests/UI/test_personas_subscription_readiness.py::test_handoff_readiness_refreshes_expiry_and_stale_snapshot_without_edits` waited for "Checking Claude" after a TTL lapse.
  - Locally it was hidden: the whole file fails under the plain command with RecoveryRequired, on base and head alike (the known local trap).
  - Re-pinned the same way as the other four: wait for the second reader, hold it past a 0.25 s poll tick, assert "credential is expired" stays with no "Checking Claude" and Start chat still disabled, then release and wait for "Ready to chat".
  - Verified under a temporary `@private_profile_test` marker, removed before the commit: green 3 of 3 at load 37-50. Red with base `anthropic_subscription.py`: "Chat now blocked: … Checking Claude subscription credential …".
  - Not added to the census. Under the plain command it still hits RecoveryRequired here, so I cannot check it under its lane's profile.
- **M2: the eager trust-setup lifecycle now has a test.** New `Tests/UI/test_console_skill_context_worker.py::test_eager_mount_discovery_closing_during_held_trust_build_publishes_nothing[app_closing|runtime_dispose]`.
  - A real `TldwCli` runs under the eager factory, and the test holds the mount discovery's off-loop trust build. It then closes the app (`on_shutdown_request`) or disposes the Console runtime.
  - It asserts that the close waits for the held build, the runtime ends disposed, no skill is read, and neither the trust service nor any candidates are published.
  - Green 3 of 3 runs (both cases) at load 50-54.
  - Negative control, a temporary edit of `ensure_local_skill_trust_service` (no source refusal, no owner check after the build): `[runtime_dispose]` fails with "closed owner published trust". `[app_closing]` still passes, because shutdown refuses the build elsewhere.
- **M3: lanes for fix 1's tests.**
  - The UI census tail gains 5 nodes: the four subscription stale-snapshot pins and the new M2 node. The floor is now 1920. Serially, all of round 13's census nodes take about 2.6 min together at load about 55, the largest 27 s.
  - `Tests/Chat/test_subscription_background_readiness.py` joins the recovered-suites job in `.github/workflows/derived-artifacts.yml`, in its sandboxed group. Its pin is in `Tests/CI/test_ci_queue_pressure_contract.py::RECOVERED_SUITES_TARGETS`.
  - That file alone, serial with `--storage-lease-guard`: 11 passed in 5.7 s. The job's whole invocation run locally: 744 passed, 1 skipped (16 min serial at load about 55). Tests/CI: 1240 passed, 1 skipped.
- **Covering run, 6 files at `-n 4`:** 42 passed, 8 failed. The 8 are the Personas file's RecoveryRequired failures, the same 8 as on base. Preflight passes.
- **Files:** `Tests/UI/test_personas_subscription_readiness.py`, `Tests/UI/test_console_skill_context_worker.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`.

**Off the PR gate (2026-11-10, owner decision; AC#4 unticked).** On CI runners the test failed intermittently in the required UI lane after round 12 merged: PR run 42841511400 (branch fu/r12-lanes), job 129933421436 shard 16 `[send-button]` "the event loop spent 112 ms of CPU against 100 ms (123 ms wall)", job 129933421583 shard 15 `[enter]` 119 ms / 128 ms. Owner decision 2026-11-10 (AskUserQuestion): "Off the gate, reorder next (Recommended)", option text: "Move it out of the required PR lane now (it keeps running in the dev-push job), and implement the paint reorder: 'Sending…' paints one frame before the user row, which measured Sending CPU median 56→30 ms. Re-gate it once it passes on CI." Both census entries were removed in the hotfix branch fu/hotfix-restyle (floor 1913 → 1911); the subagent's attempt was refused by the permission system as a CI bypass, and the owner then authorized the coordinator to make this one edit. The paint reorder and the dev-push watch follow on the round-13 send branch; re-gating waits for CI evidence.

**Mounted-ready qualification on the merged PR #3480 tree (November 12, 2026 UTC).**

Exact source `e10ce63fe8bbd03104849aae007835e82c88f719` was loaded in the test process and
all four xdist workers. Enter and Send-button both passed ten isolated runs and ten
whole-file runs at `-n 4` on a host with 18 logical CPUs. Every qualifying whole-file run
passed all 17 tests. The recorded one-minute load was 7.63–10.71 for isolated runs
and 8.35–14.06 for whole-file runs. A supplemental whole-file run also passed
at load 18.51 and was excluded from AC#1. No timing assertion used a load waiver.

| Run | Sending frame median / max | Input mutation median / max | Input frame median / max |
| --- | ---: | ---: | ---: |
| Isolated | 51.9 / 74.6 ms | 2.4 / 3.6 ms | 6.8 / 37.5 ms |
| Whole file, four workers | 43.9 / 78.0 ms | 2.4 / 3.3 ms | 7.4 / 31.1 ms |

The timer covers the original Send, held native configuration read, actual frame, and
input while that read is held. The fixture settles the mount-only discovery worker
before timing. This qualification applies to mounted-ready Send and does not cover
Send overlapping cold mount discovery or a physical terminal flush.

AC#1 is met on that tree. AC#4 remains open pending hosted qualification and re-enrollment.

### Round 13, paint reorder (2026-11-10)

**Owner decision (2026-11-10, AskUserQuestion), verbatim:** "Off the gate, reorder next (Recommended)": "Move it out of the required PR lane now (it keeps running in the dev-push job), and implement the paint reorder: 'Sending…' paints one frame before the user row, which measured Sending CPU median 56→30 ms. Re-gate it once it passes on CI."
- **The evidence it rests on:** CI's round-12 PR runs failed the test at 112 ms of loop CPU (`[send-button]`, shard 16) and 119 ms (`[enter]`, shard 15).
- **This supersedes the earlier "paint order unchanged" ruling.**
- **AC#4 is unticked.** Dev e0186fa10a (#3500, the fu/hotfix-restyle hotfix) already took the two nodes off the PR census and unticked AC#4. This branch does not re-add them; re-gating waits for CI evidence.

**Change** (`tldw_chatbook/UI/Console_Modules/send_acknowledgement.py`; commits 1b444d220f and d583254ddd).
- **What paints when.** The status line, composer, header and tabs are pushed first and painted in a frame of their own. The user row mounts after that frame.
- **The row is held back until then.** From `begin` until its own mount, the acknowledgement keeps the pending row out of every projection (`hold_row` / `show_row`), so nothing else can draw it earlier.
- **The deferred mount** (`_mount_row`) is posted on the status line (`_row_anchor`). It re-checks that the line's "Sending…" repaint has been drawn (`_sending_frame_pending`). Of the flags it reads, only the screen-level ones (`_dirty_widgets`, the screen's `_layout_required`) clear when the frame is drawn; the widget's own `_repaint_required` and `_layout_required` clear when it hands its repaint to the screen. `console_transcript_fill._ready` reads only `_layout_required`, not the other two. The mount waits at most `_PAINT_HOPS` frame-length hops, then shows, projects and mounts the row. (Corrected in review r13n-rev3: this line used to say both modules read the same flags.)
- **The hand-off to the send** (`_RowMount.after`) runs after that mount, and still waits until the row is laid out. The send never starts before the row is on screen.
- **Fallback.** If no mount is owed (another tab is shown, or the paint failed), the row is shown again and a whole sync repaints it. With no status line, or a refused post, the row mounts at once, as before.
- **Why each piece is needed.** These are first-frame coalescing rates: "Sending…" and the row drawn in the same frame.
  - **Screen-level `call_after_refresh`:** 3 of 6 profiled cases. It can run before the child widgets post their repaints.
  - **Anchored on the status line only:** 3 of 40 received-intent cases, with failures at 100-113 ms of loop CPU.
    - A probe found the first mount of the row coming from a whole sync (`_sync_native_console_chat_ui`) that ran between the paint and the deferred mount, with 15-16 widgets still dirty.
    - Separately, Textual 8's `_on_timer_update` runs the screen's after-refresh callbacks even while another task's `batch_update` holds the frame back.
  - **Holding the row plus the frame check:** 0 of 40.
- **New diagnostic.** One warning, "Console send acknowledgement row mount failed (exception_type={})", which logs the exception type only. The production diagnostic inventory was reviewed with `--statements` and rewritten.

**TASK-33620.5's live ordering claims, re-checked. Both still hold, and no pin was weakened.**
- **Tabs before the row:** the tabs sync stays in the first part.
- **The whole acknowledgement on screen before admission:** `test_enter_paints_a_sending_row_before_admission_and_validation` (80x24) and the extended file's 160x45 and 235x52 variants pass unchanged, including their gap rule.
- The one intended change, the row appearing a frame after "Sending…", is covered by the new tests below.

**Red/green.** Both new tests are in `Tests/UI/test_console_send_acknowledgement.py`, a whole-file census entry, so they are already on the lane.
- **`test_sending_paints_a_frame_before_the_row_and_the_send_waits_for_it[80x24|160x45]`** asserts:
  - the first frame with the run segment and a running header comes strictly before the first frame with the user row;
  - the composer still shows the draft in that frame, with no empty-draft reason;
  - the row's frame comes before the send starts, and the frame on screen at the send has the row.

  Red on cd7d3aa10e for both sizes ("Sending and the row painted in one frame"). Green in 15 of 15 runs (30 cases) at load 17-23.
- **`test_a_whole_sync_before_the_rows_frame_does_not_mount_the_row`** runs `_sync_native_console_chat_ui` between the paint and the deferred mount and asserts the row is not mounted by it. Red on 1b444d220f (`[True] == [False]`). Green on d583254ddd.

**Covering runs, `-n 4`, on d583254ddd.**
- **Acknowledgement core, extended, resend guard and the received-intent file:** 64 passed and 1 failed, at load 31-39. The failure was the 100 ms node `[send-button]` in the file: 133 ms of loop CPU, 218 ms wall. That is the incidental tail described under the measurements below.
- **The other 34 files that touch the acknowledgement:** 948 passed, 18 failed. They are the same 18 that fail with cd7d3aa10e's `send_acknowledgement.py`:
  - canvas policy worker ×3;
  - send draft snapshot ×4;
  - tab switch binds composer ×8;
  - Library pause recovery ×2;
  - Library auto outcome ×1.
- **Size and import budgets:** size ratchets 224 passed. Preimport payload budget and first-send preimport: 25 passed.
- **Other checks:** the textual worker contract passes, and so does preflight.

**Dev-push watch.** I took the simpler path: no workflow edit on this branch, because the `library-shell-dev` job is on fu/r13-lib, not dev. The step to add at restack, as the last step of `library-shell-dev`:

```yaml
      # TASK-38948 (owner decision 2026-11-10, "Off the gate, reorder next"):
      # the 100 ms Send-feedback nodes left the required UI lane; they keep
      # running here on every push to dev until CI shows them passing.
      - name: Watch the 100 ms Send-feedback nodes
        if: strategy.job-index == 0
        run: |
          pytest \
            "Tests/UI/test_console_received_intent_feedback.py::test_actual_send_paints_sending_and_accepts_input_within_100ms[enter]" \
            "Tests/UI/test_console_received_intent_feedback.py::test_actual_send_paints_sending_and_accepts_input_within_100ms[send-button]" \
            --timeout=180 \
            --durations=0 --durations-min=0 \
            --tb=short
```

**Repeated runs on d583254ddd:** both new tests passed 8 of 8 runs (24 cases) at load 15-19.

**Re-measured, before and after the reorder** (2026-11-12). These are paired runs of the 100 ms test: cd7d3aa10e's `send_acknowledgement.py` (base) against d583254ddd's (fix), alternated run by run. 10 runs alone and 10 in the file, `-n 4`, both routes, load 21-28 on 18 CPUs. Each cell is median / max in ms over 10 cases. "Row" is the wall time to the user row's frame, and "same frame" counts cases where "Sending…" and the row were drawn together.

| Run | Route | Sending loop CPU | Sending wall | Row wall | Same frame |
|---|---|---|---|---|---|
| Alone, base | enter | 45 / 70 | 72 / 105 | 72 / 105 | 10 |
| Alone, base | send button | 46 / 116 | 73 / 141 | 73 / 140 | 10 |
| Alone, fix | enter | 32 / 123 | 49 / 162 | 103 / 210 | 0 |
| Alone, fix | send button | 32 / 85 | 51 / 95 | 89 / 126 | 1 |
| In file, base | enter | 44 / 132 | 69 / 170 | 69 / 169 | 9 |
| In file, base | send button | 37 / 81 | 50 / 91 | 50 / 90 | 10 |
| In file, fix | enter | 34 / 79 | 61 / 103 | 117 / 155 | 0 |
| In file, fix | send button | 30 / 90 | 46 / 117 | 87 / 192 | 0 |

- **Medians.** The Sending loop CPU median falls about 25-30%: alone 45-46 → 32 ms, in the file 37-44 → 30-34 ms. The Sending wall median falls by 8-24 ms in three of the four route rows (in-file `[enter]` goes 69 → 61 ms).
- **The row now paints one frame later.** Its median wall is 87-117 ms, against 50-73 ms when it came with "Sending…".
- **Failures: 2 of 40 on each side.**
  - Base: 116 ms (alone, `[send-button]`) and 132 ms (in file, `[enter]`), both with "Sending…" and the row in one frame.
  - Fix: 104 ms and 123 ms (alone, `[enter]`), both with "Sending…" alone in its frame. These are the incidental tail: settling work that lands in the window, outside the send path.
- **The one coalesced fix case** read 62 ms and passed. It is the hop-bounded fallback: past `_PAINT_HOPS`, the row mounts anyway.
- **AC#1 is not met.** Load was 21-28, and the fix still misses the 100 ms loop-CPU budget in some cases from the incidental tail. Re-gating waits for CI evidence, per the ruling.
### Round 13, review r13n-rev3 (2026-11-12)

**Owner decision (2026-11-12, AskUserQuestion), verbatim:** "Accept the frame (Recommended)": "Keep the reorder as is. Record in TASK-33620.5 and TASK-38948 that the rule yields for that one frame, and pin the behaviour in the test. Clearing the empty state in the first frame would put back much of the CPU cost the reorder removed."
- **What the rule yields.** In a fresh chat, the "Sending…"-only frame still shows the empty transcript ("Ready · … type a message to begin") for the roughly 30-55 ms until the row's frame. TASK-33620.5 AC#2 otherwise forbids that.
- A dated note recording this is in TASK-33620.5's notes too.

**Pins added** (both in `Tests/UI/test_console_send_acknowledgement.py`, a whole-file census entry).
- **The empty state in the Sending frame.** `test_sending_paints_a_frame_before_the_row_and_the_send_waits_for_it[80x24|160x45]` now also asserts that the empty state is visible in the Sending frame and gone by the row's frame.
  - The file's `paint_state` matched "type a message to begin" line by line, and at 80x24 the empty state wraps "type a" / "message to begin.". It now joins the lines before matching.
  - That also makes the existing "no empty state after the acknowledgement" checks meaningful at 80x24; they used to be vacuous there. They pass.
  - Red under a temporary edit that hides the empty-state panel in the first part of the paint: both sizes failed with "the Sending frame no longer shows the empty state". Green on head.
- **I2: the draw check.** New `test_a_batch_holding_the_sending_frame_still_draws_it_before_the_row`, the reviewer's deterministic case.
  - It wraps `paint_acknowledgement` so the app's batch is opened before the real paint and closed 50 ms after it.
  - It asserts the first frame with "Sending…" comes before the first frame with the row.
  - Red in 3 of 3 runs with `_sending_frame_pending` forced to return False. Green in 8 of 8 runs on head.

**Fixes.**
- **The mount is returned, not stored on the screen.** `paint_acknowledgement` now returns its `_RowMount`, or `None` when another tab is shown, and `_paint_then` attaches the hand-off to that object. The `ROW_MOUNT_ATTRIBUTE` screen attribute and its cleanup are gone, so a stale mount that never ran can no longer capture a later send's hand-off.
- **`_mount_row` mounts whenever the row is still pending,** not only when it was the call that un-held it. A wrapper that discards the return value then still ends with the row mounted.
- **`schedule_acknowledged_send`** now calls `ack.show_row` only when there is a token.
- **Docstrings.** `_sending_frame_pending` now says which flags clear when the frame is drawn and which clear on the hand-over. `_row_anchor` now gives the real reasons the order holds: the tabs-sync await before the post, the screen running after-refresh callbacks only when it is idle with nothing dirty, and the draw check. The line forwards `InvokeLater` before its own idle posts the `Update`, so the post alone does not order anything.
- **Note (b)** is corrected above.

**Runs** (`-n 4`, through run.sh):
- **Acknowledgement core, extended, resend guard and the received-intent file:** 66 passed at load 13-21. That includes the two new tests and the stricter 80x24 empty-state checks.
- **The other 34 files that touch the acknowledgement:** 946 passed, 20 failed, at load 14-21.
  - 18 are the same 18 as before.
  - `test_console_library_pause_recovery.py::…[stream]` fails alone on cd7d3aa10e and d583254ddd too, so it is not from this round.
  - `test_console_sync_tick_storage.py::test_sync_ticks_during_a_run_check_the_character_scope_once_a_second` failed now and then on its last check (`32 >= 34`).
    - **This round's first reading was wrong.** It blamed the reorder (4 of 24 on the reorder versions against 0 of 28 on cd7d3aa10e, sampled at different times and loads).
    - **Review r13n-rev4 ran an interleaved A/B.** It used cd7d3aa10e's and d640ccadee's `send_acknowledgement.py`, alternated per iteration under the same load. They failed 20 of 64 and 17 of 64, so the reorder is not the cause.
    - **The real cause, from 35 caught failures.** At the first idle tick, `_console_control_bar_replay_whole_sync` and `_console_sync_requested` were both True.
      - `config_sync.sync_console_core_state` (`config_sync.py:530`) sets them when another thread holds both config locks (`cold=True`).
      - Both of the test's ticks then returned False at `_sync_native_console_chat_ui`'s entry guard and never reached the Character presentation work.
      - That is the designed TASK-35645 / TASK-36511 whole-sync replay, and the test did not wait it out.
    - **Fixed in the test (review r13n-rev4).** `_tick` now also waits for no owed or scheduled replay: `_settled` checks `_console_sync_in_progress`, `_console_control_bar_replay_whole_sync` and `_console_control_bar_sync_scheduled`. The reviewer measured it fixed at 0 of 48 against 17 of 48 unfixed, at load 14-31. My 10 runs of the file at `-n 4` passed 10 of 10 (2 of 2 tests each) at load 7-8; low load was the condition the flake did not need, so the reviewer's loaded A/B is the stronger evidence.
    - The file is still on no PR lane; laning it is a follow-up.
- **Size ratchets:** 224 passed. Preimport payload budget and first-send preimport: 25 passed.
- **Preflight:** passes. The diagnostic inventory verified with no change this round.
### Round 13, review r13n-rev4 (2026-11-12)
- **The sync-tick failure is not from the reorder.** The Round 13 "Runs" note above now gives the real cause and the reviewer's interleaved A/B (20 of 64 against 17 of 64), and `Tests/UI/test_console_sync_tick_storage.py::_tick` waits out an owed or scheduled whole-sync replay.
- **Two wrappers return the paint's result now.** `test_console_send_resend_guard.py`'s `held_paint` and this file's `paint_then_whole_sync` used to drop `paint_acknowledgement`'s returned row mount, so they ran the degraded path. Their own assertions are unchanged.
- **Covering runs, `-n 4`, load 7-9.** The acknowledgement core, extended, resend guard, received-intent and sync-tick files: 68 passed. Preflight passes.

**Dev-push watch committed (2026-11-12 restack).** The step "Watch the 100 ms Send-feedback nodes" is the last step of `library-shell-dev` in `.github/workflows/derived-artifacts.yml` (first shard only, the UI lane's flags; not in the required aggregate), pinned by `Tests/CI/test_derived_artifacts_workflow.py::test_library_shell_runs_by_node_on_dev_pushes_only_and_is_never_required`.
<!-- SECTION:NOTES:END -->
