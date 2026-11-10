---
id: TASK-38948
title: "The 100 ms Console Send-feedback test fails under host load, and no lane runs it"
status: In Progress
assignee:
- '@claude'
created_date: 2026-11-05 21:35
updated_date: 2026-11-12 15:09
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
<!-- SECTION:NOTES:END -->
