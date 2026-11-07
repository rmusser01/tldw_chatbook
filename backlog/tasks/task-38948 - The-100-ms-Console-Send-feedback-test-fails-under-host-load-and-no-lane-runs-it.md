---
id: TASK-38948
title: "The 100 ms Console Send-feedback test fails under host load, and no lane runs it"
status: In Progress
assignee:
- '@claude'
created_date: 2026-11-05 21:35
updated_date: 2026-11-07 23:51
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
- [ ] #1 On a host loaded below its CPU count, the test passes 10 of 10 runs alone and in its file at -n 4, with the median and maximum of all three intervals recorded
- [x] #2 Under heavy load the test either still passes 10 of 10, or reports a miss together with the measured loop lag instead of failing as a product regression; the choice is recorded in the notes
- [x] #3 A temporary delay of about 150 ms before the Sending paint makes the test fail (negative control)
- [x] #4 The test runs on a PR lane, with the measured lane time recorded (about 27-49 s for both cases serially at load 50 on dev)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure the three intervals under the current load.
2. AC#2 choice: keep the 100 ms wall budgets, but when a separate probe process shows the host was contended in the interval's window, report the miss (record_property plus a warning) with the measured stall. Hold the event loop thread's CPU time for the Sending interval to the budget at any load, so added loop work still fails.
3. Negative control: a temporary ~150 ms delay at the start of `request_visible_send`, both busy (loop CPU) and blocking-sleep, with the result and load recorded for each.
4. Run 10x alone and 10x in-file at -n 4, recording the median and max of each interval. AC#1 needs load below 18; record the load seen.
5. Add both cases as UI census tail entries, with the measured serial time.
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

**Files.** `Tests/UI/test_console_received_intent_feedback.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `backlog/docs/lessons-testing-evidence.md`.

**Not met: AC#1** (10 of 10 alone and in-file on a host loaded below its CPU count). Re-run 10x alone and 10x in-file at `-n 4` once the host is under 18. On such a host the wall budget is enforced, so those runs are the strict check.
<!-- SECTION:NOTES:END -->
