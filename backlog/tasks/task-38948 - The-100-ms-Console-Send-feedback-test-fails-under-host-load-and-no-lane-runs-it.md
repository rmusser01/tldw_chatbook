---
id: TASK-38948
title: "The 100 ms Console Send-feedback test fails under host load, and no lane runs it"
status: In Progress
assignee:
- '@claude'
created_date: 2026-11-05 21:35
updated_date: 2026-11-07 07:56
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
Test-only change. Base 8b762b52fa, 2026-11-07. The host's load average never fell below its 18 CPUs during this work (it ranged 29-190), so AC#1 is not met and the task stays In Progress.

**AC#2 choice: report under heavy load.** `_assert_send_budget` checks each of the three intervals (Sending frame, input mutation, input frame) against 100 ms of wall time. A miss fails while the 1-minute load average is under `os.cpu_count()`. At or over it, the miss is reported instead, as a `<label>_host_load_miss` property and a `UserWarning` giving the load and the loop's lag. The lag is the event loop thread's CPU time in the interval and the time it spent off the CPU. The loop's CPU is stamped on the loop thread at both ends: `action_cpu`/`feedback_frame_cpu` for Sending; `key_cpu` (when the loop starts handling the typed key) and `mutated_cpu`/`frame_cpu` for input, through property setters on `_TimedInputFrame`.

Two stricter rules were tried first and dropped, with evidence (lesson in `lessons-testing-evidence.md`):
- Subtracting a separate probe process's scheduling stalls still failed at load 104-130 (406 ms, 181 ms of probe stall), and its 10%-stall contention signal missed a 181 ms miss at load 48 (9 ms of probe stall).
- Holding the loop's CPU to 100 ms at any load false-failed once at load 31 (258 ms of CPU, no product change).

**Negative control (AC#3).** A temporary edit at the start of `send_acknowledgement.request_visible_send`, run with the test's load reading forced to 0 (an unloaded host's verdict):
- with no delay: 2 passed;
- a 150 ms busy-wait: both routes failed (228/233 ms; loop CPU 136/128 ms);
- a blocking 150 ms sleep: both routes failed (215/202 ms; off the CPU 163/161 ms).

At the real load (46-63) the same delays were reported, not failed. That is the AC#2 trade-off: a loaded host only reports.

**Runs at `-n 4`** (AC#1's measurements, all at load over 18):
- Alone, 10 runs at load 47-65: 10 of 10 passed, 3 misses reported. Over 20 cases, median/max: Sending 55.1/98.1 ms (loop CPU 33.8/47.6); input mutation 9.9/109.8 ms; input frame 17.8/113.2 ms.
- In its file, 10 runs at load 33-56: 10 of 10 passed (17 passed each), 7 misses reported. Over 20 cases, median/max: Sending 55.5/133.9 ms (loop CPU 35.2/48.7); input mutation 4.7/102.6 ms; input frame 14.2/118.8 ms.
- Before: on base the filing measured 19 of 20 failing alone at load 50-56.

**Lane (AC#4).** UI census tail, one entry per route (`[enter]`, `[send-button]`). Serially in the lane's invocation at load about 130: 9.1 and 6.2 s of call, 25 s of session for both.

**Files.** `Tests/UI/test_console_received_intent_feedback.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `backlog/docs/lessons-testing-evidence.md`.

**Not met: AC#1** (10 of 10 alone and in-file on a host loaded below its CPU count). Load stayed at 29-190 throughout. Re-run `fb48`-style once the host is quiet: 10x alone and 10x in-file at `-n 4`.
<!-- SECTION:NOTES:END -->
