---
id: TASK-38948
title: "The 100 ms Console Send-feedback test fails under host load, and no lane runs it"
status: To Do
created_date: 2026-11-05 21:35
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
- [ ] #2 Under heavy load the test either still passes 10 of 10, or reports a miss together with the measured loop lag instead of failing as a product regression; the choice is recorded in the notes
- [ ] #3 A temporary delay of about 150 ms before the Sending paint makes the test fail (negative control)
- [ ] #4 The test runs on a PR lane, with the measured lane time recorded (about 27-49 s for both cases serially at load 50 on dev)
<!-- AC:END -->
