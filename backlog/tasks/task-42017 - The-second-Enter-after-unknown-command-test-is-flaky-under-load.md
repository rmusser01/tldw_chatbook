---
id: TASK-42017
title: "The second-Enter-after-unknown-command test is flaky under load"
status: Done
assignee: ['@claude']
created_date: 2026-11-19 19:39
updated_date: 2026-11-20 16:38
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/UI/test_console_send_admission_off_pump.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41904 (round 17). `test_a_second_enter_after_an_unknown_command_hint_sends_it_as_text` failed about 2 of 4 serial runs at base, at load 42-55.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The cause is named and the test passes 10 of 10 at `-n 4` with the load recorded
<!-- AC:END -->

## Implementation Plan

1. Reproduce the failure under load at `-n 4` and read the failing assertion and the run's state (what was sent, the composer, the flight).
2. Name the cause from that evidence (temporary instrumentation, reverted).
3. Fix the test (or the product, if the evidence shows a product race), with the smallest change at the cause.
4. Show 10 of 10 at `-n 4` with the load recorded.

## Implementation Notes

- **Cause.** The test's last wait, `_settled`, returns once the screen's send flight is empty (`flight.tasks` and `flight.deferred`), then pauses 1 s. The flight ends when the replayed send is handed to the runtime, not when the runtime accepts the turn; the user row and the composer's clear come with that durable acceptance (TASK-38690's lesson covers the composer half). On a loaded host the turn took longer than the 1 s pause, so the assertion saw nothing sent. The press was never dropped.
- **Evidence.** Base, 8 concurrent serial runs of the test alone (`-p no:xdist`, load 21-25): 8 of 8 failed with `assert [] == ['/nope x']`. Temporary spies on `send_acknowledgement._replay` / `_start_send` (reverted), 4 concurrent serial runs: in every run the replay scheduled the held press and `_start_send` ran within 3 ms of the tail release, and `_settled`'s wait returned at 0.06-0.11 s. In the run that failed, the assertion at 1.062 s saw `[]` and the composer still holding `/nope x`; both were there at 1.25 s.
- **Fix (test only, `Tests/UI/test_console_send_admission_off_pump.py`).** After `_settled`, wait for the turn to finish (`_first_turn_done`: reply painted, no live activity) before asserting exactly `[UNKNOWN]` and an empty composer. A wait that times out now adds what was sent and the draft as notes. Only this test in the file asserted a positive send after `_settled` alone; the others wait for `stream_calls` or `_first_turn_done` first, or assert that nothing was sent.
- **Negative control.** With `_defer` temporarily dropping a repeat press (the defect the test was written for), the test failed: `timed out waiting for ...`, notes `sent []`, `draft '/nope x'`. Reverted.
- **Runs after the fix.** The same 8-concurrent serial setting: 8 of 8 passed (load 25.0 -> 32.9). AC run, 10 runs at `-n 4`, two at a time: 10 of 10 passed, 17.1-21.7 s each; 1-minute load averages at the start/end of each pair: 25.5/28.5, 28.5/27.6, 27.6/25.4, 25.4/32.8, 32.8/31.2. The file at `-n 4` at base had 8 other failures (the `HeldMcpRead` tests, matching TASK-41900's "admission off pump 8"); not touched here.
- No lane change: the test is not in a PR lane (only a sibling node is in the UI census).
