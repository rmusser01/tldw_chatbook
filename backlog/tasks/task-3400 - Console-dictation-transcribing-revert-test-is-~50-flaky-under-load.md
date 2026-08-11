---
id: TASK-3400
title: Console dictation transcribing-revert test is ~50% flaky under load
status: Done
assignee: []
created_date: '2026-08-07 19:37'
updated_date: '2026-08-11 02:40'
labels:
  - tests
  - flaky
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_console_dictation_streaming.py::test_the_transcribing_indication_reverts_on_a_mid_capture_stop fails roughly half the time when run in isolation on a loaded machine. It is a wall-clock race, not a product defect: the test clicks the mic a third time and waits 4.0s for the label to reach 'Rec ●' via _wait_for_mic_label. Measured during task-3023 at 19/36 failures with the change and 17/36 without it, under matched load, so it predates that work. It passes reliably when the whole file runs (the surrounding tests warm the machinery), which is why it has gone unnoticed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The test passes deterministically when run in isolation, repeated 20 times
- [x] #2 The fix waits on the observable state transition rather than widening the timeout
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce in isolation under synthetic + real ambient CPU load.
2. Instrument the test with timing marks to see which phase is slow.
3. Identify the exact failing assertion via repeated heavy-load runs.
4. Fix by waiting on the real observable precondition instead of asserting a wall-clock-coupled assumption.
5. Two-arm verify (baseline vs fixed) under matched heavy load, then 20x isolated, then full files.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: not the 4.0s _wait_for_mic_label deadline named in the task description (that phrasing/'Rec dot' label predates the CN-01 button-label rename) but a hard assertion two lines below it: `assert button.has_class("-active")`, right after the stop-click's `_wait_for_mic_label(..., "Dictate")` resolves. Textual clears a Button's `-active` press-flash via its own fixed ~0.2s real-clock timer, started when the click landed, independent of this capture's async stop-and-transcribe unwind (asyncio.to_thread -> run_worker -> a posted ConsoleDictationEvent). Under load that unwind can take longer than 0.2s, so by the time the label reads "Dictate" the `-active` class has already cleared, and the bare assert fails outright (not a timeout).

Evidence: could not reproduce under light/moderate synthetic CPU load (up to 28 "yes" burners x 24 isolated runs: 0 failures; timing instrumentation showed all three _wait_for_mic_label calls completing in well under 200ms even then). Reproduced reliably with much heavier contention (90 "yes" burners + `nice -n 19` on the pytest process, 14-core machine): 13/14 isolated runs failed, ALL at the identical `assert button.has_class("-active")` line, `has_class` returning False every time. Two-arm comparison under matched heavy-load conditions: baseline (unmodified) 13 FAIL / 1 PASS out of 14 runs before the harness's wall-clock cap cut the loop off; fixed version 14/14 PASS in the same run, then a second heavy-load batch ran 19/19 PASS before the same cap. AC1: 20/20 consecutive isolated passes under normal conditions.

Fix (test-only, Tests/UI/test_console_dictation_streaming.py): removed the `assert button.has_class("-active")` line. The surrounding `while button.has_class("-active"): await pilot.pause(0.01)` loop already is the correct "wait on the observable transition" -- it exists to satisfy the real precondition for the next click (Textual ignores clicks while `-active` is set), and is a no-op if the class has already cleared. No change to _wait_for_mic_label or its 4.0s deadline, no widened timeout anywhere -- just deleted an assumption that isn't guaranteed and wasn't load-safe. Checked the file for the same pattern elsewhere: the other two `has_class("-active")` waits in the file (lines ~394, ~475) already loop without first asserting the flag is set, so this was the only vulnerable spot.

Verification: full-file runs after the fix -- Tests/UI/test_console_dictation_streaming.py 88 passed; Tests/UI/test_console_dictation.py (untouched, shares the _wait_for_mic_label helper) 12 passed.

Files changed: Tests/UI/test_console_dictation_streaming.py (one assertion removed, comment rewritten to explain the real reasoning).
<!-- SECTION:NOTES:END -->
