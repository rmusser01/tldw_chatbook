---
id: TASK-3400
title: Console dictation transcribing-revert test is ~50% flaky under load
status: To Do
assignee: []
created_date: '2026-08-07 19:37'
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
- [ ] #1 The test passes deterministically when run in isolation, repeated 20 times
- [ ] #2 The fix waits on the observable state transition rather than widening the timeout
<!-- AC:END -->
