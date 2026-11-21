---
id: TASK-42143
title: "test_restore_as_requires_fresh_final_name_confirmation may time out under load on its 15 s waits"
status: Done
assignee: ['@claude']
created_date: 2026-11-20 20:06
updated_date: 2026-11-21 18:44
labels:
- follow-up
- tests
- console
- flaky
priority: low
references:
- Tests/UI/test_console_archive_recovery_flow.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the consoletests stream (round 18). `Tests/UI/test_console_archive_recovery_flow.py::test_restore_as_requires_fresh_final_name_confirmation` (line 621) failed once in a loaded local run. The file's `wait_until` (line 50) allows 15 s off Windows (60 s on Windows), polling every 30 ms, and the test makes three such waits: for the rename modal, the revised confirmation and the restored workspace. The failing run's output was not kept, so which wait timed out is not known. Two other nodes of this file are in the UI census; this one is not.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The node is run repeatedly under load (at least 10 runs, load recorded), and the failure rate and any failing wait are recorded
- [x] #2 If it fails: the cause is named and fixed, either with a bound justified by measured durations or with a wait on a product signal, and the same repeated run passes
<!-- AC:END -->

## Implementation Plan

1. Time each of the node's `wait_until` calls over at least 10 runs under load (temporary probe), recording the load, the failure rate and any wait that timed out.
2. If it fails: name the cause and fix it with a bound justified by the measured durations or with a wait on a product signal, then repeat the same run.

## Implementation Notes

No failure reproduced, so nothing changed in the test. A temporary probe in `wait_until` logged each wait's elapsed time and outcome.

- **Node alone, 10 runs at `-n 4`, load 31-55 (1-min average):** 10 of 10 passed.
- **Whole file at `-n 4`, 3 runs, load 32-53:** the node passed 3 of 3.
- **Longest wait in those 13 runs, against the 15 s bound:**
  - rename modal (line 629): 0.000 s
  - revised confirmation (line 637): 0.004 s
  - restored workspace (line 655): 0.988 s
  - the fixture's own waits: transcript 0.024 s, answerable 0.002 s, results card 0.076 s
  - `request_manual`'s confirm card: 0.259 s

None timed out, so AC #2's condition did not occur. The bound has more than 15x headroom over the slowest measured wait. A longer bound would have no measurement behind it, so it was not added.

The same whole-file runs failed three `test_native_rag_only_lifecycle_matrix_without_index` cases every time. Each timed out at 15.0 s on the fixture's results-card wait, because the turn had already failed: "Local chat dictionary backend is unavailable" / "ordinary Library catalog widened". They fail the same way on the unmodified file at dc371f0842 (3 failed, `-n 4`), and TASK-41037 tracks them. In a loaded whole-file run, a local "15 s wait timed out" in this file comes from those cases. The original failing output was not kept, so it cannot be shown that the reported failure was one of them.

The probe was removed. `Tests/UI/test_console_archive_recovery_flow.py` is unchanged.
