---
id: TASK-41135
title: "Two test hangs no timeout ends: the 239-file non-UI set at -n 4 stalls with all workers idle, and test_original_stop_hook_holds_same_ordinary_tree runs over an hour"
status: To Do
created_date: 2026-11-15 02:20
labels:
- follow-up
- tests
- console
- flaky
priority: medium
references:
- Tests/Chat/test_console_ordinary_tree.py
- pyproject.toml
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** A hang that `--timeout` does not end blocks a whole worker's queue (about 90 tests) and forces runs to be interrupted, so covering runs lose results without saying so.

**Evidence.**
- TASK-40300's notes (endpoint branch): 239 non-UI files at `-n 4` "hung at 91% (all four workers idle for 18 min, a pre-existing hang)", the same at 95% on the round-1 tree.
- TASK-40321's notes: `Tests/Chat/test_console_ordinary_tree.py::test_original_stop_hook_holds_same_ordinary_tree` "made no progress for over an hour while the others drained; neither `--timeout=300` method ended it". Both arms were interrupted with SIGINT.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The stop-hook node's hang is reproduced and its cause recorded (a stack dump of the stuck worker)
- [ ] #2 The node completes or fails within its timeout, and the cause is fixed in the product or the test
- [ ] #3 The 239-file set runs to completion at -n 4, or the node that stalls it is identified and handled the same way
<!-- AC:END -->
