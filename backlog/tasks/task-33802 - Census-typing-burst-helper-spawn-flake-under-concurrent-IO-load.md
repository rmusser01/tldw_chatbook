---
id: TASK-33802
title: Census typing-burst helper-spawn flake under concurrent I/O load
status: To Do
assignee: []
created_date: '2026-10-02 18:25'
labels:
  - perf
  - testing
  - flaky
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Console storage-unit census (Tests/Performance/test_console_keystroke_work_census.py::test_console_storage_units_stay_within_their_ratchets) sometimes fails with "typing (whole burst) helper_spawns: 1 > ceiling 0" (storage_admissions 3 in the same burst): some private-SQLite open lands inside the 40-keystroke window, which holds the credential-poll and draft-spend timers still.

Evidence gathered 2026-10-02 while fixing TASK-33801:
- 7 of 24 runs at dev 52620c3a08 (and TASK-33801's branch on it) failed this way, every one while a parallel pytest -n 6 sweep was running on the same machine.
- 0 of 46 runs at earlier commits (92a95170a5 through 30ca4552b3), measured without a concurrent sweep.
- Interleaved A/B under four CPU burners (yes > /dev/null): 0/8 at 30ca4552b3, 0/8 at 52620c3a08. 0/10 with per-spawn stack instrumentation.
So the cause is load-dependent and not shown to be any one commit; CPU load alone does not reproduce it, concurrent I/O-heavy test sweeps did.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The code path of the helper spawn billed to the typing burst is identified from a captured stack (record the caller and thread of every HelperLease.start while counting["phase"] is the burst, under a concurrent pytest -n 6 sweep)
- [ ] #2 Either that work is moved off the keystroke path, or the census holds it still for the burst like the credential poll and draft-spend refresh, with the reason recorded; no ceiling is raised
<!-- AC:END -->
