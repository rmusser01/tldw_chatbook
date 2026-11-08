---
id: TASK-39153
title: "Lane the four files TASK-39141 made runnable: test_library_shell split by node, test_library_ingest_flow, test_console_inspector_rail_minors and test_persistent_terminal_privacy_boundary"
status: To Do
created_date: 2026-11-08 06:54
labels:
- follow-up
- ci
- tests
- library
priority: medium
references:
- Tests/UI/test_library_shell.py
- Tests/integration/test_library_ingest_flow.py
- Tests/UI/test_console_inspector_rail_minors.py
- Tests/Architecture/test_persistent_terminal_privacy_boundary.py
- scripts/ui_pr_gate_census.txt
- scripts/check_ui_pr_gate_census.py
- .github/workflows/derived-artifacts.yml
dependencies:
- TASK-39141
- TASK-39152
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 12 of the 2026-10-23 burn-down (TASK-39141's timings, round-12 branch fu/r12-ctlr). Checked on the round-12 stack 9091820bf2, 2026-11-08: none of the four files is in `scripts/ui_pr_gate_census.txt` or in any workflow.

**Timings** (from TASK-39141's notes: run.sh, `-n 4`, the Architecture file serial):
- `test_library_shell.py`, 848 nodes:
  - 992 s on base at load 45-49, with 785 nodes failing fast;
  - 2,073 s on head at load 45-160, and 5,267 s at load ~100;
  - 1,302-1,365 s for three of the files together at load 33-57.

  One UI census entry is budgeted at about 30 s, so this file has to be split by node across shards. Its stale nodes are re-pinned first (TASK-39152).
- `test_library_ingest_flow.py`: 17-27 s, and 63 s at load ~100.
- `test_console_inspector_rail_minors.py`: 12-24 s, and 51 s at load ~100 (5 passed on head).
- `test_persistent_terminal_privacy_boundary.py`: 42-121 s, and 151 s at load ~100 (8 passed on head). It runs serially, so it belongs in the Architecture guards job.

**The UI lane.** On the round-12 stack the UI lane runs 36 shards (TASK-39144). The census rules and floor are in `scripts/check_ui_pr_gate_census.py`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `test_console_inspector_rail_minors.py` and `test_library_ingest_flow.py` run on a PR lane, with three timings of each entry recorded
- [ ] #2 `test_persistent_terminal_privacy_boundary.py` runs in the Architecture guards job, with three timings of that step recorded
- [ ] #3 `test_library_shell.py` runs on PR lanes, split so that no entry exceeds the census's per-entry budget and no shard goes over its cap, with measured times recorded
- [ ] #4 None of the four files runs on the fast PR contract
<!-- AC:END -->
