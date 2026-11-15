---
id: TASK-41140
title: "Console Performance reds left by TASK-40323: held-pointer [skill], managed received custody over its 10 s bound, and three genuine-capture vars pins"
status: To Do
created_date: 2026-11-15 02:20
labels:
- follow-up
- console
- tests
- performance
- dev-red
priority: medium
references:
- Tests/Performance/test_app_wiring_held_pointer_currentness.py
- Tests/Performance/test_console_managed_received_acceptance.py
- Tests/Performance/test_console_skill_discovery_genuine_capture.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** These subprocess pins guard where skill reads and received-send custody run. Red, they cannot catch a regression in either.

**Evidence.**
- `test_app_wiring_held_pointer_currentness.py[skill]`: red at its script `:138`, `source.runtime_reads` still holds a read right after the trust task returns (empty 0.5 s later) (TASK-40323's notes).
- `test_console_managed_received_acceptance.py`: after the 077e5bfa09 and 064d5cb014 re-pins it reaches its end and fails its 10 s received-custody bound, 11.0 s at load 65-70; a trace-settlement thread was opening a ChaChaNotes connection through the private SQLite helper process (TASK-40323's notes).
- `test_console_skill_discovery_genuine_capture.py`: through run.sh at the stack top, 3 failed, 6 passed; the round-14 report names the same stale `vars(...)` globals pin that 064d5cb014 moved (`run_preparation_read` now lives in `Backup_Recovery.runtime_producer_lifetime`).

The stock skill-trust `[send_enter]` / `[send_button]` cases ("Cold Send did not reach original configuration SQL") are TASK-34563.35's behaviour, not this task's.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The genuine-capture nodes are re-pinned to the module that owns the read since 064d5cb014, red then green
- [ ] #2 The held-pointer [skill] read lifetime is classified (test timing or a read that outlives its trust task) and fixed accordingly, red first
- [ ] #3 The managed received-custody path is measured; the trace-settlement connection is moved off the custody window or the bound is justified with measurements
<!-- AC:END -->
