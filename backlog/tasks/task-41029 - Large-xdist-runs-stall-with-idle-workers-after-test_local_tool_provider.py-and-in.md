---
id: TASK-41029
title: "Large xdist runs stall with idle workers after test_local_tool_provider.py and in test_console_archive_recovery_flow.py"
status: To Do
created_date: 2026-11-15 02:03
labels:
- follow-up
- architecture-review-2026-10-25
- tests
- ci
priority: high
references:
- Tests/Agents/test_local_tool_provider.py
- Tests/UI/test_console_archive_recovery_flow.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
In Phases 3 and 4 of the 2026-10-25 architecture review, covering runs stalled for hours instead of finishing or timing out. Each stall happened on plain dev as well:
- **Agents covering set (`-n 4`).** TASK-36950.3's covering run (Tests/Agents plus 85 files outside it that use AgentService) stopped at 88%, with two xdist workers idle in lock waits after `Tests/Agents/test_local_tool_provider.py`; the same set at its base, dev 26b2e677dd, hung at 93% (2026-11-13).
- **The 85 outside files.** A later run of just those files hung at 93% in `Tests/Chat/test_console_ssh_custody.py::test_unknown_peer_proof_keeps_*`, and its base side (26b2e677dd) sat at 87% for 8 h before SIGINT. An orphaned `control-peer.py serve` child was alive each time; TASK-40716's parked workers are a candidate cause.
- **680-file Phase 3 run.** It stalled at 86% (2026-11-12) with two workers inside `Tests/UI/test_console_archive_recovery_flow.py`, in `test_admitted_restore_delivers_app_completion_when_source_cannot_receive` and `test_receipt_refresh_refusal_keeps_native_source_safe`. The main thread was waiting on a lock, and a `private_sqlite_helper` child sat idle. The run on dev 4a9c4c1132 stalled the same way for 2 h. Run on its own, the file finishes in 2.5 min.

pytest-timeout (300 s, thread method) fired in none of these. A stalled run loses every later result, and on CI it burns the job until the job's time limit.

These stalls were not re-run end to end on ecad404bd4, because each reproduction takes hours at the current load. TASK-41028 covers a separate hang that reproduces in minutes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The task notes name a run set that reproduces each stall in the description, and that set runs to completion at `-n 4` on dev, three runs in a row
- [ ] #2 A test that blocks is reported as a pytest-timeout failure that names the test, not left as an idle worker
- [ ] #3 The task notes name the cause of each stall in the description
<!-- AC:END -->
