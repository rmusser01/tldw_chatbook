---
id: TASK-33420
title: SSH session start stays warm when the budget runs out mid bundle upload
status: Done
assignee:
  - '@claude'
created_date: '2026-09-28 17:39'
updated_date: '2026-09-29 07:53'
labels:
  - console
  - workspaces
dependencies:
  - TASK-33400
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On a cache miss, a call budget that expires while the bundle is being uploaded makes the session start fail protocol-class (`_WriteStalled`), switching the binding to one-shot calls for the rest of the run although the host is reachable. Found by the final review of the TASK-33400..33408 branch.


Same family, found while planning: once the host has answered the handshake (NEED or READY), a READY that does not arrive before the deadline is classified UNREACHABLE today, which flips a proven-reachable binding to BLOCKED.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A budget expiry during the bundle upload fails only that call and does not switch the run to one-shot calls
- [x] #2 The binding's status is unchanged by it
- [x] #3 After the host has answered the handshake, a stall or deadline during the session start never flips the binding to BLOCKED
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-09-29-ssh-session-followups-2.md, Task 1.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
After the host answers (NEED), a stalled bundle write or READY read goes to the new `RemoteSessionWorker._stalled_after_answer`: when the deadline was the call's budget + grace the call fails `OP_TIMEOUT` (status-preserving, key not disabled, next call tries a session); at the 30 s cap it is protocol-class (stuck loader, one-shot for the run). The registry no longer shares an `OP_TIMEOUT` start failure with queued callers (`remote_session_worker.py`, `remote_session_registry.py`; tests in `Tests/Tools/test_remote_session_{worker,registry}.py`, `test_remote_executor_ssh.py`).
AC #3 was narrowed from "no session start failure" to stalls and deadlines: an ssh exit after the answer (EOF, e.g. 255) keeps exit-code classification, consistent with ADR-181's natural-death rule, because a dropped channel is fresh transport evidence; noise after the answer is unchanged (only our own loader could emit it).
Trade-off: each caller queued behind a budget-expired start runs its own start with its full budget (the k-th waits up to about (k+1) x (budget + grace), bounded by `max_concurrent_calls`); passing only the remaining budget would push a nearly spent waiter into a pre-answer stall (UNREACHABLE, so BLOCKED), which is worse. Recorded in ADR-181.
<!-- SECTION:NOTES:END -->
