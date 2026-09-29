---
id: TASK-33421
title: A call whose SSH session closes before its request is written re-acquires instead of failing
status: Done
assignee:
  - '@claude'
created_date: '2026-09-28 17:39'
updated_date: '2026-09-29 07:53'
labels:
  - console
  - workspaces
dependencies:
  - TASK-33401
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
If the laptop closes a healthy session (idle reap, run end, app exit) after a call registered its request but before the REQUEST bytes were written, the call returns REMOTE_OP_FAILED although nothing was sent; TASK-33401 only covered calls not yet registered. Found by the final review of the TASK-33400..33408 branch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A call whose request bytes never reached the session re-acquires (fresh session, or one-shot after run end or app exit)
- [x] #2 A call whose request was written is never retried
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-09-29-ssh-session-followups-2.md, Task 2.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`RemoteSessionWorker._write` now raises `_StdinClosed` (a `BrokenPipeError` subclass) only from its pre-check, when stdin was already closed and nothing was written. `call()` wraps only the REQUEST write: `_StdinClosed` on a session `close()` retired becomes `SessionClosed`, which the executor's existing re-acquire loop turns into a fresh session, or one-shot after run end or app exit.
A CANCEL write on a closed stdin keeps the existing classified path, and a mid-write EPIPE stays a plain `BrokenPipeError`, so a written request is never retried. Only `remote_session_worker.py` changed; tests: `test_close_between_register_and_write_raises_session_closed`, `test_cancel_write_on_closed_stdin_is_not_session_closed` (worker) and `test_close_between_register_and_write_reacquires` (executor).
<!-- SECTION:NOTES:END -->
