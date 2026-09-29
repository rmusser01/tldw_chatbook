---
id: TASK-33400
title: SSH session start against a dead host fails fast for every queued call
status: Done
assignee:
  - '@claude'
created_date: '2026-09-28 14:13'
updated_date: '2026-09-28 16:30'
labels:
  - console
  - workspaces
dependencies:
  - TASK-33202
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When a host dies or blackholes, concurrent tool calls queued behind one failing session start each start their own session in turn, so the last caller waits roughly k times the connect or handshake timeout, while one-shot calls fail together. The handshake also ignores the call's time budget (fixed 30 s). Deferred from PR #2879's final review (ruling R14).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Callers queued behind a session start that fails transport-class all get that failure without starting another session
- [x] #2 A call arriving after such a failure still tries a fresh session (no sticky failure)
- [x] #3 A session handshake never outlives the call's budget plus grace, and a stalled handshake is classified exactly like the one-shot path
- [x] #4 Any session start failure leaves no ssh process running and no open pipe on the laptop
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-09-28-ssh-session-followups.md, Task 1.
<!-- SECTION:PLAN:END -->

## Implementation Notes

Registry fan-out is blocked via `_start_failures` dict recording transport errors before queued callers re-acquire (entry-time gate). Worker handshake is capped by `budget + grace` (30 s max). Cleanup uses new `_abandon_start()` to reap ssh and close pipes on any start failure. All changes in `remote_session_registry.py`, `remote_session_worker.py`, `remote_workspace_executor.py` with TDD coverage (19 registry tests, 36 worker tests pass).
