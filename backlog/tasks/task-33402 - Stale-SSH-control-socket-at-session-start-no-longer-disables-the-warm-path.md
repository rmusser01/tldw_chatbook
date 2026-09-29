---
id: TASK-33402
title: Stale SSH control socket at session start no longer disables the warm path
status: Done
assignee:
  - '@claude'
created_date: '2026-09-28 14:13'
updated_date: '2026-09-28 16:40'
labels:
  - console
  - workspaces
dependencies:
  - TASK-33202
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A stale ControlMaster socket (for example after laptop sleep) makes the session start fail with MUX_ERROR, which is currently treated as a protocol failure and switches the binding to one-shot calls for the rest of the Console run, losing the warm path until the next run. Deferred from PR #2879's final review (ruling R14).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A mux failure at session start fails no tool call: that call runs one-shot over the restarted master
- [x] #2 The next call in the same run gets a session again; only a repeated mux start failure in the run switches the binding to one-shot for that run
- [x] #3 The binding's status is unchanged by a mux start failure
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-09-28-ssh-session-followups.md, Task 2.
<!-- SECTION:PLAN:END -->

## Implementation Notes

Registry's `acquire()` checks `_mux_failed` set first (before transport-class recording); first MUX_ERROR returns None (one-shot), second disables the key for the run. New `logger.info` line when stale socket is detected. Changes in `remote_session_registry.py` with 21 registry tests passing.
