---
id: TASK-33404
title: Host fork failure fails one SSH session request, not the whole session
status: Done
assignee:
  - '@claude'
created_date: '2026-09-28 14:13'
updated_date: '2026-09-28 16:44'
labels:
  - console
  - workspaces
dependencies:
  - TASK-33202
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When the host cannot fork (process limit, EAGAIN) or open a pipe for a request, the fork-server exits, ending every in-flight request of the session and, through the one-shot taxonomy, possibly flipping the binding to BLOCKED. Deferred from PR #2879's final review (ruling R14).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A request the host cannot start fails alone with a typed, status-preserving error; other requests on the session complete
- [x] #2 The session keeps serving new requests after such a failure
- [x] #3 The regenerated worker bundle matches a fresh build
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-09-28-ssh-session-followups.md, Task 5.
<!-- SECTION:PLAN:END -->

## Implementation Notes

Added `HOST_SPAWN_FAILED = 71` constant in `remote_session_frames.py`. Serve's `start()` catches fork/pipe OSError and appends STATUS(71) without exiting. Worker's `_result()` maps non-admitted STATUS(71) to REMOTE_OP_FAILED (status-preserving). Bundle regenerated and verified with `--check`. All 108 tests passing.
