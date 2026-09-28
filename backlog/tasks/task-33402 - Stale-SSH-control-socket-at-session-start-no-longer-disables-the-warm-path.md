---
id: TASK-33402
title: Stale SSH control socket at session start no longer disables the warm path
status: To Do
assignee: []
created_date: '2026-09-28 20:30'
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
- [ ] #1 A mux failure at session start fails no tool call: that call runs one-shot over the restarted master
- [ ] #2 The next call in the same run gets a session again; only a repeated mux start failure in the run switches the binding to one-shot for that run
- [ ] #3 The binding's status is unchanged by a mux start failure
<!-- AC:END -->
