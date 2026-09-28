---
id: TASK-33404
title: Host fork failure fails one SSH session request, not the whole session
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
When the host cannot fork (process limit, EAGAIN) or open a pipe for a request, the fork-server exits, ending every in-flight request of the session and, through the one-shot taxonomy, possibly flipping the binding to BLOCKED. Deferred from PR #2879's final review (ruling R14).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A request the host cannot start fails alone with a typed, status-preserving error; other requests on the session complete
- [ ] #2 The session keeps serving new requests after such a failure
- [ ] #3 The regenerated worker bundle matches a fresh build
<!-- AC:END -->
