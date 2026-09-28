---
id: TASK-33401
title: SSH idle-session reaping never delays or fails an unrelated call
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
Idle-session reaping closes other runs' and bindings' sessions synchronously on whichever tool call happens to run it, under that host's concurrency slot, and each close can take seconds against a wedged host. The reaper can also close a session another call was just handed, which surfaces a spurious tool error. Deferred from PR #2879's final review (ruling R14).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Reaping idle sessions never blocks the calling tool call on closing them
- [ ] #2 A call whose session is closed by the laptop (idle reap, run end, app exit) before its request is sent is not failed: it gets a fresh session, or the one-shot path when the run or app has ended
<!-- AC:END -->
