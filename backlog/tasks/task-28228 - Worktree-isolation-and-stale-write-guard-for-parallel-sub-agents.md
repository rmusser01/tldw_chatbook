---
id: TASK-28228
title: Worktree isolation and stale-write guard for parallel sub-agents
status: To Do
assignee: []
created_date: '2026-09-02 06:39'
labels:
  - agents
  - safety
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred rows C3+C7 combined, promoted by TASK-26041: their shared precondition shipped — fleet_coordinator runs concurrent sub-agent children against ONE shared working tree, so concurrent-edit races are now a live risk, not a future one. Two complementary halves: (a) optional per-child git-worktree isolation like hermes; (b) a stale-write guard on the fs tools (read-version tokens; a write whose base changed since read refuses with a diff) — optimistic locking exists only for notes today (Tools/note_management_tools.py:318).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A fleet child can opt into an isolated worktree; its changes merge back explicitly, never silently
- [ ] #2 fs_write/fs_edit refuse when the target changed since the tool last read it, naming the conflict
- [ ] #3 Single-agent behavior is unchanged by default
- [ ] #4 The refusal path is exercised by a test that races two writers
<!-- AC:END -->
