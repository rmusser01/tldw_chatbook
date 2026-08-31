---
id: TASK-26035
title: Filesystem checkpoints with per-turn baseline and rollback
status: To Do
assignee: []
created_date: '2026-08-31 15:47'
labels:
  - agents
  - workspaces
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
There is no way to undo a turn's file changes as a unit. Verified on origin/dev: change review offers per-file revert and undo-all (UI/Screens/change_review_screen.py:1347-1348) over turn-scoped snapshots (Workspaces/change_turn_tracker.py:172,315,396 with rows in DB/AgentRuns_DB.py:321-348), but there is no time-indexed baseline a user can roll back to - and no way to ask what the tree looked like three turns ago. Hermes snapshots the working tree before file-mutating tools each turn into a shadow git store, with list, diff, session-diff and a safe restore plan. Chatbook already has the shadow-repo mechanism this would build on.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A restorable checkpoint is captured per turn before file-mutating tool calls, reusing the existing shadow-repo snapshot mechanism
- [ ] #2 Checkpoints are listable with enough context to choose one (turn, time, what changed)
- [ ] #3 Restoring produces a preview of what will change before it is applied and requires confirmation
- [ ] #4 Restore refuses rather than proceeding when the working tree has uncommitted changes it would destroy, unless the user explicitly overrides
- [ ] #5 Checkpoints are pruned by a documented retention and size bound
- [ ] #6 Checkpoint storage stays out of the user's own git history - it never creates commits, branches or stashes in the project repo
- [ ] #7 Files outside the allowed roots are never captured or restored
<!-- AC:END -->
