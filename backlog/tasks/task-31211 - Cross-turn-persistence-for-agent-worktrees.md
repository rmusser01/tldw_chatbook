---
id: TASK-31211
title: Cross-turn persistence for agent worktrees
status: To Do
assignee: []
created_date: '2026-09-03 11:45'
labels:
  - agents
dependencies: []
priority: medium
---

## Description

Phase 2 of TASK-28238 made worktree merge-back same-turn-only: `AgentService._agent_worktrees` resets each `run_turn`, so a worktree not merged or discarded in the turn that spawned it becomes unreachable through tools (the tool schemas state this honestly). The end-of-turn GC sweep prunes only clean/terminal leftovers; dirty ones stay on disk for manual recovery. Decide and build the durable story: either persist worktree handles across turns (rehydrate the map from `git worktree list` + AgentRuns_DB at turn start, so a later turn can still merge a survivor's work) or a Console housekeeping surface that lists leftover `agent/*` worktrees and offers merge/discard with the same confirm gate.

Constraint carried from the phase-2 rulings: implicit deletion of unmerged work is never acceptable — any destructive path needs the per-call user confirm.

## Acceptance Criteria

- [ ] A worktree with unmerged changes from a previous turn can be merged or discarded through a user-confirmed path (tool or UI), not just by hand in a terminal.
- [ ] Liveness/ownership checks remain DB-backed (a still-running survivor's worktree is never offered for merge or deletion).
- [ ] The same-turn-only sentences in the two tool schemas are updated to match the new behavior.
