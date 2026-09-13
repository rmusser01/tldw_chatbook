---
id: TASK-31211
title: Cross-turn persistence for agent worktrees
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-03 11:45'
updated_date: '2026-09-13 02:12'
labels:
  - agents
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 2 of TASK-28238 made worktree merge-back same-turn-only: `AgentService._agent_worktrees` resets each `run_turn`, so a worktree not merged or discarded in the turn that spawned it becomes unreachable through tools (the tool schemas state this honestly). The end-of-turn GC sweep prunes only clean/terminal leftovers; dirty ones stay on disk for manual recovery. Decide and build the durable story: either persist worktree handles across turns (rehydrate the map from `git worktree list` + AgentRuns_DB at turn start, so a later turn can still merge a survivor's work) or a Console housekeeping surface that lists leftover `agent/*` worktrees and offers merge/discard with the same confirm gate.

Constraint carried from the phase-2 rulings: implicit deletion of unmerged work is never acceptable — any destructive path needs the per-call user confirm.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A worktree with unmerged changes from a previous turn can be merged or discarded through a user-confirmed path (tool or UI), not just by hand in a terminal.
- [ ] #2 Liveness/ownership checks remain DB-backed (a still-running survivor's worktree is never offered for merge or deletion).
- [ ] #3 The same-turn-only sentences in the two tool schemas are updated to match the new behavior.
- [ ] #4 Durable records preserve original base and exact conversation/binding/source ownership; legacy unknown-base and scratch work is never automatically adopted or deleted.
- [ ] #5 Only terminal work with persisted positive physical-owner drain can be acted on; held or cleanup-uncertain writers and ambiguous mutation results remain preserved and non-actionable across restart.
- [ ] #6 Each recovery action captures current writable authority and revalidates source and binding after exact consent; a transactional claim prevents double execution and no uncertain effect is automatically replayed.
- [ ] #7 Recovery owns an independent retained cancellation signal; session navigation parks/remounts the card and session close does not affect unrelated turns.
- [ ] #8 Discard removes confirmed changes while retaining the detached baseline checkout with an explicit cleanup-pending receipt; automatic root deletion is disabled and unsupported platforms refuse.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes, existing amended decision.
ADR path: backlog/decisions/155-agent-worktree-recovery.md; backlog/decisions/158-agent-runs-migration-order-after-worktree-qualification.md.
Reason: ordinary local Git with exact selected writable authority, durable original-base/ownership, positive physical completion, and user-confirmed recovery.
Spec: Docs/superpowers/specs/2026-09-12-agent-worktree-restoration-design.md.
1. Creation restoration (Docs/superpowers/plans/2026-09-12-agent-worktree-creation-restoration.md): implemented and independently reviewed at a11db916ff/d63d4f2a19.
2. Durable storage and completion callbacks (Docs/superpowers/plans/2026-09-12-agent-worktree-records-and-drain.md): implemented and reviewed at4756f88793/94f7110db1; schema20 retains caps.
3. Connect actual child ownership (Docs/superpowers/plans/2026-09-12-agent-worktree-ownership-integration.md), then confirmed operations (Docs/superpowers/plans/2026-09-12-agent-worktree-confirmed-operations.md).
4. Wire visible confirmation and retained manual recovery (Docs/superpowers/plans/2026-09-12-agent-worktree-console-recovery.md), verify real Git/SQLite and mounted wide/narrow UI, independently review and close only when all AC are met.
Automatic deletion remains disabled. Command-boundary checks do not promise atomic protection against concurrent external Git metadata replacement. The former qualified-backend blocker is superseded by the user-approved scope. Current review/evidence/rulings: Docs/superpowers/reviews/2026-09-12-agent-worktree-restoration.md.
<!-- SECTION:PLAN:END -->
