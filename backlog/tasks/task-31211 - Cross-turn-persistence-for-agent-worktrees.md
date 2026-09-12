---
id: TASK-31211
title: Cross-turn persistence for agent worktrees
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-03 11:45'
updated_date: '2026-09-12 20:29'
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
ADR required: yes. ADR path: backlog/decisions/155-agent-worktree-recovery.md. Reason: durable local worktree ownership/base, physical drain proof, uncertainty policy and controller-owned recovery lifetime. Execute the four reviewed slices of Docs/superpowers/plans/2026-09-12-agent-worktree-recovery.md under Docs/superpowers/specs/2026-09-12-agent-worktree-recovery-design.md. The selected product path is a Console recovery list for previous-turn work, leaving model tools scoped to current-turn handles and directing older work to the list. Verify real SQLite reopen, two-turn Git effects, consent/drift/cancellation/uncertainty and mounted wide/narrow behavior before closeout.

Execution qualification remains open: actual Git2.39.5 temp-repository tests show linked administrative paths can reopen a replaced parent despite both root pins. Only the test-isolation prerequisite has been implemented and independently reviewed (32557b80f2; four affected tests passed). Do not enable the new mutation capability or mark these tasks Done until the execution boundary is redesigned and qualified. See backlog/docs/agent-orchestration-followups-2026-09-12.md for current status.

ADR-158 (backlog/decisions/158-agent-runs-migration-order-after-worktree-qualification.md) changes only migration order: caps18→19, recovery later19→20 after execution qualification. Recheck actual schema before source edits; all existing authority and definition-policy requirements remain binding.

First execute Docs/superpowers/plans/2026-09-12-agent-worktree-interim-safety.md: refuse unqualified agent-driven creation/merge/discard without shared-tree fallback, align disclosure, and retain checkouts during automatic/failed-start cleanup. This implements the existing unsupported-authority and retention criteria; it does not complete functional recovery.
<!-- SECTION:PLAN:END -->
