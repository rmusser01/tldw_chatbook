---
id: TASK-31211
title: Cross-turn persistence for agent worktrees
status: Done
assignee:
  - '@codex'
created_date: '2026-09-03 11:45'
updated_date: '2026-09-13 04:30'
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
- [x] #1 A worktree with unmerged changes from a previous turn can be merged or discarded through a user-confirmed path (tool or UI), not just by hand in a terminal.
- [x] #2 Liveness/ownership checks remain DB-backed (a still-running survivor's worktree is never offered for merge or deletion).
- [x] #3 The same-turn-only sentences in the two tool schemas are updated to match the new behavior.
- [x] #4 Durable records preserve original base and exact conversation/binding/source ownership; legacy unknown-base and scratch work is never automatically adopted or deleted.
- [x] #5 Only terminal work with persisted positive physical-owner drain can be acted on; held or cleanup-uncertain writers and ambiguous mutation results remain preserved and non-actionable across restart.
- [x] #6 Each recovery action captures current writable authority and revalidates source and binding after exact consent; a transactional claim prevents double execution and no uncertain effect is automatically replayed.
- [x] #7 Recovery owns an independent retained cancellation signal; session navigation parks/remounts the card and session close does not affect unrelated turns.
- [x] #8 Discard removes confirmed changes while retaining the detached baseline checkout with an explicit cleanup-pending receipt; automatic root deletion is disabled and unsupported platforms refuse.
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented durable earlier-turn recovery through Console: Recover agent work. Metadata-only pages are scoped to the exact owning conversation and currently selected writable named repository. Each Apply/Merge/Discard uses the shared confirmed engine, fresh post-consent authority/source checks and transactional claims. Tool descriptions distinguish current-turn membership from earlier-turn Console recovery.

Schema20 preserves original base and exact conversation/binding/source/child/execution ownership before child launch. Only terminal work with persisted positive physical drain is actionable; held/cleanup-unproven writers, uncertain results and legacy unknown-base work stay protected across restart. Confirmed discard removes changes and the exact branch while retaining and disclosing a detached baseline checkout. Ordinary Git retains the documented concurrent external metadata-replacement limitation.

Manual recovery owns its Event, worker DB connection and existing capacity owner. Navigation parks its exact card, session close cancels only accepted owning-session work, and waiter cancellation cannot release an admitted physical worker. Final startup fixes cover partial Git-reader launch and rejected/queued/already-entered manual executor submission.

Independent task and final integration/correction reviews are complete through e135a085f2. Real Git/SQLite creation, migration/reopen, owner drain, denial/drift/claim/merge/discard and mounted recovery evidence are recorded in Docs/superpowers/reviews/2026-09-12-agent-worktree-restoration.md. Final correction selections passed 88 affected/capacity cases and2 actual card/Git flows, following11 real RED failures. Counts from earlier selections overlap.

Verification is targeted, not a full-suite or live-provider claim. The inherited Requests warning remains; UI-ready passes at 973/973 with zero headroom. Existing ChatScreen size/no-growth guards (including the disclosed 21-line/3-method addition) and eight stale historical diagnostic-label expectations remain failures. The current diagnostic inventory/sink guard passes, and changed-file static comparisons add no diagnostic identities with edited-range formatting passing.

ADR check: existing backlog/decisions/155-agent-worktree-recovery.md and 158-agent-runs-migration-order-after-worktree-qualification.md govern these delivered contracts; no additional ADR for the failure-path fixes. User guide and lessons are updated. All code is committed locally.
<!-- SECTION:NOTES:END -->
