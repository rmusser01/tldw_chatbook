# ADR-158: AgentRuns migration order after worktree qualification

Status: Accepted
Date: 2026-09-12
Tasks: TASK-13154.7, TASK-31210, TASK-31211
Related: ADR-155, ADR-157
Supersedes: ADR-157 migration sequencing only; its definition policy is incorporated unchanged.

## Decision

Implement per-definition child wall-time caps independently of worktree recovery. AgentRuns currently has schema18. The cap column uses the next migration,18→19. Worktree persistence is planned after that migration,19→20, and must recheck the actual schema when its execution design is qualified. Do not ship a placeholder recovery migration or reserve an unused version merely to preserve the original task order.

The complete definition policy in ADR-157 remains binding: optional finite positive caps, unchanged uncapped identity/defaults, frozen roster admission, minimum intersection after legacy floors, retained capped-lineage ceiling on continuation, and existing cooperative cancellation/physical-owner semantics. This decision changes migration ordering, not those behaviors. Canonical Settings ownership and preset work still precede its cap field integration.

ADR-155's authority, exact confirmation, physical drain and uncertainty requirements remain binding. Its new mutations stay unavailable until the complete Git/filesystem boundary has real evidence. The execution qualifier found that nested source/child root pins alone leave linked administrative paths able to reopen a replaced repository. No alternative Git backend is approved here. Recovery tasks remain incomplete.

## Context and alternatives

The original plan reserved schema19 for recovery and20 for definition caps. Real temporary-repository races blocked the first recovery implementation before product edits; definition caps neither read nor depend on recovery rows. Keeping the reservation would turn a planning order into an artificial delivery dependency. Shipping an empty migration adds no value. Implementing speculative recovery storage before the backend is chosen risks durable fields whose meaning immediately changes.

Renumbering the unimplemented cap migration is the smallest correction. Existing schema18 data must survive a real18→19 upgrade and repeated reopen; the later recovery migration must preserve cap values. If another integrated migration takes the next version first, reconcile actual schema before writing code. Never rewrite an already-shipped migration.

## Consequences and evidence

Caps can be implemented, verified and reviewed while recovery's execution boundary remains unresolved. Both source plans/specs and Backlog plans name the new sequence. No current database or product source is changed by this decision. Cost if the task order changes again: renumber the still-unimplemented migration and update its qualification fixtures; never silently reuse a shipped version.

Required cap evidence remains actual SQLite migration/data/identity tests and gated runtime continuation tests, followed by mounted Settings tests. Required recovery evidence remains unchanged and must include preservation of the cap column/data once its migration is implemented.

[Cap plan](../../Docs/superpowers/plans/2026-09-12-definition-wall-cap.md) · [Recovery plan](../../Docs/superpowers/plans/2026-09-12-agent-worktree-recovery.md)
