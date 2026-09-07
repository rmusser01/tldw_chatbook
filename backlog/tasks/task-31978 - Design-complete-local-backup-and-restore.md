---
id: TASK-31978
title: Design complete local backup and restore
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-07 22:24'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define a user-facing, complete local backup and recovery contract covering both replacement and isolated-profile restore, with explicit data coverage, credential handling, interruption recovery, and verification evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The design records approved scope and all accepted review corrections.
- [x] #2 The specification defines archive, profile isolation, rollback, security, UI, and verification contracts without unresolved placeholders.
- [x] #3 A canonical ADR records ownership and recovery decisions and is linked from the design and task.
- [ ] #4 The written specification receives user approval before implementation planning.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: new archive contract, coordinated persistence lifecycle, credential policy, isolated profile launch, and interruption recovery across storage owners.

1. Inspect existing backup, storage, profile, security, and recovery boundaries.
2. Record user decisions and incorporate the accepted design-review corrections.
3. Write a cohesive design and canonical ADR with user flows and release evidence.
4. Self-review scope, consistency, links, and actionable recovery outcomes.
5. Obtain user review of the written specification before implementation planning.

This is a design-only task. Implementation work will be decomposed after spec approval.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Design-only work: wrote the [complete local backup and restore specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
and [ADR-126](../decisions/126-complete-local-backup-and-recovery.md). No production code or implementation plan was added.

The specification includes the approved local-data baseline, both restore destinations,
opt-in encrypted credential transfer, and all accepted review corrections. It defers
external-folder overwrite and defines independent startup recovery, enforced maintenance,
isolated launch configuration, encrypted exact rollback, preserved temporary-media
options, and product-level recovery evidence. The written draft also proposes a standard
age encryption helper and a small reopenable recovery-profile catalog for user review.

Revision 2 incorporates the second user-requested review and the explicit instruction
to apply all findings. The spec and ADR now define stable logical maintenance locks
and the legacy/external-writer boundary; restore/retire/preserve target sets; immutable
digest-bound archive input and limits before decryption/manifest parsing; restricted
SQLite schema validation/migrations; fixed bootstrap admission for custom recovery
roots with scope-aware blocking; and separate backup versus replacement downtime.
They also require early helper packaging qualification and a complete recovered-media
reference, deletion, cleanup, and subsequent-backup lifecycle. Corresponding release
evidence was added for every correction. The amendments remain documentation-only.

Revision 3 incorporates the third review's five accepted corrections: backup-source
discovery no longer gates inspection/isolated recovery from damaged configuration;
restored generations retain per-owner activation requirements across all supported
launches; encrypted rollback captures supported affected credential values and
distinguishes stored-data recovery from external authentication; archive publication
uses new-file-only atomic no-replace semantics; and a directory manifest preserves
empty structure with an explicit supported-metadata policy. Spec/ADR decisions,
review-resolution rows, and targeted release evidence were updated together.

Revision 4 incorporates all three accepted fourth-review corrections. Source inventory
is revalidated after maintenance admission closes, with renewed previews for scope or
budget changes and completeness tied to final captured coverage. Dependent indexes
and query caches cannot remain active against restored sources; ADR-030 compatibility
and reconciliation rules apply to omitted, retained, and restored projections, with
no automatic rebuild. Intentional recovered-media deletion persists a validated owner
tombstone and marked references, so subsequent complete backups preserve deletion
without confusing it with unexpected missing bytes. The spec, ADR, review mapping,
and targeted release criteria cover all three, including crash and restart scenarios.

Self-review checked coverage, activation boundaries, profile/credential isolation,
rollback versus portable redaction, cancellation, compatibility, resource limits,
interruption outcomes, and the accepted-review mapping. Local Markdown links and
placeholder/whitespace checks were verified. No runtime tests or full suite were run
because this task changes documentation only; implementation evidence is specified
without claiming it has already passed.

The CLI allocated 31759 despite a fresh sweep finding 31977 as the maximum across
425 local branch/remote refs and 68 worktrees. The newly created task was immediately
renumbered to 31978 before references were published, following the existing collision
lesson. ADR-126 was allocated from the same sweep; both allocations need merge-time
rechecking. Existing unrelated working-tree and staged changes were left alone.

Acceptance criterion 4 remains unchecked and status remains In Progress pending
the user's review of the written specification. After approval, transition through
writing-plans and create atomic implementation tasks; do not implement from this
design task alone. No new general lesson was necessary.
<!-- SECTION:NOTES:END -->
