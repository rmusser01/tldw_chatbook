---
id: TASK-32003
title: Persist per-generation recovery activation requirements
status: In Progress
assignee: []
created_date: 2026-09-07 23:59
labels:
- backup-recovery
dependencies:
- task-31978
- task-31988
- task-31991
- task-31992
- task-32001
updated_date: 2026-09-10 20:27
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Restored capabilities remain inactive across every supported launch until their own owner review completes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Restored capabilities remain inactive across every supported launch until their own owner review completes.
- [ ] #2 Missing/corrupt activation records and imported approvals cannot grant execution authority.
- [ ] #3 Safe local inspection works and one owner approval never activates unrelated automation or queued work.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute original component04 Task20 and ADR126: first persist private per-generation/per-owner requirements independently of journal/report/catalog, with idempotent require, owner-specific local approval and missing/corrupt/mismatched refusal. Then bind restored generations before publication fences clear; wire every supported startup/composition and owner-specific existing review path so local inspection remains available without automatic execution, reconnection or replay. Verify fresh-process/relaunch/headless/corruption and one-owner isolation with targeted tests; review, lint, Bandit and scoped commits. This initial independent slice implements durable store only; Task18 publication integration and actual consumer gates remain required before task completion.
<!-- SECTION:PLAN:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-04-restore-recovery.md#task-20)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Initial original Task20 durable-store slice implemented in activation.py + test_activation.py under ADR126. Original one-owner approval regression failed behaviorally then passed; final22focused cases pass2.03s (/private/tmp/chatbook-activation-store-final.log). Covers private strict requirements/owner approval records, restart/realchildexit, missing/corrupt/linked/public/different-generation denial, no read-time creation, unchanged idempotent requirements and failed durable retries. Review P2 retry-barrier gap fixed by pinned existing-record identity/validation plus file fsync/F_FULLFSYNC and directory/ancestor flush before returning success; independent finalreview approved /private/tmp/chatbook-activation-store-review.md. FullRuff/formatclean, productionBandit0. No global locks/new journal; independent immutable per-owner files avoid lost updates. Startup association/consumer gating/owner reconciliation remain required before taskDone; no ACchecked or wholefeatureclaim. Integration source map /private/tmp/chatbook-task20-activation-source-map.md.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
