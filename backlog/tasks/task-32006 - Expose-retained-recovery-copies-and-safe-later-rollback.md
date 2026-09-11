---
id: TASK-32006
title: Expose retained recovery copies and safe later rollback
status: In Progress
assignee: []
created_date: 2026-09-08 00:01
labels:
- backup-recovery
dependencies:
- task-31978
- task-32001
- task-32004
- task-32005
updated_date: 2026-09-11 06:25
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Users can inspect retained recovery copies and roll back later only after intervening changes are preserved.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Users can inspect retained recovery copies and roll back later only after intervening changes are preserved.
- [ ] #2 Pending evidence and held artifacts cannot be deleted or swept automatically.
- [ ] #3 App-owned operation state survives navigation and closes without abandoning unsafe publication.
<!-- AC:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-04-restore-recovery.md#task-23)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Root beginning originalTask23 retained-copy list/hold/explicitdelete and servicecomposition. Reuse existing exact typedjournal/ciphertext proof and journalnative lock forreadholds/exclusivedeletion; noagebasedsweep,catalogauthority,unknownfilecleanup,newlockingframework. Firstboundedfiles new recovery_copies.py/test_recovery_copies.py; laterrollback/service uses existingengine and NEWencrypted safetycopy beforepostrestore replacement. DependentTask22 credential work runsseparately; no sharedfileeditsbyroot. Plan stages: actualretainedcopylist/hold/deletebehavior tests; minimal nativecoordination; laterrollback/facadecomposition; focusednative tests/static/Bandit/independentreview.
First2-file retainedcopyunit independently APPROVED byruntime_final_capture, /private/tmp/recovery-copies-independent-review.md, patchSHAc511e1f34189401f87e814559df0ba46888b02ce9b106930731ba220a084437a. Actual encryptedciphertext nativefileflock protects sameprocess/freshchild readholds againstexplicitdelete; terminal/pending/corrupt/changed/symlink/exactunlink tests pass2in21.19s, freshstartupidentityafterdelete1pass4.39s; earlierpredicate+actualfirstcohort5pass10.23s. Ruff/formatclean, productionBandit0. Service/laterrollback must holdactualciphertext through acquisition/unlock/nativeworker cancellation; rawarchive_reader notretroactivelyqualified. Rootcontinuing originalfacade/laterrollback; taskremainsInProgress.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->