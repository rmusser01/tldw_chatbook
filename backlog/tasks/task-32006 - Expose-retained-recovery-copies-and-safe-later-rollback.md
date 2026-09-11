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
updated_date: 2026-09-11 07:45
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
Service composition underway in original scope: actual live app backup returns writers before packaging; retained archive excludes later native writes (1 passed7.58s). Actual isolated service restore with damaged ambient config and durable catalog after service.close passed1.06s; initial fixture incorrectly expected pre-relocation profile name, corrected to explicitly selected recovered name. Added restore facade/executor and durable restart status work; no UI/completion claim. Review found helper subprocess cwd could select feature sources when fallback test helper path used; copied exact HEAD helper into index snapshot for subsequent isolated dependency qualification. First replacement-service test setup could not build age due omitted offline Go env; corrected known cache env, no source change/network authorization.
Root service integration: qualify persistent private inspection/candidate work outside control and replacement targets, including narrow default service-storage ownership in inventory so opening recovery does not make later default backup incomplete. Add actual default-control warm-up backup red/green, preserve unknown/malformed control refusal and pending work across close. Files planned: recovery_service.py, service_storage.py, inventory.py and focused service tests. Existing original spec sections control placement/active staging exclusion apply; no broader storage scope.
Root persistent service-storage dependency independently approved by P (/private/tmp/chatbook-service-storage-independent-review.md): exact3files service_storage.py, inventory service exclusion, standalone test_service_storage.py.5passed0.91s; actual default-app warmup red then Complete backup1pass10.64s. Private control/work disjoint; unknown sibling and missing/corrupt marker remain blocking/no repair; existing dirs never chmod. Ruff1→1 baseline inventory import order, Bandit0. Required dependency for later rollback default-workspace correction. Full service/later integration remains In Progress.
Root independent later-rollback review completed: approved six-file correction1 patch67ec4965... after sole default-workspace issue resolved using committed9ac805b02 service_storage. Actual sourcecipher-only rawconfig/schema/WAL restoration, new encrypted post-edit safetycopy, knownabsence/safetyonly/credentialproof and freshchild Finish/Rollback receipts inspected.18 originaldistinct cases +newdefault case green; correction3pass22.82s, Ruff0/Bandit0. Exact-index native spotcheck+census next. Pre-safety abort remains separate originalTask22 unit; no wholefeaturecompletionclaim.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->