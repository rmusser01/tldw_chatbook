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
updated_date: 2026-09-11 09:02
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
Service independent review P confirmed actual P1: freshchild killed during activation-pair write leaves ordinary bootstrap._records fenced, so pending discovery/status/start_recovery could not reach already-capable exact executor. Root correction uses bounded strict pending-only fixedrecord discovery for recovery UI; private/no-follow/typed/version/path/hashname validation retained, ordinary bootstrap unchanged, executor still independentlyvalidates selectedpair. Adding actual killedpair serviceRollback regression +malformedpending refusal; no broad startupbypass. Fullservice remains unfrozen during correction.
2026-09-11: Root service correction actual interrupted replacement activation-pair discovery + exact malformed fixed pending record refusal (version/missing_version/relative_selector) and actual pre-safety abort (pending and credential review) passed six cases6.84s /private/tmp/recovery-service-pending-correction.log. Reader exposes recovery only; ordinary startup still refuses split pairs and native executor validates selected authority. Ruff clean after one test import separator; production Bandit0. Independent C review checking equivalent isolated split-pair recovery, not yet claimed complete.
2026-09-11: Corrected base RecoveryService and isolated split-pair dependency ready for scoped combined commit so the new facade-dependent crash test lands with its service. Root exact committed-production snapshot (no pending A UI/CLI sources): full service19 + isolated pair4 =23passed46.59s /private/tmp/recovery-service-combined-final.log; census11passed14.69s /private/tmp/recovery-service-combined-census.log, no census delta. P independent original service review/C second review findings corrected; P approved C exact pair correction /private/tmp/chatbook-isolated-pair-recovery-independent-review.md. Final service/tests Ruff clean, service/isolated/control production Bandit0. Updated root report /private/tmp/chatbook-recovery-service-corrected-report.md; hashes /private/tmp/recovery-service-corrected-frozen-hashes.json and /private/tmp/isolated-pair-frozen-hashes.json. Original Task25 extraction facade/opened-owner-state integrations and broader UI/release qualification remain pending.
2026-09-11: Root actual public Complete live backup→independently discovered current target→CLI replacement reaches native staging but refuses staging_target_alias. /private/tmp/test_cli_complete_replacement_probe.py; /private/tmp/cli-complete-replacement-diagnostic.log. Exact preserved item recovery.control:service is default_control_root().parent and contains service control-work staging. Read-only A tracing smallest internal-control/payload boundary correction and actual default-selector namespace overlap before source edits. Initial probe-only missing external conftest fixture/data_dir logical-item assumption corrected; no production change. Existing component replacements passed but did not cover this public default service-preservation case.
2026-09-11: Root accepts A finite default service-work boundary proposal /private/tmp/chatbook-default-recovery-control-boundary-preflight.md. Released service_storage.py readonly exact recognized default service container/work descendant predicate; staging.py only preserved-item overlap may reuse it. Restore/retire/destination/bootstrap/enrolled-root guards and preserve fingerprint remain unchanged. Add focused test_service_storage.py cases; actual public CLI stage red already confirmed. Ordinary initial profile binding remains separate identified Task22 gap, proposal pending.
Actual Complete archive/current target CLI probe now advances past default service-control staging alias after bounded service_storage/staging fix (12 focused storage cases pass) and reproduces replacement_local_binding_required. This confirms original first-replacement composition gap: ordinary profile has registered sources but no durable profile binding. Release finite replacement.py/control_records.py implementation to reuse actual registration and bind under native held UNBOUND+selected-source authority after confirmed replacement; typed destination/config-owner matching must support different source/current IDs. No new enrollment UI, registry, permissions, or blanket control overlap exceptions. Root retains storage/staging changes.
Further read-only composition trace: restore-plan fingerprint currently hashes mutable entry names/mtime for preserved fixed recovery bootstrap; bind_profile and existing register_pending necessarily update that recognized protocol-control directory before recheck. Root will narrowly qualify recognized intentionally_excluded recovery.control fixed-bootstrap/service directories by stable local identity and ancestry with existing control validators, retaining full payload/config/path fingerprint checks. No silent whole-plan refresh or arbitrary preserved-directory exemption. Fixed service-work actualstage byte positive + alias/unsafe-parent/ordinary preserved-payload negatives now pass with helper cohort16cases1.21s.
Root five-file control/payload corrections independently approved by C: service-work3files patcha542f4748cca91aec204617424c134814be47e9ff1e5f9703e5826aa14caac39 and fingerprint2files patch101f89445f5d5c6a6910e90b1ac9b20ac6661b57ae9a1ba24bf0514ad48ab417. Recognized installed control storage retains stable identity/ancestry/mode with strict existing validators; service children retain identities too. Actual binding+pending control activity no longer invalidates payload review; ordinary config/data, damaged/unknown control, aliases/replacement still refuse. Root exact committed-production snapshot new28+existingguard7+census11 =46passed11.69s /private/tmp/recovery-control-root-index.log; scoped Ruff0/Bandit0. No pending A binding or Notes source in snapshot. First ordinary CLI replacement qualification remains A ongoing.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->