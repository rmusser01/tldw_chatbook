---
id: TASK-32004
title: Restore and reopen an isolated profile
status: In Progress
assignee: []
created_date: 2026-09-08 00:00
labels:
- backup-recovery
dependencies:
- task-31978
- task-32000
- task-32001
- task-32002
- task-32003
updated_date: 2026-09-11 04:14
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Isolated recovery creates and reopens a separate profile without altering original local data.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Isolated recovery creates and reopens a separate profile without altering original local data.
- [ ] #2 Damaged current configuration and databases do not prevent archive-only recovery.
- [ ] #3 Fresh launch respects relocated paths, credential/device isolation, durable activation, and projection readiness.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute original component04 Task21 only: (1) implement inert private ProfileCatalog register/resolve with opaque IDs and checked explicit config/data locators, test source-preserving restart and corrupt/linked/changed mapping refusals; (2) compose stage/publication/installed validation/activation/catalog under native maintenance after dependency contracts are ready, with fresh local identities; (3) fresh-process launch verifies catalog/admission/activation and filters inherited selectors; (4) focused isolated archive-only/reopen fixtures, scoped guards/Ruff/Bandit, review and docs. No task completion or launch exposure from the catalog primitive alone.
<!-- SECTION:PLAN:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-04-restore-recovery.md#task-21)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Starting independent ProfileCatalog primitive from original Task21 while publication/activation integration dependencies finish. Existing task found; no duplicate. Catalog is a convenience locator registry, never restoration/activation authority, and cannot authorize a launch on its own. Constructor and lookup create no files; private immutable per-ID records, repeated exact registration may be verified durably; changed mappings refuse. No edits to cli/isolated executor yet; ADR-126.
ProfileCatalog primitive implemented and independently reviewed, no launch/executor exposed. Private per-opaque-ID strict records persist explicit config/data locators only; constructor/resolve never write; exact registration retries revalidate and reflush file/catalog/control/ancestor associations, changed ID mappings refuse. Locators checked without config parsing; linked/damaged/public records refuse, targets remain unchanged. Initial18 behavioral reds ->18green; review identified absentcontroldir and missingparentretrybarrier, both proven2reds thenfixed. Final23passed .72s (/private/tmp/chatbook-catalog-reviewed.log), fulltouchedRuff/formatclean/Bandit0. Review /private/tmp/chatbook-profile-catalog-review.md approved scopedfixes. Exactnewproducerrow ProfileCatalog.register/open1 generic_boundary. This conveniencecatalog is not restoration/activation authority; catalogrebuild and isolatedexecutor/freshlaunch remain required originalTask21work. ADR126.
Read-only Task21 preflight /private/tmp/chatbook-isolated-executor-preflight.md inspected. Ruling: fixed bootstrap admission remains canonical regardless of custom control_root; spec rev4 lines244–267 mandates custom root associations there. Reject proposed default_bootstrap_root override. Catalog-before-commit typed seam, fresh process selectors, and inert encrypted credentials still require scoped implementation after Unit5 review/release. No Task21 source edits authorized yet.
Task21 isolated executor implementation released after Unit5 committedb71a37edb and independent review. Author rag_publication owns new isolated_restore.py/test_isolated_restore.py and narrow journal/publication typed catalog-before-commit seam; staging/credentials isolated retain-encrypted no-shared-keyring policy before candidate hashes; cli/__main__ explicit verified process selectors; config runtime installation identity; profile_catalog observation if needed. Fixed bootstrap stays canonical; custom controls register existing associations. Archive_reader/archive_models narrow acquisition-bound encrypted copy metadata and crypto.transform optional streamed input digest check authorized if needed to retain real encrypted credentials without password. Ruling: original§6 declared/protocol-aware threat model, no arbitrary Python-object forgery defense or weak-reference attestation registry required; use existing typed acquired-archive metadata minted by real authenticated acquire, recheck actual cipher bytes before retention, no inference from sibling filename or imported manifest. A small private sealed receipt is acceptable only if existing trust boundary actually requires it, not speculative. Tests real archive-only restore/corruptambient/originalbytes/untrustedlaunch/cataloginterruptions/credentialcollision/multiprofile, no broadsuite/model/network. Root owns tracking/docs/census/staging/commit. No replacement/laterrollback/UI edits.
Task21 real config-only/partial isolated positive found explicit selected paths.data_dir is absent after current planned publication, so catalog correctly refuses. Extend author ownership narrowly to restore_plan.py: represent that explicitly selected isolated data directory as an empty planned container only when no selected publication root already supplies it. Include normal fingerprint/metadata/absence/native scope validation and show it in immutable plan; never mkdir implicitly during finalization, overwrite existing roots, infer ambient data or force unrelated payload selection. Original partial validated-group separate-profile contract only.
Real explicit CLI launch --help passed selector verification but existing app runner rebuilt generated package CSS before help, creating only a generated timestamp diff in tldw_chatbook/css/tldw_cli_modular.tcss. Root verified no other CSS changes/preexisting dirty CSS; restoring that exact incidental timestamp only. Future CLI runtime fixture uses disposable exact feature export to avoid tracked generated writes. Real config-only/partial publication/catalog3cases passed; read services must use actual explicit owner, since --help intentionally does not initialize global DB owner.
Task21 bounded13file freeze independently APPROVED /private/tmp/chatbook-isolated-executor-independent-review.md, no findings. Report /private/tmp/chatbook-isolated-executor-report.md, patch cfdaf12981aff2f9a12ca4406f1876d71e4cb0334761a7178eb04f3b3a28e429.18 distinct focused cases across documented runs,13 compatibility; final credentials3pass13.14s and interpreter-start network guarded real launcher/reader1pass15.48s. Fixed bootstrap/custom association, catalog-before-commit under held session, actual fresh selected process reads Notes/Chat/Media and writes fresh client ID; retained authenticated ciphertext remains after staging removal, shared keyring forbidden. Not interactive TUI qualification. Ruff130→127 and Bandit4→4 no added findings. Root census two added actual private retention sinks: retain_encrypted create_private_file/write, qualified operation-owned encrypted credential archive retained after authenticated acquisition. No arbitrary object attestation. Root exact-index gate and scoped commit follow.
Root exact staged export Architecture inventory11passed15.85s. Two new retained-cipher sink census rows independently APPROVED rag_enhanced_qualification; exact13source/test hashes match. Scoped15path commit approved. Original downstream runtime/replacement/service/UI qualification remains; no whole backup completion claim.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->