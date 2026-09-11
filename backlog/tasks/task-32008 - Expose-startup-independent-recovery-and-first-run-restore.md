---
id: TASK-32008
title: Expose startup-independent recovery and first-run restore
status: In Progress
assignee: []
created_date: 2026-09-08 00:02
labels:
- backup-recovery
dependencies:
- task-31978
- task-31988
- task-31995
- task-32004
- task-32005
- task-32006
- task-32007
updated_date: 2026-09-11 09:14
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: First-run and damaged-installation users can inspect and restore without normal startup succeeding.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 First-run and damaged-installation users can inspect and restore without normal startup succeeding.
- [ ] #2 Minimal recovery reuses qualified services and never bypasses admission, target verification, or activation gates.
- [ ] #3 CLI and UI credentials stay out of process arguments, environment, logs, and persisted requests.
<!-- AC:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-05-user-workflows.md#task-25)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Released originalTask25 first bounded slice to A after read-only /private/tmp/chatbook-task25-preflight.md: seven original launcher/__main__/earlycli/first-run files and focused tests. Same RecoveryService commands, secrets only getpass/widgets, explicit recovery before normal config/app imports, optionalcontrolroot neverbypassesfixedadmission. First-run Restore opens existing app-ownedview without completingwizard/provider/modelwork. Pre-bootstrap damaged-state reader/real opened-ack/activation summary/inert extraction remain explicitly coordinated dependencies; no counterfeit success or unsupported-schema stage bypass. No source edits beyond named slice without concrete original-plan need/release.
Root released finite automatic damaged-bootstrap slice after A verified unchanged native `_read_recovery_file` instead of chmodding private_paths read. Fixed startupdecision first;16MiB O_RDONLY/no-follow TOML; malformed/inaccessible/failed orcancelledunlock routesminimal recovery beforeconfig/app. Missing preserves deliberatefirst-run. Standalone strict ConfigEncryption/getpass can verify input then transient config.set_encryption_password before appimport only if existing import handoffconfirmed; no argv/env/durablepassword/no config.py edits. Existing validated unlocked startup must remainavailable. Separate opened-ack/activation/extraction contracts remain original outstandingwork.
Release P originalTask25 inert extraction from /private/tmp/chatbook-inert-extraction-preflight.md: only new inert_extraction.py/test_inert_extraction.py; frozen actualsealed/group/destination/limits preview and native no-replace extraction. Opaque .bin payloads +inert relative mappingreport, selectedgroups only/no dependencyexpansion, no DB/config/app/owner migration/credentials/activation execution. Destination policy is consistently explicitabsentmanual directory independentof managedregistry availability (not a fallback downgrade); refuse known source/work/fixedbootstrap/defaultservice and caller-suppliedactualcustomcontrol overlaps. No existing data overwrite/merge or claimsrestored/opened. Finite owned publicationintent/nativebarrier and ambiguousoutcome retention allowed; no newgeneraljournal/resume/cleanup framework. Root ownsservicewrappers.
2026-09-11: P froze standalone inert manual extraction2files /private/tmp/chatbook-inert-extraction-report.md, patch SHAc79bd9410c110ba2ff8d9715f23677da61496797fb3d3584dfaf540cab62a3df. Root full-source review no current finding; exact committed-production snapshot + frozen unit21newcases pass and10architecture pass, only census5newproducer rows failed as expected,10.73s /private/tmp/inert-extraction-root-index.log. Added exactly5 reviewed input/private report-intent/payload boundary rows; independent A source+census review pending, clean census rerun pending. Root's initial hash-loader wrapper mistook metadata keys for filenames and copied nothing/no tests ran; corrected wrapper uses files mapping, not a product failure. Service/UI/CLI extraction still not exposed.
Original extraction view composition planned within existing single-worker service: start_extraction_preview(inspection_id, group_ids, destination, limits) retains native verification on worker; start_extraction(inspection_id, reviewed_plan) uses same acquired source and always adds actual control/work protected roots. Result reports inert_extracted/path/report only, no restoration/open approval. Root adding focused new service composition test first; service source extension waits C isolated branch correction release/freeze.
A independently approved exact inert extraction frozen two-file patch plus precisely five census rows: /private/tmp/chatbook-inert-extraction-independent-review.md; no findings/source edits. Root clean census11passed7.91s after root21newbehavioral cases pass10.73s. Production Ruff/Bandit0. Ready for bounded standalone engine commit, service/UI exposure remains original pending work.
2026-09-11: Root verified CLI recover parser likewise lacks new native pre-safety --abort choice. Released exact launcher.py mutually-exclusive parser choice + test_launcher.py real pre-safety abort dispatch proof with A's corresponding F9 correction. Existing dispatcher uses actual service action validation; no new engine. This updates exactly two firstslice hashes for focused C re-review after prior eightfile approval. Extraction service followup frozen separately with4actual+census11pass15.34s; A independent review queued.
A independently approved frozen extraction service followup: /private/tmp/chatbook-inert-extraction-service-independent-review.md, exact3file hashes/patch verified;4native composition+census11 passed15.34s in committed production snapshot, Ruff0/Bandit0. Standalone extraction engine already committeda19114179; facade ready for scoped commit. One embedded test trailing whitespace exposed by prior commit check is removed here; source behavior unchanged, subsequent mutation shell scripts fail fast.
Ready to commit corrected original first8file startup-independent/first-run slice. Root exact committed-dependency snapshot new launcher+wizard34cases22.15s /private/tmp/task25-correction1-root-new.log; A affected7cases12.25s, correction1affected15cases20.17s. C initial/correction1 independent approvals /private/tmp/chatbook-task25-first-slice-independent-review.md and /private/tmp/chatbook-task24-task25-correction1-independent-review.md. Bounded malformed/encrypted-config prebootstrap, actual isolated restore and stopped native replacement recovery, secret getpass and --abort preserved. Task25Ruff174→174/Bandit17→17 no additions; correction4filesRuff0/Bandit0. Actual initial Complete CLI replacement gap separately under32006; child-open receipt/owner state/inert workflow remain pending.
Root continuing original CLI exposure through existing RecoveryService: explicit extract command selects existing dependency-group IDs and absent manual destination, previews then confirms inert bytes (never restoration/opened); explicit later rollback command uses retained-copy operation + independently selected targetconfig + oldpassword and newconfirmed safety-copy password, shows restore/retire/preserve then confirms. Own launcher.py + new test_recovery_cli_actions.py only; no engine or schema changes. UI extraction remains pending. A firstbinding source and C preserved dependency planner are separate.
Root CLI inert-extraction two-file unit independently APPROVED P /private/tmp/chatbook-recovery-cli-inert-independent-review.md; exact sourcec4fc644d64be3e41fda59b54c35aa416a4066ed2e942fde2c9407de4eae0889e/test2f09e3cd78d4a9b6102c00bb8fe534902a413152a02bb2ac9c12482e46062784. Root exact committed-production snapshot3new+4existingguard7passed2.00s /private/tmp/recovery-cli-inert-index.log; final Ruff0/Bandit0. Real unsupported bytes selected/cancel/nooverwrite/brokenconfig unchanged/noapp/no network. Initialwrapper internal Payload JSON rendering defect corrected by explicit human preview fields and suppressed internal-plan status dump. UI extraction/later CLI/opened receipt remain original pending.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->