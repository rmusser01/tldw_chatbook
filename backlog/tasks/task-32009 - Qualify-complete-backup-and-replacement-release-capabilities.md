---
id: TASK-32009
title: Qualify complete backup and replacement release capabilities
status: In Progress
assignee: []
created_date: 2026-09-08 00:03
labels:
- backup-recovery
dependencies:
- task-31978
- task-31985
- task-31993
- task-31994
- task-31995
- task-31996
- task-31997
- task-31998
- task-31999
- task-32000
- task-32001
- task-32002
- task-32003
- task-32004
- task-32005
- task-32006
- task-32007
- task-32008
updated_date: 2026-09-11 15:22
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Complete and replacement capability labels are backed by end-to-end owner, archive, native, and product evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Complete and replacement capability labels are backed by end-to-end owner, archive, native, and product evidence.
- [ ] #2 Both restore destinations and later rollback preserve expected data under ordinary and interrupted operations.
- [ ] #3 User/release documentation states qualified platforms, exclusions, credential limits, and recovery actions without overstating guarantees.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute original Task26 only (Docs/superpowers/plans/2026-09-07-backup-recovery-06-release-evidence.md). Stages: (1) document implemented user flows and current evidence without release claims; (2) finish actual complete/isolated/replacement/later-rollback product qualification and synthetic owner fixture; (3) qualify native protocol/platform capability gates and named crash/adversarial cases; (4) update packaging, CI and user help, run named feature/inventory/lifecycle checks, review and record exact evidence. Dependencies remain unfinished; no acceptance criteria checked.
<!-- SECTION:PLAN:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-06-release-evidence.md#task-26)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Root begins bounded documentation slice: Docs/Backup-and-Recovery.md only, verified against actual F9 screen and startup-independent launcher. No capability flag changes or broad test runs. Current completed F9 replacement regression is commit3afae0913; later-rollback UI probe currently refuses before mutation and is under bounded diagnosis. User documentation must not imply whole Task26 or cross-platform qualification.
Documentation-only first slice drafted Docs/Backup-and-Recovery.md against actual F9/launcher/native_qualification.json. Independent source review /private/tmp/chatbook-user-docs-independent-review.md found missing recover --ask-password instruction; corrected with concrete rollback example and finish/abort distinction. Relative links and diff whitespace verified. Documents credential exclusions/password loss/private plaintext staging, explicit destinations/restart/safety copy, interrupted versus later rollback, retained copies, inert extraction, owner setup and currently unqualified platforms/product flow. No test/Bandit run for prose-only edit, no release flags changed; later rollback and Task26 acceptance remain open.
Release evidence-ledger documentation slice only: create original Task26 backlog/docs/backup-recovery-release-evidence.md with exact checked existing helper/native declarations and full F9 regression receipt, explicit unavailable combinations and unresolved product gates. This does not promote any installed capability or substitute historical suite claims for actual evidence; all other Task26 steps stay open.
Initial release evidence ledger created via Backlog document tool (tool interpreted path relative to docs; root moved only that new document to the exact original-plan path). Root verified native protocol2/primitive-only scope and identity, sole qualified helper darwin/arm64 declaration, permanent F9 sourcehash and actual84.67s receipt. Ledger explicitly separates historical packaging/native declarations from current product regression, records command/snapshot limits, laterrollback red and remaining Task26 gates. No new test/native/packaging run claimed and no capability changes. Incremental docs-only commit; acceptance remains unchecked.
Begin exact original Task26 Steps1/2/6 conjunction contract in qualification.py and new test_complete_roundtrip.py only. Implement specified release_capability(helper,owner_coverage,admission,archive,native_publish,restore,product_flow) all-gates API with behavioral red then green. This wiring unit grants no platform evidence and is not yet connected to product availability; no existing native qualification changed. Full synthetic/product/crash/packaging/CI qualification remains open.
Exact Task26 conjunction2 independently APPROVED /private/tmp/chatbook-release-conjunction-independent-review.md. API release_capability implements all7specified gates; test covers alltrue and eachsinglefalse. Isolated committed140d480e3+owned2: behavioral stubFalse red1failed0.95s /private/tmp/release-conjunction-red.log, final1passed0.46s /private/tmp/release-conjunction-final.log afterliteral/formatcorrection. Scoped Ruff1→1 identical inheritedunusedos import; format2clean; Bandit0, compile/diffcheck. No native qualification evidence or existing dispatch/product availability changed; API wiring, synthetic fullroundtrip and remaining releaseevidence are unfinished. Commit2source/test+task only.
Root now executes original Task26 Step7 packaging-checklist update only: document separate complete-capture/replacement qualification, exact immutable-build/native evidence, isolated owner/product/recovery checks and unavailable matrix cells. No platform promotion, CI run, dependency installation, or release upload. Existing checklist facts retained; only Packaging/PACKAGING_CHECKLIST.md belongs to this prose unit.
Original Task26 named owner-inventory + ProductionApp service-composition guards executed on clean committed b039a94f3 snapshot /private/tmp/backup-release-guards-dqqbng74 via private runner (HOME/XDG/config and network guard before imports): 13 passed, 2 failed in9.37s; /private/tmp/backup-release-guards.log. Inventory/static checks passed. Both actual lifecycle cases fail during TldwCli/load_settings with raw_source_selection_changed; root delegated read-only fixture/source-boundary diagnosis, no admission guard weakened. Packaging checklist exact diff independently approved by P; all relative links and diff verified, no packaging qualifications asserted.
Lifecycle diagnostic /private/tmp/chatbook-production-lifecycle-source-diagnosis.md and selection JSONL prove fixture-only retargeting: collection imports/enrolls config to private ProductionApp path; shared isolate_test_environment changes environment to each test path while SAME config module/native participant retain original enrolled path (active operations0). Root releases exactly Tests/ProductionApp/test_service_composition_lifecycle.py: preserve AST tests and original async graph/count/provider-close/mounted worker assertions, run each actual case in fresh child with private HOME/config/data selected before app imports, network guard+Null fixture keyring, existing catalog suppression. No shared fixtures/source guards/enrollment maps/product edits. Proper red already2fail; qualify actual children and final named file after correction; no lifecycle claim from fixture import failure.
Root documentation followup (Task26Step7) limited Docs/Backup-and-Recovery.md + existing release evidence ledger: explain committed current-profile setup panel, update actual saved-state later-preview status (preview-only; dispatch approval blocked), distinguish current isolated/owner evidence, record conjunction/packaging completion and unresolved lifecycle fixture check. No new capability labels/claims.
Root releases original Task26Step3 fixture Stage1 ONLY, exact Tests/Backup_Recovery/test_complete_roundtrip.py, per /private/tmp/chatbook-complete-roundtrip-fixture-preflight.md (installed71-owner map separately recorded). Two private configs initialized in separate fresh children before product imports; actual custom core/media/research DBs, explicitly shared prompts, ordinary semantic records/soft deletion/future queued work/empty owned root where existing APIs available; actual installed inventory/public Complete capture, semantic archive readback and resumed writes. Establish known selectors using existing actual native locator APIs after seed exits; no fabricated StorageItems/owner completeness/production guard edits. Preserve conjunction unit. No broad rich-owner fixture, new adapter declarations, restore/replacement/later UI claim yet. Rich/optional owners remain explicitly accounted unused/excluded or blockers, never falsely populated. Native network containment+fixture keyring, clean snapshot and bounded processes required.
Lifecycle test-only correction independently APPROVED by C (/private/tmp/chatbook-lifecycle-composition-independent-review.md). Root verified exact source+clean7bff snapshot hashes and AST-identical original async assertion bodies. Named file4passed52.34s, with real graph25.53s and scheduler24.45s children, both zero blocked network attempts. Ruff3→2 existing cleanup warnings; Bandit original cleanup B110 unchanged, two extra intended testassert B101 and fixed-interpreter no-shell exceptions documented. No production/sharedfixture/guard changes. Commit exactone test + taskrecord; wholeproduct qualification remainsopen.
Stage1 exact two-profile capture module now green in clean73e + capture fix/test only: 2 passed18.48s /private/tmp/complete-stage1-final.log. Root corrected final raw-vs-canonical manifest test expectation, retaining raw ZIP manifest equality and every payload hash check. Ruff0; production Bandit0, test six B101 assertions (two existing, four new behavioral assertions); no security issue. Agent final review unavailable after usage limit, root source/test review done; no full roundtrip/native volume/release qualification claimed.
Final capture Stage1 independent review subsequently available and APPROVED: /private/tmp/shared-directory-capture-independent-review.md, matching exact tested two-file snapshot. Supersedes prior temporary reviewer-unavailable note. Still only Stage1 A-live/B-closed capture/resume, not whole release qualification.
Next original finite Stage2a released: test-only plan_restore attempt against actual Stage1 capture with all roots and common destination for shared builtin concrete tree. Source preflight /private/tmp/two-profile-stage2-preflight.md predicts destination_collision. Author may factor private seed/capture helper preserving Stage1 assertions, add one permanent regression; no production changes/native authority mutations/fake inventory, no restore/fullmatrix claim. Native run coordinated after current model UI/core cohort.
39cfec085 core3 and78bd37031 Settings localmodel review committed after final independent approvals/tenpass combined workflow. 5f529fba7 two-profile capture likewise committed. Release doc refresh now authorized ONLY Docs/Backup-and-Recovery.md and backlog/docs/backup-recovery-release-evidence.md to reflect these exact bounded delivered controls/evidence and explicit HF freshprocess projection reuse unavailable; no release qualification promotion or speculative guidance. Actual Stage2a planner red1/35.50s destination_collision documented; no support claim.
Stage2a actual Complete2profile capture→boundedacquire→explicitall54root restoreplan now1pass20.95s with minimal exact-shared-tree planner support, /private/tmp/two-profile-stage2a-planner-positive.log. No actualpublication/openclaim. Newcomponentstageandnegative/compatibility18pass1.44s verifies existingphysicaldedup and ordinarysynthetic/file behaviors. Independent childaliasfinding corrected with nativebehaviorred→green. Stage2a test preserves allfiveStage1 childprograms byte-identical and originalseed/capture functionbody apart from factoring/return. Plan-only support readycommit; next original finiteStage2b executor/publication/freshmounted2profile reads needs separate tests, no scopeexpansion.
Doc2 final bounded update reflects committed5f529 capture,39cfec085 modelcore,78bd37031 Settings, plus root Stage2a amendment after322aea0c2 plannercommit: actual plan-only1pass20.95s and component/compatibility18pass1.44s, no isolatedpublish/openclaim. Pprose2+corrected staleStage1 citation independently Aapproved /private/tmp/recovery-local-model-docs-independent-review.md; root checked exact finalcapturegit hashes and replaced pre-finaljsoncitation with finalreview. Userguide buttonsequence/source-backed HFfreshreopen limitation preserved. Links/diffcheck pass, no prose tests warranted.
Stage2b test-only execution now released to C in test_complete_roundtrip.py after322aea0c2 sharedplanner commit: actual isolated executor and two fresh paired/mounted selected app reads of finite seeded owners, preserving sourceaftercaptureedits and explicit setup/queueinertness. Reuse originalpreflight exact APIs/privateenv. No richer/optionalcredential/model/external/temp matrix, no replacement/later/nativequalification/sourceowneredits in this slice. First actualproductrefusal preserved before any sourcecorrection.
Parallel test-authoring (native runs serialized) released only original Step5 real recovered-media deletion process-death cases, new test_recovered_media_crashes.py. Existing preflight /private/tmp/chatbook-task26-crash-volume-preflight.md identifies sameprocess exception coverage insufficient for kill evidence. Real private owner writes/committedtombstone/unlink and newprocessrecover, no syntheticreceipt/catalogstate, no production edits or general cleanup. Preserve exact perboundaryidentity/reference/sibling assertions, guardenv beforeimports. No multiplevolume claim.
Fresh committed snapshot /private/tmp/backup-release-committed-NjCxam at 6bee8367a: owner inventory census 11 passed in 10.42s; receipt /private/tmp/backup-owner-inventory-6bee.log. Stage2b actual restore_isolated completed publication, then test failed on captured0755 vs correctly planned private0700 directory mode (mtime preserved); narrow test-only correction released, no production permissions change. Recovered-media real child-exit crash cases released under31994; product gates remain unqualified.
Stage2b exact test-only isolated execution/fresh paired mounted two-profile opens independently APPROVED /private/tmp/two-profile-stage2b-independent-review.md. Clean6bee8367a+testSHA8b585160b46a1cbdb5159c461e674b0f8874a5c78e2ec41438c696459a372104:1passed56.74s /private/tmp/two-profile-stage2b-native-corrected.log. Real restore_isolated, all selected root mappings, actual service launch and native opened receipts, separate installation IDs/core DBs, shared prompts, saved notes/messages/media/research/deletions, queued-before-open→existing interrupted-on-restart behavior, ordinary restored writes and unchanged selected original hashes. Required setup remains; no owner approval. First1failed37.25s preserved: test expected captured0755 instead of planned private0700; narrow correction independently checks desired/archive/applied/disclosure/mtime, no production edits. Actual production launch env and select_profile explicitly NullKeyring before app imports, guard sees zero network attempts. Ruff0/AST/diff clean, Bandit7→8 only testB101. Frozen /private/tmp/two-profile-stage2b-report.md. Existing seed/capture/plan programs unchanged. This proves finite isolated/open cohort only; richer owners/fullF9/replacement/later/multivolume/release remain open.
Next original Task26Step3 finite rich-owner cohort released after bfb1d437a Stage2b commit, test-only test_complete_roundtrip.py to P: ordinary actual app Writing project/chapter/scene/version, Study deck/card, Quiz completed attempt/snapshot and Notifications read/dismissed persistent identity; exact owner APIs/readback per /private/tmp/two-profile-rich-domain-preflight.md. Extend seed+captured payload assertions only, preserve current discovery/sharedcore aliases/resume/digest/native/timeout logic. Run one existing Stage1 case on clean committed export plus test only, preserve concrete first failure before any production changes. This adds no product functionality or new owners and does not imply richer Stage2b/fullrelease coverage.
Root documentation-only ledger update released: record committed d067f3e41 process-death2 and bfb1d437a real two-profile isolated/freshopens1 exact receipts, distinguish Stage2a prior scope from later proof; no userguide capability promotion. Rich-domain capture and second-media-cycle tests underway separately and not yet claimed.
Ledger-only d067f3e41/bfb1d437a/census update independently approved /private/tmp/isolated-crash-ledger-independent-review.md after precise Stage2a supersession, tombstone-commit and restored0700/sourcepermissions wording corrections. Final ledgerSHAfe24978000d20c4b6474050ad92745b1df536d0243687727222134b26700f472, links/sourcefacts/diff verified, no prose tests. No qualified capability or stage beyond the exact native receipts is asserted.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->