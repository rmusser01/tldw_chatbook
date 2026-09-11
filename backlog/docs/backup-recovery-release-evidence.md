---
id: DOC-2
title: Backup and recovery release evidence
---

# Backup and recovery release evidence

Status: incomplete qualification. This ledger tracks original Task26 / TASK-32009. A passing native primitive or individual product regression is not a whole-product release claim. No Complete or replacement release capability is enabled by this document.

## Declared platform evidence

| Layer | Checked evidence | Scope and limit |
| --- | --- | --- |
| Bundled age helper | `Packaging/backup_age/qualification.json`: darwin/arm64; age1.3.2; Go1.26.2; helper protocol1; Python3.12.11; APFS; recorded wheel digest | Installed evidence declares reproducible build, pipe handling, inherited native network denial, integrity and same-protocol interoperability. This continuation inspected the declaration; it did not repeat packaging qualification. |
| Native publication and admission | `tldw_chatbook/Backup_Recovery/native_qualification.json`: Darwin25.5.0, arm64, Python3.12.11, APFS, flags76583040; protocol2 | Installed evidence explicitly covers only publish_new, publish_file, publish_directory and admission. It does not establish isolated restore or replacement release qualification. |
| Other combinations | Helper declaration lists darwin/amd64, linux/amd64, linux/arm64, windows/amd64 unavailable; Python3.11/3.13 untested | No inference from arm64 macOS results. Other OS/filesystem, runtime and upgrade combinations require actual qualification. |

The exact native identity includes filesystem flags and interpreter version. Application capability checks must use the installed evidence, not a broader OS label or this prose.

## Verified F9 replacement regression

Repository test: `Tests/Backup_Recovery/test_f9_replacement_workflow.py`, committed in `3afae0913cf2f030705f77704e62e153e9770901`.

Actual result: **1 passed in84.67s**, call83.40s. Executed in the isolated production extraction `/private/tmp/f9-complete-restart-probe-g8ga7iaj`, with the production safety-scope screen matching commit95c021d6c (`9b5841214a181ef6ef01d45f68b575f9d7abff4984f95acffef3ddf7a494b246`). The snapshot predates the Notes/Skills/OpenAI prerequisite follow-ups; no claim is made that this receipt exercised those later changes.

Exact command, with the project virtual environment activated:

```sh
GOMODCACHE=/private/tmp/task31985-go/mod GOCACHE=/private/tmp/chatbook-task12-gocache GOPROXY=off python /private/tmp/f9-workflow-run.py -q Tests/Backup_Recovery/test_f9_replacement_workflow.py
```

The private wrapper creates HOME/XDG/config and installs the network guard before imports. It provides a120s outer watchdog; the permanent test retains its own110s seed,100s child and bounded service waits. The helper uses the preprovisioned offline cache. The fixture uses NullKeyring and generated disposable local data; it does not use developer profiles, credentials, models or external endpoints.

The test exercises public Complete capture and native note readback, ordinary writes resuming before packaging, normal CLI F9 shutdown, real exec into fresh minimal recovery, new archive inspection, explicit selection of35 displayed preserved safety items,21 initially unchecked credential omissions, actual Abort untouched replacement, explicit acknowledgements and fresh password/review, validated replacement, clean shutdown and zero blocked network attempts in the fresh child. It does not inject engine scope, owner approvals or a successful service result.

Local receipts: `/private/tmp/f9-workflow-transfer-final.log`, `/private/tmp/chatbook-f9-replacement-workflow-transfer-report.md`, `/private/tmp/chatbook-f9-workflow-root-review.md`. These temporary files are local audit aids, not portable build inputs. Permanent test SHA256: `313275b31fe26c2b2c03fff5abdd49a33030b4311725783bbf38ef7d5cd23896`.

## Two-profile capture: Stage1 only

Commit `5f529fba7` preserves shared directory group metadata during capture and adds `test_two_known_profiles_complete_capture_preserves_native_state_and_resumes` in `Tests/Backup_Recovery/test_complete_roundtrip.py`. The actual private fixture has two known profiles with custom database paths and a shared prompts database; one application is live and the other profile is closed at capture. Public Complete capture preserves declared shared peers, archive metadata and native stored content, then both profiles resume ordinary writes. Installed owners are accounted for as included, unused, or explicitly excluded; unused owners are not populated-owner roundtrip evidence.

`/private/tmp/complete-stage1-final.log`: **2 passed in18.48s**, comprising the native capture case (call17.85s) and the existing seven-gate conjunction test. The clean snapshot `/private/tmp/complete-stage1-51kujhi2` was exported from `73e419651a77c67525cc36c01dc1abebe50ae1d3` with the exact capture correction and test. It used private HOME/XDG/config/cache/temp, a network guard before application imports, NullKeyring and the preprovisioned offline helper caches. `/private/tmp/shared-directory-capture-independent-review.md` verifies the final tested two-file scope and owner-status accounting: `capture.py` SHA256 `211a0534239bd039f18c64cdee73c83e268f160dcce93efb951d9435ccbd836e` and `test_complete_roundtrip.py` SHA256 `4d3f56a88e4adfbaf4e2031474afff87c850ed6d18f79b5a1f8c3fdaacd3695a`, both matching commit `5f529fba7`. This is capture and resumed-write evidence, not two-profile isolated restore or a complete create-to-rollback sequence.

## Two-profile isolated restore and fresh opens

Commit `bfb1d437a` adds the finite Stage2b scenario to `test_complete_roundtrip.py`. Clean `6bee8367a` plus only the test change passed **1 test in56.74s** (`/private/tmp/two-profile-stage2b-native-corrected.log`), using the same private runner and offline environment as Stage1. Test SHA256: `8b585160b46a1cbdb5159c461e674b0f8874a5c78e2ec41438c696459a372104`. Independent review and frozen source/artifact hashes: `/private/tmp/two-profile-stage2b-independent-review.md`, `/private/tmp/two-profile-stage2b-frozen-hashes.json`.

The real isolated executor publishes the captured profiles into separate destinations. Each profile then opens through the actual recovery service, paired CLI selection, and a fresh mounted TldwCli process. Real opened receipts, distinct installation identities, saved notes/messages/media/research/deletions, shared prompts, retained Needs setup, and ordinary writes under the new identities are checked. Captured queued ingest remains queued before opening; ordinary startup records its existing interrupted-job state without replay. Selected original config/database hashes remain unchanged. The terminal transport is adapted for a headless mount; no opened receipt, owner approval, or successful result is manufactured. Production launch filtering and paired selection both enforce NullKeyring; network-attempt assertions remain zero.

The first run failed1 in37.25s after successful publication because the fixture expected captured0755 directory permissions. The corrected assertion independently checks archived desired metadata, the plan's applied private0700 policy, exact mtime retention and disclosed normalization. No production code or source permissions were changed; restored permissions follow the disclosed private0700 plan. This proves the finite captured cohort's isolated restore and two opens; source removal, richer populated owners, interactive F9 restore, replacement/later rollback and full release qualification remain separate.

## Recovered-media deletion process interruptions

Commit `d067f3e41` adds `Tests/Backup_Recovery/test_recovered_media_crashes.py`. The first and only native run passed **2 tests in6.05s** (`/private/tmp/recovered-media-crashes-first.log`) on clean `6bee8367a` plus that test, Darwin25.5.0 arm64/Python3.12.11. Command: `GOMODCACHE=/private/tmp/task31985-go/mod GOCACHE=/private/tmp/chatbook-task12-gocache GOPROXY=off python -m pytest -q Tests/Backup_Recovery/test_recovered_media_crashes.py --basetemp=/private/tmp/recovered-media-crashes-hgvwr57g-pytest`, with the project virtual environment activated and all owner work in private guarded child processes.

The actual owner commits its tombstone and pending deletion before one child exits91; the other passes through the exact native payload unlink before exiting92. Independent read-only catalog observations verify both interruption boundaries. Fresh owner processes complete pending deletion, preserve the tombstone and all reference identities, keep unrelated media bytes unchanged, and recover idempotently. Test SHA256: `90e21fcdc700b4627c3cba299426737b1ab3cc9165677b1eff4c897d801820c9`. Independent review: `/private/tmp/recovered-media-crashes-independent-review.md`; exact boundary facts and hashes: `/private/tmp/recovered-media-crashes-frozen-hashes.json`. No production code changed. This is abrupt process-death evidence; it does not establish power-loss durability, multiple-volume behavior or the complete crash matrix.

## Delivered local model and RAG setup controls

Commit `39cfec085` adds the current reviewed local-HF native identity needed for same-process projection builds. Commit `78bd37031` exposes **Settings → Library/RAG → Review local model → Approve local model**. The user separately chooses **Review recovery → Approve RAG owners**, then explicitly confirms **Reconcile / rebuild**. Approval alone performs no model load, download, configuration save, or rebuild. Config and unrelated owner approvals remain false in the actual positive fixture.

The final clean five-file snapshot `/private/tmp/local-model-core-yLmzWN` uses committed `73e419651a77c67525cc36c01dc1abebe50ae1d3` plus the exact reviewed core3 and UI2 changes. `/private/tmp/local-model-core-ui-final.log`: **10 passed in62.17s**, including six native identity/invalidated-authority cases, ordinary model compatibility, existing approved-load and retirement cases, and the mounted isolated Settings model review → separate RAG review → actual backfill readiness → semantic query. The actual UI worker alone produces readiness; the test does not repair it with a later direct reconcile call. This is an actual mounted Settings host with installed local source owners, not a full TldwCli launch qualification.

Approved source SHA256s, also verified against the committed files:

| File | SHA256 |
| --- | --- |
| `RAG_Search/model_recovery.py` | `32e0ea5c27670ad0c1a9c3e9dd37c23dda73cccf87aee81c51712ac780c578f6` |
| `RAG_Search/recovery.py` | `8e84bbaba123f3477109600ffb601242e64753fa2408ce7fa7fdc00cc913ca1c` |
| `Tests/Backup_Recovery/test_activation_local_embedding.py` | `8d03c42a59b3fa0c0981d6ecb20eae518aaf526ee935d095684bb686a6c86d5b` |
| `UI/Screens/settings_screen.py` | `b5d53134f5d8a7fcd0df8cd9eaef2e95951a1e0ee2d9e7555d039a3b743dd16c` |
| `Tests/Backup_Recovery/test_rag_model_recovery_controls.py` | `a102f33202f163064ea4b6e352306bd1270fcd603a1c4ba31c470eb9e207cd0b` |

Production paths in the table are relative to `tldw_chatbook/`; test paths are relative to the repository. Independent reviews are `/private/tmp/local-model-core-independent-review.md` and `/private/tmp/chatbook-rag-model-controls-independent-review.md`.

Eight mounted UI negative cases also explicitly assert zero blocked network attempts: cancel, changed bytes, changed selection and navigation passed4 in68.54s (`/private/tmp/rag-model-ui-negative-confirm-final.log`); draft, late preview, missing local model and ordinary profile passed4 in65.83s (`/private/tmp/rag-model-ui-negative-preview-final.log`). Their production UI bytes are identical to the final combined snapshot; core dependency hashes for these earlier batches are recorded in `/private/tmp/rag-model-ui-final-frozen-hashes.json`. Unchanged model-only approval and accepted-native-write cancellation have separate passing receipts. Eleven UI scenarios is the incremental distinct total, not an eleven-case final module rerun. No separate mounted nonlocal-provider case is claimed.

Fresh-process HF projection reuse deliberately remains `projection_model_reopen_unavailable`. The file-closure/model-review receipt does not establish that a saved projection was built from the freshly loaded model's computation. The approved design's projection contract requires sufficient provenance and explicit qualified reconciliation/rebuild; it does not permit automatic reconstruction or query when proof is missing. `/private/tmp/local-model-fresh-reopen-preflight.md` documents this existing boundary; no new provenance feature was implemented. Model receipt reopening and projection readiness reopening are different capabilities.

## Open product evidence

- The saved-state later-rollback preview originally refused with `local_snapshot_absence_unclassified`. A bounded created-destination observer now reaches the actual UI review: child exit0 in7.14s, `/private/tmp/later-created-saved-ui-preview.log`. This used the saved F9 extraction with only the observer update; it is preview-only evidence. No confirmation, target mutation or new safety copy occurred. Automatic approval review blocked the proposed execution-validation dispatch changes pending exact user authorization. Full later-rollback qualification remains open.
- Current-profile setup reporting is committed in `3ab0e5ac2`. Twelve new cases and four affected cases passed on committed `31e0034d2` plus the exact three-file change, including actual replacement/later generations, a fresh-process read and damaged-evidence refusal. Individual owner controls have separate receipts in their Backlog tasks; these do not establish the missing full product sequence.
- Stage2a two-profile isolated planning initially failed1 in35.50s with `destination_collision` after the real Complete capture. Commit `322aea0c2` now permits a common destination only for explicitly shared concrete trees with matching full relative contents, ownership, shared declarations and metadata. The unchanged actual two-profile test passed1 in20.95s on clean `5f529fba7` plus the exact planner/test overlays (`/private/tmp/two-profile-stage2a-planner-positive.log`); it maps all54 selected roots in a fresh archive-only process. Eighteen focused and existing compatibility cases passed1.44s on clean `78bd37031` plus the exact three-file change (`/private/tmp/shared-directory-plan-final-compatibility.log`). Independent review: `/private/tmp/shared-directory-planner-independent-review.md`. Stage2a supplies planning-only evidence; Stage2b above separately verifies finite isolated publication, paired fresh opens and semantic readback.
- MCP fresh-store rebackup discovery and inactive directory display/path corrections remain blocked pending the exact user approvals. No completed MCP fresh-owner workflow or rebackup qualification is inferred from the local-model controls.
- The specified seven-gate conjunction regression is committed in `b039a94f3`; it is not yet wired to product availability and supplies no native evidence. The packaging checklist update is committed in `7bff17e89`, with its qualification items intentionally unchecked.
- Stage2b above separately establishes finite two-profile publication and fresh opens after the Stage2a planning-only receipt. Rich populated-owner coverage, the complete create/inspect/isolated-open/replace/later-rollback sequence, remaining native crash/multiple-volume matrix, product release-gate wiring and qualification CI remain Task26 requirements. Run named feature and inventory/lifecycle checks only; no full-suite run is implied. The latest committed owner inventory check passed11 in10.42s on clean `6bee8367a` (`/private/tmp/backup-owner-inventory-6bee.log`); it is a census check, not populated-owner evidence.

Update this ledger with exact revision, environment, command and actual result when each gate is demonstrated. Keep Complete capture and replacement qualification separate. A generic boolean conjunction test establishes wiring only; it cannot substitute for owner, archive, native or product evidence.
