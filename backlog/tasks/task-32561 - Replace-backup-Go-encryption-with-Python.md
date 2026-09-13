---
id: TASK-32561
title: Replace backup Go encryption with Python
status: Done
created_date: 2026-09-12 15:09
labels:
- backup-recovery
references:
- https://github.com/rmusser01/tldw_chatbook/pull/2642
documentation:
- Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md
- backlog/decisions/126-complete-local-backup-and-recovery.md
updated_date: 2026-09-12 16:30
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User explicitly rejected Go as an oversight in the original backup plan. Replace the backup encryption implementation and remove Go build/package/runtime requirements, preserving the existing backup scope and archive/recovery contract. This user correction supersedes the plan's Go-helper instruction.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Backup encryption and decryption use Python and existing project crypto dependencies, with no Go toolchain or Go executable required for source installs, packaging or runtime.
- [x] #2 Preserve the existing passphrase-encrypted .tldw-backup.zip.age format, streaming budgets, authentication, cancellation/cleanup and credential/rollback behavior; verify compatibility against independent existing vectors/artifacts.
- [x] #3 Remove obsolete Go packaging/test requirements and update affected documentation, qualification checks and PR2642 without adding unrelated features; run targeted regression and Linux verification with exact remaining native-platform limitations documented.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stages 1–4: Complete. Python worker, preserved parent transport, Go delivery removal, package tests, installed macOS F9/rollback, final Linux verification and PR update are complete.
Plan: Docs/superpowers/plans/2026-09-12-python-backup-encryption.md.
ADR required: no new ADR. Existing backlog/decisions/126-complete-local-backup-and-recovery.md now links the user's Python correction; archive/process/recovery boundaries are unchanged.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
User: 'go should haven ever been used'. Plan authoring choice is superseded; do not insist on Go due to original plan. Python cryptography candidates are under read-only assessment. Project already depends on pycryptodomex. Existing pure-Python age packages and Rust-backed pyrage are not assumed suitable; stream limits and byte-password compatibility must be verified. Existing crypto public transform signature and .age format should remain stable. Linux baseline is tracked separately byTASK32560 and shows unrelated Darwin-native publication/barrier gaps.
Design/plan recorded in Docs/superpowers/specs/2026-09-12-python-backup-encryption-design.md and matching plans path, implementing the user's explicit language correction while preserving .age files and existing API. Existing pycryptodomex provides primitives; no additional crypto runtime dependency or Rust replacement. Bounded isolated Python worker retains cancellable KDF and pipe secret handling. Task1 worker implementation/review dispatched under SDD; root owns independent legacy test-vector provenance, packaging plan and records. Existing Linux test ciphertext is synthetic and predates the correction; preserve it for compatibility, with no additional Go execution.
Fixed independent interoperability fixtures committed749f116b2: official age1.3.2 ciphertext (SHAfb98c008...), previous worker empty/raw4096-byte password ciphertext (SHA3f67cd98...), prior worker streaming ciphertext (SHA747e0916...). Root verified remote/local hashes; README records fully synthetic inputs and original37pass test receipt. No newGo execution. Worker/test-only implementation active; independent read-only packaging map delegated while root prepares unchanged transport/capability integration. Design/plan commitc19068652.
Task1 worker frozen for independent review: 60 tests passed (15.95s), including three fixed pre-existing age archives and exact ciphertext reproduction with recorded entropy. Ruff/format/compile and production Bandit passed; test-only Bandit findings retained in report. Worker protocol2/python/age-v1. Parent transport and Go packaging removal remain pending. Evidence /private/tmp/task32495-python-worker-report.md; reviewer dispatched.
Task1 review found closed stdout could emit interpreter-finalization diagnostics. Narrow fix uses unbuffered FileIO at protocol boundary. Real info/encrypt/decrypt closed-pipe regressions RED3 then GREEN3; final worker module63passed16.39s. Production Bandit0, Ruff/format/compile pass;26 LOW test-only Bandit findings disclosed. Fixed worker SHA451a3dc158c3c1e8ee599f384a23046eb9d446ca94e8c5e04b8ffe9c6e068c05. Scoped re-review pending; no Go executed. Packaging dependency map completed /private/tmp/task32495-python-packaging-map.md.
Task2 production patch was rejected by automatic approval review: removal of existing binary manifest/SHA validation was judged to weaken encryption integrity. No production patch applied. Safer bounded integration will retain existing SHA256 verification over fixed Python worker bytes, with expected digest as parent-owned constant instead of a Go/platform manifest. This preserves tamper/corruption refusal while removing Go and target delivery machinery. Worker edits must update digest in same reviewed change. No additional feature/dependency/manifest/platform qualification.
Task2 implementation frozen for independent review: Python isolated parent transport, fixed worker SHA256 integrity retained, shared Python fixture, no Go invocation. Final crypto42passed12.02s; directly affected missing-backend gate1passed0.61s. Ruff/compile/diffcheck pass. Production Bandit retains same two LOW fixed-subprocess baseline findings; test LOW findings disclosed. Report /private/tmp/task32495-python-transport-report.md; review package /private/tmp/task32495-transport-review.diff. Source/editable availability no longer depends on prebuilt Go binary; native qualification unchanged.
Task4 current macOS archive-reader/archive-writer/credentials run:132passed,1failed14.18s; /private/tmp/task32495-archive-credentials.log, fixture/JUnit /private/tmp/task32495-product-58g82scb. Failure is unchanged plaintext test_explicit_directory_metadata_and_media_storage_roundtrip comparing unnormalized expected directories against reader's synthetic:false default. Exact test on clean pre-Python2220f3bdc baseline reproduces same assertion (1failed1.17s); /private/tmp/task32495-archive-baseline.log, JUnit /private/tmp/task32495-product-2y9gij34/results.xml. No Go executed, no source/test assertion changed. Baseline failure retained separately from Python regression outcomes.
Full release-gate module passed24cases3.19s through Python source backend. /private/tmp/task32495-release-gates.log and /private/tmp/task32495-product-xuwiwbo8/results.xml. Task2 independently approved and committed76a0f1657. Task3 Python-only packaging agent active; final installedF9/Linux checks await reviewed packaging.
Task3 frozen: all15Go-onlyassets removed, ordinaryPython wheel/sdist/editable delivery, workerLFattribute preserveschecksum, existingnativegates/workflowcells/12entrylists andartifactexclusions retained. Packaging10passed23.09s after boundedexistingoffline-cache read permission; static/YAML5embeddedPython/TOML/diffchecks pass, productioncheckerBandit0. Report /private/tmp/task32495-python-packaging-report.md; independentreviewactive. RootstartedinstalledF9three-mode verification againstfrozenPythonpackage.
InstalledPythonF9create/restore/open passedall3modes106.47s: plaintext,encrypted,encryptedincludedcredentials. Realordinarywheel,installedoriginandfilepreservationassertionsretained; /private/tmp/task32495-f9-endtoend.log, fixture/JUnit /private/tmp/task32495-product-dk2y2ont. Combinedreplacement+laterrollbacknowrunning withunchangeddeadlines. Task3reviewidentifiedtwoCIcandidateprovisioningdefects(cachepopulation,Windowsuvscriptspath) andoverstatednetworkreceiptwording; scopedfixactive, applicationworker/transport/nativequalificationunchanged.
CombinedF9 firstattempthit privatewrapper230s limit, not repositoryassertion. Receipt /private/tmp/task32495-f9-later-rollback.log; fixture /private/tmp/task32495-product-_tbubnj2. Firstreplacementresult succeeded/restoration_validated; laterrollback reached reviewed changedacknowledgement, with no childtraceback beforetermination. Laterchild had used ~122s ofitsunchanged180s allowance whenaggregatewrapperkilledpytest; seed110/replacement150/later180 alreadyallow440s serially. No remaining Python cwd underfixturefound. Ruling: privatewrapperaggregate480s onlyforthisexactcombinednode (440phasebudgets+40setup), preserve repository/innerdeadlinesandfirstfailedreceipt. This corrects harness cutoff without claimingfirstattemptpassed.
CombinedinstalledF9replacement+laterrollbackPASSED1in207.23s (call199.63/setup7.27), allrepositorydeadlinesunchanged. Privateaggregatewrapper480 usedafterpreserved230s harnesscutoff; successfulrunalsofinishedbelowold230. JUnit/fixture /private/tmp/task32495-product-pvzjhf_4; log /private/tmp/task32495-f9-later-rollback-bounded.log. Wheel7e408f89dc4b154a3646fc69437cf28311bd19580b30ea811e012c8bae0f5dab;2027installedfilespreserved. Lastcheckpointlater_rollback_complete succeeded/restoration_validated. Finalcross-taskreviewapprovednoCritical/Importantfindings /private/tmp/task32495-final-review.md. Linuxcorrectedworker63pass18.2s; remainingcrypto/packageinprogress.
Final Linux results against public revision 98b5d0b24: worker 63, crypto 42 and packaging 11 passed, with zero skips. Final evidence: Docs/Development/backup-python-verification-2026-09-12.md. All planned code and test work is complete; finalizing PR and records. Existing ADR-126 applies, with its Go integration choice superseded by the user's explicit Python correction and linked correction spec.
PR2642 now describes the final Python implementation and actual tests. Final evidence: Docs/Development/backup-python-verification-2026-09-12.md. Worker 63, transport 42, packaging 11 and release gates 24 passed on macOS; all three installed F9 modes and combined replacement/later rollback passed with 2,027 installed files preserved per fixture. Linux 116 passed with no skips. Archive/credentials 132 passed plus one identical pre-Python plaintext metadata assertion failure. Reviews approved with no Critical/Important findings; scoped static/Bandit checks introduce no production findings. Retained checksum requires paired updates when worker changes. Only the private combined-test outer guard changed 230s to 480s; application deadlines unchanged and failed receipt retained. Source commits 0ffb1fdc3, 76a0f1657 and 98b5d0b24; final follow-up changes are documentation/tracking only. PR remains open against dev; merge conflicts and human-authored Change summary remain merge gates.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed Go from backup source/build/package/runtime and replaced encryption with Python using existing pycryptodomex, preserving age v1 compatibility and recovery behavior. Verified installed macOS flows and 116 Linux tests; updated PR2642 and documented the unchanged native-platform limitations and pre-existing test failure.
<!-- SECTION:FINAL_SUMMARY:END -->
## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->

## Renumbering provenance

Renumbered from TASK-32495 to TASK-32561 on 2026-09-13 to resolve the collision introduced when PR2642 merged dev in `d914a76a21b10cf687eeed5d2bee619bc6b4bb6e`. The already-landed “Hands-free failure honesty: dictation-death exit, degraded copy, entry preflight” task retains TASK-32495; dev assigned that ID in `97041ab370402ba22e75ba214bab75075be11dfb` before this backup record first appeared in `c19068652a38f2407ca885cc8c8c44d9777b4a18`. This follows the landed-task rule in `backlog/docs/lessons-backlog-hygiene.md`.

The fresh sweep covered all reachable task filenames across both local repositories' refs and 23 live worktrees (maximum 32559), then confirmed no TASK-32560/32561/32562 content references across 179 and 224 unique ref tips or live worktrees. Backup-owned semantic references follow the new IDs. Historical receipt paths retain their original names.
