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

## Open product evidence

- The saved-state later-rollback preview originally refused with `local_snapshot_absence_unclassified`. A bounded created-destination observer now reaches the actual UI review: child exit0 in7.14s, `/private/tmp/later-created-saved-ui-preview.log`. This used the saved F9 extraction with only the observer update; it is preview-only evidence. No confirmation, target mutation or new safety copy occurred. Automatic approval review blocked the proposed execution-validation dispatch changes pending exact user authorization. Full later-rollback qualification remains open.
- Current-profile setup reporting is committed in `3ab0e5ac2`. Twelve new cases and four affected cases passed on committed `31e0034d2` plus the exact three-file change, including actual replacement/later generations, a fresh-process read and damaged-evidence refusal. Individual owner controls have separate receipts in their Backlog tasks; these do not establish the missing full product sequence.
- The specified seven-gate conjunction regression is committed in `b039a94f3`; it is not yet wired to product availability and supplies no native evidence. The packaging checklist update is committed in `7bff17e89`, with its qualification items intentionally unchecked.
- The original all-owner multi-profile synthetic fixture, complete create/inspect/isolated-open/replace/later-rollback sequence, native crash/multiple-volume matrix, product release-gate wiring and qualification CI remain Task26 requirements. Run named feature and inventory/lifecycle checks only; no full-suite run is implied.

Update this ledger with exact revision, environment, command and actual result when each gate is demonstrated. Keep Complete capture and replacement qualification separate. A generic boolean conjunction test establishes wiring only; it cannot substitute for owner, archive, native or product evidence.
