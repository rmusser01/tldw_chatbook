---
id: TASK-31994
title: Persist recovered media references and deletion tombstones
status: In Progress
assignee: []
created_date: 2026-09-07 23:53
labels:
- backup-recovery
dependencies:
- task-31978
- task-31986
- task-31987
- task-31992
- task-31993
updated_date: 2026-09-11 07:28
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Included temporary media becomes durable and resolves correctly across restart and subsequent backups.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Included temporary media becomes durable and resolves correctly across restart and subsequent backups.
- [x] #2 Intentional deletion survives as a tombstone without partial-backup status; unexpected missing required bytes still block completeness.
- [x] #3 Shared references, recovery holds, explicit cleanup, and interrupted publication/deletion preserve recoverable state.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implement original component03 Task11: (1–2) establish behavioral red recovered media tombstone test; (3) private profile catalog v1 with identity/digest/reference/tombstone/recovery state; (4) publish verified regular payload before durable references using existing private primitives; (5) resolve persisted references before temporary image/video lookup; (6) durable intentional deletion with shared-reference/recovery-hold checks; (7) baseline owner registration, collision/interruption/corruption/restart tests; (8) installed mutation admission and drain integration; (9–11) focused named guards, scoped lint/Bandit, spec/code review, documentation, verified task-only commit. Task10 runtime composition remains in progress independently; do not qualify complete capture or finalize this dependency until integration is demonstrated.
<!-- SECTION:PLAN:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-03-capture-archives.md#task-11)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Resuming original Task11 implementation in parallel with Task10 application composition. Reuse existing owner/native primitives. No new user feature or expanded backup scope; no completion claim from catalog tests alone.
Root rereview accepted pending-destination collision and consistent deleted-state journal fixes; actual Console image consumer now resolves metadata and bounded payload/decode off-loop, stable prepared content avoids reread, decode failure explicit, retained native work drain hooks. Corrupt-catalog sqlite3 errors contained with explicitmissing/no fallback. Final combined owner+restrictedvalidator verification running; Task20 source-profile binding and app coordinator still required.
Scoped implementation source review approved after worker/corrupt-catalog corrections. Combined recovered owner + restricted validator83passed14.10s /private/tmp/chatbook-recovered-sqlite-combined-final.log; prior imagecache27pass and added actualconsumer/adversarial tests documented /private/tmp/chatbook-recovered-media-report.md. New production modules noBanditfindings; touchedconsumer baseline unchanged. Committing working owner/consumer code together with Task13 sameconnection validation, excluding pre-existing TTS/admission/docs changes. Task remains InProgress: Task10 app composition and Task20 restoredsource profile binding still required before complete backup workflow claim.
Resume original recovered-media destination lookup requirement. Root private actual-owner relocation probe /private/tmp/recovered-media-relocation-probe.py and log confirms ready and intentionally deleted assets remain resolvable under source profile but become unknown under actual destination config selector-derived profile. Release bounded owner fix/tests to runtime_final_capture: recovered_media.py and new exact relocation test only; preserve source refs/stable IDs and validated tombstones, add only source-profile to locally mapped destination-profile aliases transactionally, conflicting alias refuses. Existing selected config dependency should supply mapping; report any missing mapping/staging prerequisite before expanding. No new schema/registry, no initial temporary-media capture implementation in this slice.
Root independent review APPROVED frozen recovered-media relocation two-file patch05e11b77ef287ebe6803003dde8b399c67aaac5d24f37bf3719a5b3957226175. Exact role/config dependency/mapped path; source-only aliasing and collision rollback; valid deleted tombstones updated with aliases in same transaction; stable IDs/payloads unchanged; sequential relocation/native recapture/real private-app lookup proof. Report /private/tmp/recovered-media-relocation-report.md. 14new+4affected covered across focused receipts, production Ruff/Bandit0;22B101 new test assertions reviewed. Exact-index census and scoped commit next; temporary-media conversion remains original unfinished work.
Final exact-index census11passed8.16s (/private/tmp/recovered-relocation-index-census-final.log). Added only _RecoveredAdapter.relocate_restore/connect_private_sqlite count1 classified qualified/recovered.media; independent C census-only review approved. Source independently approved by root (/private/tmp/recovered-media-relocation-independent-review.md). Commit exact frozen2files + census row + task.
Original temporary-media capture integration released from /private/tmp/chatbook-temporary-media-integration-proposal.md. Capture-time private recovered.media catalog/payload materialization, original sourceinventory remains authority; reuse existing schema, stable asset IDs/deletedtombstones. Referenced available/missing video counts; ordinary expired optional refs honestly reported without fabricated assets and do not alone preventComplete. Gallery bytes lacking persisted transcript identity may be retained as unreferenced assets, no inventedmessageIDs; storage controls needed peroriginal recovered lifecycle. Native accepted Console generation/adoption settlement proof prerequisite underway; no UI lifetime edit without actualred. Exact files initial recovered_media.py, DB/recovery_core.py, Video_Generation/video_metadata.py and only needed purepath factor video_store.py, focusedtemporarycapturetests; config_adapter/capture shared edits wait Chatbook correction freeze. No new archiveformat/schema/publication/recoveryauthority.
Root independent source review approved frozen seven-file temporary-media integration (956b2711904b5c51ba34f0a141b8ba9e34b65f3dc38c9a21a954a4d7bb4491e9). Read full 1571-line patch and actual capture/manifest/native recovered owner interfaces. Existing catalog/deleted refs preserved, private generated payload before refs, original source inventory retained. Restore-only restricted candidate connector/authorizer fixes private alias validation boundary; no ordinary source guards changed. Named56 cases passed across55 owner/affected62.04s + actual native restore/reopen/partial secondcapture42.02s. No new Ruff/Bandit. Whole Complete rebackup remains explicitly unqualified due inert retained eval.definitions discovery gap (Task20/26), not silently excluded. Exact-index census pending.
Root independent review approved frozen temporary7 patch956b2711904b5c51ba34f0a141b8ba9e34b65f3dc38c9a21a954a4d7bb4491e9. C independently approved root census delta: removed2 obsolete native-connect rows, added7 exact temporary materializer/reference producers. Scoped owner55 and real roundtrip1 pass receipts retained; second backup explicitly Partial due unsupported retained eval config (no Complete rebackup claim). Exact staged-production census11 passed; no broader staging.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->