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
updated_date: 2026-09-10 15:35
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
<!-- SECTION:IMPLEMENTATION_NOTES:END -->