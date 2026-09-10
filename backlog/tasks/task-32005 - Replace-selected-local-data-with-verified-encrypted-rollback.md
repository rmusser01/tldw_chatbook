---
id: TASK-32005
title: Replace selected local data with verified encrypted rollback
status: In Progress
assignee: []
created_date: 2026-09-08 00:00
labels:
- backup-recovery
dependencies:
- task-31978
- task-31993
- task-31997
- task-31999
- task-32000
- task-32001
- task-32002
- task-32003
- task-32004
updated_date: 2026-09-10 22:17
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Replacement cannot mutate live data before exact affected stored data and supported credentials have a verified encrypted rollback copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Replacement cannot mutate live data before exact affected stored data and supported credentials have a verified encrypted rollback copy.
- [ ] #2 Maintenance spans the entire safety-copy/publication/validation interval and final active inventory matches the approved generation.
- [ ] #3 Interrupted or failed replacement retains recovery evidence and never boots ambiguous or automatically active state.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute original component04 Task22 incrementally. First qualify source-preserving native-held SQLite snapshots (read-only SQLite can mutate live SHM): exact main/WAL private materialization, installed owner snapshot and limits/resource checks. Then bind reviewed affected originals including classified sidecars and damaged config, keep one native session through supported credential capture/encryption/authenticated verification, publication/installed validation/activation/commit. Add focused actual SQLite/WAL and interruption/cancellation/failure tests; do not claim replacement complete from prerequisite slices. Activation integration remains separately approval-blocked.
<!-- SECTION:PLAN:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-04-restore-recovery.md#task-22)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Read-only original Task22 prerequisite analysis completed: /private/tmp/chatbook-sqlite-rollback-source-plan.md. Existing held capture cannot directly handle damaged config binding or coherent raw config credential omissions; public path-only rollback verifier correctly refuses unproven SQLite snapshots. Root native SQLite3.49.1 private fixture (abrupt child leaves committed WAL, read-only connection backup to disposable file) copied both main and WAL-only rows but changed original -shm bytes/timestamps. Exact observation /private/tmp/chatbook-rollback-wal-observation.json, disposable fixture /private/tmp/chatbook-rollback-wal-observe-jwsd4qz7. This is diagnostic evidence, not owner/native-session qualification. Therefore replacement safety capture must materialize stable source main+WAL privately before native SQLite reads, preserving reviewed physical originals; do not silently refresh target fingerprints. No Task22 production edits or completion claim.
Original Tasks14/22 SQLite source-preservation prerequisite authorized: add capture-only private main/WAL materialization using real active MaintenanceSession source identity and existing stage, exact installed SQLite owner policy, bounded pinned copying, hot-journal refusal, cancellation, before/after source checks and positive native retirement. No live SQLite read before materialization; preserve main/WAL/SHM and original fingerprints. Propagate existing reviewed ArchiveLimits/byte_budget from held capture, enforce cumulative/member/space bounds, leave ordinary noncapture DB behavior unchanged. Actual installed owner/WAL TDD and focused limits/path/cancel evidence required. No activation, damaged-config binding, replacement executor or semantic rollback verification authority in this slice; retain inherited storage traversal edits.
SQLite source-preservation prerequisite: materialize only native-session-qualified main/WAL into bounded private staging so read-only SQLite never alters original SHM. Bind private/source identities through reuse and native open/backup; quarantine ambiguous descriptor/native connection retirement. Final focused cohort: 57 passed (44 materialization, 8 existing core/domain, 2 capture factory refusals, 3 ordinary factory/copy-close controls); earlier public capture variants 3 passed. Production Ruff 53→53 and Bandit 7→7 existing findings, no new production findings. Exact inventory delta is only three _CaptureScope.sqlite_target rows (mkdir2/open2/write1). Round-two independent review pending. Task remains In Progress: this does not implement replacement execution, rollback receipts, activation association, or fence clearing. Evidence: /private/tmp/chatbook-capture-sqlite-report.md and /private/tmp/chatbook-capture-sqlite-independent-review.md.
Final bounded independent re-review APPROVED the SQLite prerequisite; all three review findings addressed. Original main/WAL/SHM preservation, cached private identity binding, native/descriptor quarantine and short-read behavior qualified. Root is staging only this slice, preserving inherited TTS/traversal/activation work; exact staged ownership inventory verification follows.
Root final gate: unchanged ownership inventory guard passed 11 tests in the existing isolated snapshot with this slice's exact index source/doc updates; staged diff check clean. Only the three new census rows and isolated source-preservation code/test/task notes are staged. Inherited storage traversal (100 lines) and TTS census rows remain unstaged. Independent review approved; 57 focused tests and unchanged production security/static baseline recorded above. This prerequisite is ready to commit; task remains In Progress.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->