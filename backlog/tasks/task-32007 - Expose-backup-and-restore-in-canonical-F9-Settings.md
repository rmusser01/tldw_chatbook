---
id: TASK-32007
title: Expose backup and restore in canonical F9 Settings
status: In Progress
assignee: []
created_date: 2026-09-08 00:01
labels:
- backup-recovery
dependencies:
- task-31978
- task-31999
- task-32004
- task-32005
- task-32006
updated_date: 2026-09-11 09:32
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Users can discover and execute create/inspect/both restore modes from F9 with truthful coverage, risks, and progress.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Users can discover and execute create/inspect/both restore modes from F9 with truthful coverage, risks, and progress.
- [ ] #2 Recovery copies and isolated profiles are inspectable and actionable through the canonical UI.
- [ ] #3 Product-level mounted/live evidence verifies actual services and keyboard/navigation behavior without unintended execution.
<!-- AC:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-05-user-workflows.md#task-24)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Original Task24 implementation release begins from /private/tmp/chatbook-f9-workflow-preflight.md. A owns original six-file canonical F9 screen/state/Settings entry/app composition/test scope, no shell category or deprecated Settings changes. Root owns RecoveryService APIs, P later rollback. Implement qualified current backup/inspection/restore methods and app-owned lifecycle; coordinate missing service presentation contracts rather than duplicate engines or invent success. Actual mounted production navigation/close and scoped verification required before completion.
2026-09-11: C independent current F9 review found actual native pre-safety status actions=('abort',) rendered as Roll back and lacking matching handler. Original Task24 recovery action correction released to A within screen/test scope before refreeze: exact Abort label/action and native actual pending→Abort→preserved originals/cleared own pending proof. No new action engine or scope. Prior sixfile freeze superseded only when this correction lands; Task25 eightfile firstslice independently approved by C, dependencies still pending.
C independent Task24 second finding: editing archive source after inspection preserves old inspection ID/summary; subsequent Review/Confirm can restore old bytes under newly visible source name. Released source-change binding invalidation and two-real-archive UI regression to A. View must require new inspection; retained service work/archive lifetime stays unchanged. Both review findings are original UI correctness, no scope expansion.
2026-09-11: Root exactsnapshot corrected UI13+canonical1 passed23.15s; CLInew34 passed22.15s; actualmountedbackup+census passed, isolatednative succeeded but UI label assertionafterone pilot.pause raced polling. Diagnostic test-only message rerun passed8.26s unchangedproduction. Root will replace only single-pause assertion timing with bounded observation of actual rendered label, no private poll forcing; preserve final assertion and rerunexactcase. No sourcebehavior change.
Committed canonical F9 Settings first slice81f056e52 after independent correction1 approval and root committed-dependency snapshot checks: UI+canonical entry14pass23.15s; actual mounted backup passed; mounted isolated native restore and visible label1pass15.14s after reviewed test-only bounded wait. C timing-only approval /private/tmp/task24-visible-wait-independent-review.md, final Task24 patchSHAce57e1ca6bb0236a827973e36c9a34858599907f5ed51d24db168dd646321129. Abort/source-change review fixes included. Ruff485→485/Bandit36→36 baseline no new findings; final affected correction4files Ruff0/Bandit0. Task remains In Progress for original actual child-open receipt and owner-state/inert integration.
Root original Settings inert-extraction followup: expose verified dependency-group IDs/members/completeness in existing RecoveryService.summary; screen offers explicit group selection/absent manual destination, worker preview then exact-plan confirmation and inert-only result label. Reuse source-change/view-revision invalidation; no schema execution/engine changes. Own recovery_service.py, backup_restore_screen.py, new Tests/UI/test_backup_inert_extraction.py. Actual mounted widget/service byte checks, stale-review and no-overwrite required. Child-open/owner-summary remains separate A followup; no app.py edits here.
Inert extraction UI completed as original selected unsupported-group manual extraction: actual verified dependency-group choices, reviewed absolute absent destination, invalidation on source/destination changes, explicit extraction confirmation, and actual engine status. Independent review /private/tmp/ui-inert-extraction-independent-review.md approved with no findings. Exact production snapshot plus 3 frozen UI/service files: all 19 focused new/existing UI and real ProductionApp composition tests passed45.14s (/private/tmp/ui-inert-extraction-root-index.log). New behavior red3 missing UI, green3pass3.92s; scoped Ruff0/Bandit0. Late worker callbacks source-reviewed for revision/inspection/mounted identity; dynamic late-preview race not separately tested. Does not claim restored/opened activation.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->