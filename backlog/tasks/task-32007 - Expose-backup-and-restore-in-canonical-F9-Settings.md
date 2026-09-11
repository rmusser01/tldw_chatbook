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
updated_date: 2026-09-11 10:02
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
Original default-profile scope UI integration released to root, linked Task31986 default selector correction: canonical F9 backup must include known profiles plus explicit Add profile additions. Thread explicit include_known_profiles keyword through only backup preview/details/start, preserve exact target inventory API default. Display actual reviewed config-owner paths and retain recollection on capture so known-selector drift requires new review. Minimal host/inspect remains startup-independent. Focused actual service UI regression before code.
Canonical F9 and minimal recovery host now default to all known profiles plus manually added configurations. Actual discovered config-owner paths are displayed; same policy retained through preview/capture. Actual added known profile after review causes review_required/scope_changed without output. Independent UI review /private/tmp/ui-known-profiles-independent-review.md and root combined review /private/tmp/chatbook-known-profiles-root-review.md. Behavioral old explicit-subset red1failed0.88s; corrected UI1passed1.40s. All17 existing/new UI+mounted cases in combined snapshot passed (overall25pass1collector fixture setup error, since corrected). Minimal-host and collector10passed5.75s; census11passed. Scoped Ruff0/Bandit0; app same35 existing Bandit findings. Task remains In Progress for remaining full replacement/opened/owner presentation release evidence.
Original Needs setup presentation slice released to root: reuse isolated_restore's exact verified launch descriptor/generation witness (preserve existing _launch_descriptor return API), project required and still-unapproved owner requirements through service.profiles and existing profile list. Missing/damaged verified evidence stays unknown/recovery_required, never an empty ready set. Recheck actual paired generation after owner reads. No approvals/owner constructors/provider probes/process-open receipt claims; those remain separate owner/child work. Tests use actual isolated restore and durable activation records, then mounted presentation.
Required-owner presentation completed: exact existing launch verifier extracted unchanged into _launch_state; old descriptor API preserved. Service profile rows expose actual checked generation/required/pending owners, rechecking pair after reads; missing/damaged evidence is None/unchecked and launch button disabled, never empty-ready. UI shows Needs setup or owner reviews complete without claiming optional capability readiness/opened success. Independent approval /private/tmp/profile-requirements-independent-review.md (AST verifier equivalence). Exact snapshot summary red3 KeyError2.43s -> green3pass3.12s; actual mounted list red3 missing text -> green3pass3.66s. Final exact snapshot new3 +existing isolated3 +UI13+census11:30passed28.90s (/private/tmp/profile-requirements-root-index.log). Root Ruff0, new test formatted, scoped Bandit0. Early mixed feature attempts failed in concurrent publication work and are not qualification; controlled snapshot receipts above exclude pending A/P changes. Child-mounted open acknowledgement remains separate incomplete work.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->