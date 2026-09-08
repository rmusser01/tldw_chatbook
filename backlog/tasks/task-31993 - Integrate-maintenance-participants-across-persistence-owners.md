---
id: TASK-31993
title: Integrate maintenance participants across persistence owners
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-07 23:53'
updated_date: '2026-09-08 10:44'
labels:
  - backup-recovery
dependencies:
  - task-31978
  - task-31988
  - task-31989
  - task-31990
  - task-31991
  - task-31992
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Every participating persistence owner drains safely and is covered by the shared admission protocol.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every participating persistence owner drains safely and is covered by the shared admission protocol.
- [ ] #2 Unsaved drafts and unfinished cross-store work are neither discarded nor falsely reported captured.
- [ ] #3 Real multi-process evidence proves coherent ownership boundaries and safe resumption without deadlocks.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Establish the uncovered persistent owner behavioral RED regression and exact installed producer census.
2. Bind installed app and headless participants to the reviewed shared admission protocol, including raw writers and shared files; propose coordinator API changes before implementing them.
3. Close new mutation admission, drain actual transactions, handles and cross-store publication, stop watchers at safe boundaries, and preserve dirty editor save/discard decisions.
4. Preserve unresolved/pending work, release normal holds only after proven retirement, and resume in reverse dependency order after exclusive capture release.
5. Verify real two-process coherent DB/assets and config/registry boundaries, timeout, app-close, failures and deadlock ordering.
6. Run required focused participant, SQLite, census and service-composition guards plus scoped static checks and self-review.
7. Record exact evidence, source handoffs and ADR-126 in notes/report; retain In Progress and unchecked criteria pending independent review.
ADR required: yes
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: direct implementation of approved persistence, admission and recovery lifecycle contract.
<!-- SECTION:PLAN:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-02-inventory-admission.md#task-10)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

## Implementation Notes

Foundation phase only; Task10 remains In Progress and all criteria remain unchecked.
Under ADR-126 and controller rulings50–51, added the native pause-hint observation
and moved process lease acquisition/retirement waits outside the coordinator RLock.
Retiring native holds remain explicit; startup is not retired and no producer is
promoted from participant_pending. Added the Participant protocol and conservative
missing-coverage guard. Owner/app/headless integration remains the next phase.

Behavioral RED: first missing-owner assertion failed with DID NOT RAISE; then native
gate and blocked-acquisition lock-order tests failed (2 failed/1 passed). GREEN:
16 focused cases and final79 combined participant/admission/bootstrap cases passed.
Required guards:379 passed/1 skipped (existing Windows-only functional posture),
using the approved Python3.12 interpreter and private pytest caches. Existing
requests and AST SyntaxWarnings remain; no full suite ran. Exact commands, evidence
limits, source handoffs and remaining work are in the task10 execution report.

Files: Backup_Recovery/admission.py, storage_admission.py, new participants.py,
Tests/Backup_Recovery/test_participants.py, owner inventory documentation and this
task. No new ADR: direct implementation of ADR-126, retaining ADR-004 restart-required
config mapping and ADR-036 ownership boundaries. No automatic startup-token release,
GC drain, owner allowlist, output publication or diagnostic inventory refresh.
