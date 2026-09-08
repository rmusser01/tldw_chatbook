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

### Phase2 implementation plan (before code)

1. Establish behavioral RED for Event/Sync file connection native retirement and private local pause gate/pending acquisition refusal.
2. Add reservation before path/authority lookup, opaque process-local pause and live installed repository operation provenance (PID, actual Thread/task identity, exact path/native scope); track actual SQLite resource retirement and same-holder probes.
3. Integrate Event/Sync transaction scopes with finally-native-close on their creating thread; retain in-memory connection semantics. Never cancel writes as a drain mechanism.
4. Verify real native SQLite/operation lifetimes, late/cancelled acquisition, descendant confinement, task/thread and stale capability refusal, close failures, and incomplete runtime-coverage refusal with startup retained.
5. Run affected focused guards/static checks; self-review and commit this bounded phase. Preserve foundation evidence, In Progress status and unchecked ACs for whole Task10 review.
ADR required: yes
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: direct implementation of approved lifecycle authority under controller rulings50–52. No new ADR; no startup-retirement success or production responder until later runtime cohorts.

### Phase2 implementation notes

Under ADR-126 and controller rulings52–53, implemented the private process-local
pause gate, pre-root/bootstrap/authority acquisition accounting, live repository
operation provenance and actual SQLite resource metadata/retirement. Event/Sync
transactions now commit/rollback and finally-close file connections on the creating
thread, retaining memory behavior. Ordinary custom constructor allocation and
close/reinitialize failures retain exclusion conservatively. No startup lease is
retired, no responder is installed, and runtime coverage remains incomplete.

Evidence:32 focused cases pass, including real worker SQLite lifetimes and separate
native maintenance observer processes.145 earlier affected admission/bootstrap/
repository guards passed. Final required SQLite/census/service guards passed in a
combined run (469 passed/1 existing Windows skip overall), whose additional core
capture checks had1 failure/2 setup errors. Core alone passes62; the three failing
cases alone pass3 and reduced service-plus-three ordering passes7. The exact combined
command on immutable clean phase BASE0f4ff4ba reproduces the failure family (468
passed/1 skip/1 failure/4 errors), establishing pre-phase2 app callback/bootstrap
concurrency debt for the remaining Task10 lifecycle cohort. BASE's unexpected RAG
metadata network attempts were blocked by Tests/network_guard; no download or
network permission change occurred. Do not claim the combined-order run is green.

Scoped ruff fatal checks, new-module formatting and diff checks pass. Self-review
covered actual operation/handle accounting, cancellation/retirement, native scope,
constructor and thread-affinity behavior, memory/rollback semantics and coverage
limits. Source syntax counts are unchanged; inventory guards pass without source
row changes. Updated owner inventory and the evidence lesson from the actual
wrong-thread closed-handle assertion incident. Detailed APIs, exact commands,
source handoffs and remaining cohorts are in task-10-phase2-report.md. Task31993
stays In Progress with all acceptance criteria unchecked for whole-task review.

Final self-review added three behavioral RED cases for forged/instance-shadowed
validation callbacks, then enforced exact token type and direct class validation
for operation/pause identity. The final32 focused cases and scoped static checks
pass; no caller-supplied validation callback can confer authority.
