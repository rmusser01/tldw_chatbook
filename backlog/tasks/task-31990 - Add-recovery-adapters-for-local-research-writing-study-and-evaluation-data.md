---
id: TASK-31990
title: Add recovery adapters for local research writing study and evaluation data
status: Done
assignee:
  - Codex
created_date: '2026-09-07 23:51'
updated_date: '2026-09-08 07:44'
labels:
  - backup-recovery
dependencies:
  - task-31978
  - task-31986
  - task-31987
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Inactive optional features retain their existing local durable data in baseline inventory.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Inactive optional features retain their existing local durable data in baseline inventory.
- [x] #2 All census owners in this cohort have lossless data/asset capture, checked schemas, and relocation evidence.
- [x] #3 Discovery and validation never start evaluations, models, or server requests.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership contract; reuse ADR-126.

1. Add the focused import-light discovery regression, establish importability and behavioral RED.
2. Trace the exact research, writing, shared study/quiz and evaluation producer census, including secondary benches, custom paths and persisted outputs; distinguish external exports and server mirrors.
3. Implement installed declarations and checked capture under fixed maintenance authority, preserving all records and referenced bytes. Qualify exact installed schemas and a real supported historical schema without constructor execution during discovery/validation.
4. Add real persistence fixtures, lossless capture/relocation and adversarial schema/asset/custom-path evidence.
5. Run the domain tests, targeted private-SQLite and architecture inventory guards; update exact census rows.
6. Run scoped lint/format and diff checks, self-review and document actual evidence/limits. Commit only this scope; leave AC unchecked/In Progress pending controller independent review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the installed research, writing, shared study/quiz, EvalsDB and evaluation-definition recovery cohort under [ADR-126](../decisions/126-complete-local-backup-and-recovery.md). Whole SQLite snapshots preserve records, history, relationships, soft deletions and managed BLOB bytes. Exact current schema catalogs include a genuine pre-lease research v0 fixture from commit 4f535bd3c4afd07565eb17b1ffa335c2c274c17a and its installed transition to v1; writing remains its actual unversioned layout, and EvalsDB is v5.

Controller-approved prerequisites add stable shared-group composition for the exact core/study/quiz cohort, operation-private bounded native file copy/read for the managed evaluation YAML, and lazy package exports with unchanged public identities. Fixed authority, selected namespaces plus bootstrap.unbound, local binding proof, default-only SQLite capture factories and native resource retirement remain required. Research/writing file-operation contexts now close after commit/rollback, fixing an observed deferred-checkpoint source-header change; memory connections retain their lifetime. The evaluation config source and recovery declaration share one import-light canonical resolver.

The two owner inventories document exact connection/copy/source sites and every evaluation producer disposition. Word benches and character probes persist through EvalsDB, including run snapshots, conversation replies, annotations and review state; caller-selected exports/inputs and server-owned data are classified separately. Study/quiz semantic validators check actual BLOB bytes, complete asset identifiers and historical question/answer references. Selected-profile local character dependencies cannot resolve against another profile.

Final targeted evidence (read-only project Python 3.12.11; no full sweep): required private-SQLite modules 328 passed/one existing Windows posture skip; core/admission/bootstrap/inventory 157 passed; domain/architecture/Research/Writing covering command 96 passed; final focused domain file 57 passed after the final selector and complete-asset-identifier refinements. Scoped ruff error checks, new-module formatting and git diff-check pass. Existing RequestsDependencyWarning and source-parse invalid-escape warnings are retained. Full commands, outputs, RED/GREEN and remaining downstream limits are in `.superpowers/sdd/2026-09-07-complete-local-backup-restore/task-7-report.md`.

Independent spec and quality review passed after one fix round; all acceptance criteria are verified. Other historical/physical schema variants, sealed candidate/schema budgets, actual startup drain, archive metadata/credential processing, activation and publication/replacement remain their named later tasks. Native FD-close-failure quarantine has code review and simulated-boundary evidence with real descriptors and process observers; actual OS/hardware fault qualification remains outstanding. No user data, keyring, server/model execution, shared environment changes, full suite, subagents, remote push, merge or publication was used.

### Review fix round 1 plan

Under the existing ADR-126 and controller-approved narrow scope, reproduce the raw parent-directory retirement gap with child-local simulated close failures and independent native admission observers. Cover source/destination copy parents plus captured/ordinary reads, retain ambiguous native retirement without descriptor-number retry, then consolidate repeated SQLite schema and capture guards behind small internal helpers while preserving literal qualified open/copy producers. Run the focused domain, affected admission/ordinary-owner and exact census guards, document RED/GREEN and limitations, and commit this review fix without completing the task before independent review.

### Review fix round 1 notes

Corrected raw-file parent retirement by passing an optional private close callback through bootstrap pinning and the existing private-path walker/symlink transition. Actual close uncertainty now latches quarantine before cleanup can retry a descriptor number; normal missing/refused reads release admission. Six child-local simulated failures use real directory FDs and independent maintenance observers, including source/destination parents and ordinary reads. Consolidated the three SQLite owners' identical schema and capture guards into `DB/recovery_sqlite.py`, retaining all literal private authority sites and census rows.

Final covering evidence: 180 domain/path/bootstrap tests passed, 53 admission/ordinary-owner tests passed, and 38 exact census tests passed. Scoped lint/format and diff checks pass. Three existing path-test failures were reproduced at the fix base and resolved by isolating their fault injections to the private-path module while retaining real ordinary admission; the incident is documented in the testing lessons. Full RED/GREEN commands and simulated-failure limitations are appended to the Task 7 report. No actual OS/hardware fault qualification is claimed. Status and ACs remain pending independent review.
<!-- SECTION:NOTES:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-02-inventory-admission.md#task-7)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.
