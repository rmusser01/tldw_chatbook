---
id: TASK-31990
title: Add recovery adapters for local research writing study and evaluation data
status: In Progress
assignee:
  - Codex
created_date: '2026-09-07 23:51'
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
- [ ] #1 Inactive optional features retain their existing local durable data in baseline inventory.
- [ ] #2 All census owners in this cohort have lossless data/asset capture, checked schemas, and relocation evidence.
- [ ] #3 Discovery and validation never start evaluations, models, or server requests.
<!-- AC:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-02-inventory-admission.md#task-7)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.

## Implementation Plan

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership contract; reuse ADR-126.

1. Add the focused import-light discovery regression, establish importability and behavioral RED.
2. Trace the exact research, writing, shared study/quiz and evaluation producer census, including secondary benches, custom paths and persisted outputs; distinguish external exports and server mirrors.
3. Implement installed declarations and checked capture under fixed maintenance authority, preserving all records and referenced bytes. Qualify exact installed schemas and a real supported historical schema without constructor execution during discovery/validation.
4. Add real persistence fixtures, lossless capture/relocation and adversarial schema/asset/custom-path evidence.
5. Run the domain tests, targeted private-SQLite and architecture inventory guards; update exact census rows.
6. Run scoped lint/format and diff checks, self-review and document actual evidence/limits. Commit only this scope; leave AC unchecked/In Progress pending controller independent review.

## Implementation Notes

Implemented the installed research, writing, shared study/quiz, EvalsDB and evaluation-definition recovery cohort under [ADR-126](../decisions/126-complete-local-backup-and-recovery.md). Whole SQLite snapshots preserve records, history, relationships, soft deletions and managed BLOB bytes. Exact current schema catalogs include a genuine pre-lease research v0 fixture from commit 4f535bd3c4afd07565eb17b1ffa335c2c274c17a and its installed transition to v1; writing remains its actual unversioned layout, and EvalsDB is v5.

Controller-approved prerequisites add stable shared-group composition for the exact core/study/quiz cohort, operation-private bounded native file copy/read for the managed evaluation YAML, and lazy package exports with unchanged public identities. Fixed authority, selected namespaces plus bootstrap.unbound, local binding proof, default-only SQLite capture factories and native resource retirement remain required. Research/writing file-operation contexts now close after commit/rollback, fixing an observed deferred-checkpoint source-header change; memory connections retain their lifetime. The evaluation config source and recovery declaration share one import-light canonical resolver.

The two owner inventories document exact connection/copy/source sites and every evaluation producer disposition. Word benches and character probes persist through EvalsDB, including run snapshots, conversation replies, annotations and review state; caller-selected exports/inputs and server-owned data are classified separately. Study/quiz semantic validators check actual BLOB bytes, complete asset identifiers and historical question/answer references. Selected-profile local character dependencies cannot resolve against another profile.

Final targeted evidence (read-only project Python 3.12.11; no full sweep): required private-SQLite modules 328 passed/one existing Windows posture skip; core/admission/bootstrap/inventory 157 passed; domain/architecture/Research/Writing covering command 96 passed; final focused domain file 57 passed after the final selector and complete-asset-identifier refinements. Scoped ruff error checks, new-module formatting and git diff-check pass. Existing RequestsDependencyWarning and source-parse invalid-escape warnings are retained. Full commands, outputs, RED/GREEN and remaining downstream limits are in `.superpowers/sdd/2026-09-07-complete-local-backup-restore/task-7-report.md`.

ACs remain unchecked and status remains In Progress for controller independent review. Other historical/physical schema variants, sealed candidate/schema budgets, actual startup drain, archive metadata/credential processing, activation and publication/replacement remain their named later tasks. Hard native FD-close-failure quarantine is conservative code-reviewed behavior, not runtime fault-injected release qualification. No user data, keyring, server/model execution, shared environment changes, full suite, subagents, remote push, merge or publication was used.
