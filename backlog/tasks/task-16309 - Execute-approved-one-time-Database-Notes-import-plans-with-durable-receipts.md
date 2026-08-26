---
id: TASK-16309
title: Execute approved one-time Database Notes import plans with durable receipts
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 16:13'
labels:
  - notes
  - folders
  - import
dependencies:
  - TASK-15705
  - TASK-15706
  - TASK-16230
references:
  - Docs/superpowers/specs/2026-08-12-notes-folder-import-sync-design.md
  - backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md
  - >-
    backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute a user-approved one-time import plan into local Database Notes with bounded, interruption-safe progress and a durable device-private receipt so partial outcomes are honest and failed items can be retried without repeating successful work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 Execution accepts only a fully resolved, explicitly approved immutable plan for local Database Notes; invalid or unresolved plans produce no note, folder, membership, or receipt mutation.
- [x] #2 Approved Create new and Update existing actions persist title, content, keywords, and the approved manual folder memberships while preserving independent content-replacement and membership-addition choices with optimistic note versions.
- [x] #3 Folder creation and note identities are deterministic and collision-safe so retry or restart reconciliation cannot duplicate already committed notes, folders, or memberships.
- [x] #4 A private profile-scoped SQLite ledger durably records session, item, payload, folder, effect, cancellation, failure, and completion states without participating in portable export or centralized database backup.
- [x] #5 Execution is bounded, runs off the UI event loop, reports immutable progress, stops cooperatively between batches, and returns honest imported, updated, skipped, failed, and retryable counts without rolling back confirmed work.
- [x] #6 Retrying a session accepts only the same approved plan, selects only unfinished or retryable work, reconciles crash gaps against current target state, and never repeats a confirmed successful effect.
- [x] #7 Completed single-note receipts provide device-private prior observations for repeat planning while public diagnostics and ordinary logs exclude note content, source paths, fingerprints, note identifiers, raw exceptions, and private receipt fields.
- [x] #8 Focused execution, receipt-schema, private-SQLite policy, idempotency, crash-injection, cancellation, privacy, and affected Notes regression tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no

ADR paths: `backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md`, `backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md`

Reason: ADR-059 already assigns one-time import receipts and provenance to the device-private Notes sync owner, and ADR-073 already fixes optimistic updates, binding privacy, backup exclusion, and interruption semantics. This task implements those accepted boundaries for the local one-time-import executor without adding server authority or lasting-sync behavior.

1. Define approval, execution, progress, outcome, and redacted-diagnostic models with a private canonical plan digest.
2. Add the first migration of the private Notes sync-state SQLite owner for import sessions, folders, payload effects, and repeat observations, including owner-registry and backup-exclusion coverage.
3. Add deterministic local target operations for folder ensure/reuse, note create/read/update, keyword synchronization, and idempotent manual membership attachment.
4. Execute approved plans in bounded batches with per-effect durable transitions, optimistic conflicts, cooperative cancellation, and immutable progress callbacks.
5. Reconcile crash windows and implement failure-only retry plus device-private prior-observation lookup without persisting or logging source content or paths.
6. Run focused and affected regressions, static checks, privacy/backup audits, and closeout review before marking the task Done.

Detailed executable plan: `Docs/superpowers/plans/2026-08-14-task-16309-notes-import-executor.md`
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a local-only, approved-plan executor and a profile-private receipt ledger. The executor validates immutable approval authority, creates deterministic note and folder identities, applies content and membership effects independently with optimistic versions, checkpoints every effect, supports cooperative cancellation and bounded async execution, reconciles crash windows, and retries only unfinished or retryable work. Completed single-note receipts feed private prior observations back into planning without entering public diagnostics, portable export, or centralized backup.

Final adversarial review exposed and fixed two authority-boundary gaps that the earlier green suites missed. Payload fingerprints now bind the exact Python codepoints that execution persists for title, content, keywords, and template name; a composed/decomposed Unicode substitution under the same approval conflicts before any target mutation. Membership attachment now carries the observed note version into the repository and guards that version in the atomic insert, revive, or active-row write; a deterministic interleaving therefore yields `needs_attention` / `version_conflict` without attaching a membership or recording stale completion. The repository's pre-existing no-version API remains compatible. The incident and required test shapes are recorded in `backlog/docs/lessons-testing-evidence.md`.

Verification evidence:

- Exact 11-file Task 6 gate: **1,358 passed, 4 skipped in 63.46s**. Every skip is an explicit native-Windows guard.
- Real file-backed async import/reopen/replay smoke: **1 passed in 0.63s**; the selected root and nested hierarchy, note membership, durable receipt, and duplicate-free replay were verified after reopening both databases.
- Expanded targeted affected execution/planner/receipt/folder rerun after the review fix: **850 passed in 32.24s**. This is supporting evidence distinct from the exact 11-file Task 6 gate above.
- Independent final re-review: **approved with no Critical, Important, or Minor findings**; the reviewer independently ran **13 targeted tests**, Ruff on the seven-file fix, and the diff check.
- Static/structural: all six added Python files pass Ruff and Ruff format; automated `origin/dev` hunk mapping reports **0 Ruff findings on changed lines**. Across the eight surviving modified Python files, whole-file Ruff reports **171 current findings and 171 on `origin/dev` (delta 0)**. Ruff format reports **3 current debt files and the same 3 on `origin/dev` (delta 0)**. These inherited findings were disclosed rather than expanded into unrelated cleanup. All 14 branch-present changed Python files compile, and `git diff --check` passes.
- Privacy/storage: no production logging or raw-exception emission in the import modules, no raw `sqlite3.connect` in the receipt owner or executor, receipt tables store only opaque identifiers/private digests/bounded state, inventory row C49 is exact, and `notes.sync_state` is private-file-only and excluded from backup/export registration.
- Backlog duplicate-ID guard: **1,936 task files, no duplicate IDs**.

ADR required: no new ADR. ADR-059 and ADR-073 already govern device-local receipt ownership, optimistic conflict handling, privacy, backup exclusion, and interruption semantics; the implementation follows and retains both links.

Primary implementation files are `note_import_execution_models.py`, `note_import_receipts.py`, `note_import_executor.py`, `note_import_planner.py`, `note_folder_repository.py`, `private_sqlite.py`, and `config.py`, with focused Notes/private-SQLite tests, the SQLite owner inventory update, and the executable implementation plan. The implementation series runs from `25e720482` through `22c053b2d`; closeout hardening is in `cbdaf09a4` and `a0057cc8b`.
<!-- SECTION:NOTES:END -->

## Definition of Done

<!-- DOD:BEGIN -->
- [x] #1 All acceptance criteria are checked and backed by automated evidence.
- [x] #2 The implementation plan was followed or deviations are documented in Implementation Notes.
- [x] #3 Focused unit and integration tests cover execution, storage, interruption, retry, and privacy behavior.
- [x] #4 Relevant static analysis, formatting, duplicate-task, and diff checks pass.
- [x] #5 ADR-059/073 and affected storage documentation remain accurate and linked.
- [x] #6 The final diff is self-reviewed and receives independent code review.
- [x] #7 Implementation Notes summarize the approach, decisions, exact verification, and modified files.
- [x] #8 Any reusable incident is recorded in the applicable lessons file, or closeout states that none was warranted.
- [x] #9 The task is set to Done only after every item above is complete.
<!-- DOD:END -->
