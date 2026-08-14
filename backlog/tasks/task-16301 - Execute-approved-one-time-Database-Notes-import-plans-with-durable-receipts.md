---
id: TASK-16301
title: Execute approved one-time Database Notes import plans with durable receipts
status: In Progress
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
    backlog/decisions/060-notes-sync-round-trip-and-interoperability-constraints.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute a user-approved one-time import plan into local Database Notes with bounded, interruption-safe progress and a durable device-private receipt so partial outcomes are honest and failed items can be retried without repeating successful work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Execution accepts only a fully resolved, explicitly approved immutable plan for local Database Notes; invalid or unresolved plans produce no note, folder, membership, or receipt mutation.
- [ ] #2 Approved Create new and Update existing actions persist title, content, keywords, and the approved manual folder memberships while preserving independent content-replacement and membership-addition choices with optimistic note versions.
- [ ] #3 Folder creation and note identities are deterministic and collision-safe so retry or restart reconciliation cannot duplicate already committed notes, folders, or memberships.
- [ ] #4 A private profile-scoped SQLite ledger durably records session, item, payload, folder, effect, cancellation, failure, and completion states without participating in portable export or centralized database backup.
- [ ] #5 Execution is bounded, runs off the UI event loop, reports immutable progress, stops cooperatively between batches, and returns honest imported, updated, skipped, failed, and retryable counts without rolling back confirmed work.
- [ ] #6 Retrying a session accepts only the same approved plan, selects only unfinished or retryable work, reconciles crash gaps against current target state, and never repeats a confirmed successful effect.
- [ ] #7 Completed single-note receipts provide device-private prior observations for repeat planning while public diagnostics and ordinary logs exclude note content, source paths, fingerprints, note identifiers, raw exceptions, and private receipt fields.
- [ ] #8 Focused execution, receipt-schema, private-SQLite policy, idempotency, crash-injection, cancellation, privacy, and affected Notes regression tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no

ADR paths: `backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md`, `backlog/decisions/060-notes-sync-round-trip-and-interoperability-constraints.md`

Reason: ADR-059 already assigns one-time import receipts and provenance to the device-private Notes sync owner, and ADR-060 already fixes optimistic updates, binding privacy, backup exclusion, and interruption semantics. This task implements those accepted boundaries for the local one-time-import executor without adding server authority or lasting-sync behavior.

1. Define approval, execution, progress, outcome, and redacted-diagnostic models with a private canonical plan digest.
2. Add the first migration of the private Notes sync-state SQLite owner for import sessions, folders, payload effects, and repeat observations, including owner-registry and backup-exclusion coverage.
3. Add deterministic local target operations for folder ensure/reuse, note create/read/update, keyword synchronization, and idempotent manual membership attachment.
4. Execute approved plans in bounded batches with per-effect durable transitions, optimistic conflicts, cooperative cancellation, and immutable progress callbacks.
5. Reconcile crash windows and implement failure-only retry plus device-private prior-observation lookup without persisting or logging source content or paths.
6. Run focused and affected regressions, static checks, privacy/backup audits, and closeout review before marking the task Done.

Detailed executable plan: `Docs/superpowers/plans/2026-08-14-task-16301-notes-import-executor.md`
<!-- SECTION:PLAN:END -->

## Definition of Done

<!-- DOD:BEGIN -->
- [ ] #1 All acceptance criteria are checked and backed by automated evidence.
- [ ] #2 The implementation plan was followed or deviations are documented in Implementation Notes.
- [ ] #3 Focused unit and integration tests cover execution, storage, interruption, retry, and privacy behavior.
- [ ] #4 Relevant static analysis, formatting, duplicate-task, and diff checks pass.
- [ ] #5 ADR-059/060 and affected storage documentation remain accurate and linked.
- [ ] #6 The final diff is self-reviewed and receives independent code review.
- [ ] #7 Implementation Notes summarize the approach, decisions, exact verification, and modified files.
- [ ] #8 Any reusable incident is recorded in the applicable lessons file, or closeout states that none was warranted.
- [ ] #9 The task is set to Done only after every item above is complete.
<!-- DOD:END -->
