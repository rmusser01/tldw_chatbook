---
id: TASK-19003
title: Ship the Import once Notes workflow
status: To Do
assignee: []
created_date: '2026-08-20 07:40'
labels:
  - notes
  - import
  - ux
dependencies:
  - TASK-16230
  - TASK-16309
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose the existing immutable Notes import planner and executor through a production Library review, progress, and durable receipt flow while the legacy Sync entry remains operable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Users can select one or more files or one folder; selected files require an existing-or-new destination path that is not created before approval.
- [ ] #2 Review shows every planner classification, collision, confirmation, content-replacement choice, and membership choice before approval.
- [ ] #3 Discovery, planning, prior-observation lookup, collision review, and cancellation create no note, folder, receipt, configuration, or private SQLite schema mutation.
- [ ] #4 Execution receives the exact final immutable approved plan and reports bounded progress, cooperative cancellation, partial completion, and retryable failures truthfully.
- [ ] #5 The latest import progress and receipt remain revisitable across Library navigation for the current application session; cross-process receipt browsing is not implied.
- [ ] #6 Import once is available only for local Library Notes and refreshes the note list and folder tree after settlement.
- [ ] #7 The retained canvas, focus restoration, and 60-column compact layout remain usable, and legacy Sync stays visible and operable.
<!-- AC:END -->

## Decision Record Check

ADR required: no new ADR
ADR paths: `backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md`, `backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md`
Reason: TASK-16230 and TASK-16309 already implement the accepted planner, executor, and private receipt boundaries; this task supplies their production UI and a read-only pre-approval lookup.
