---
id: TASK-19003
title: Ship the Import once Notes workflow
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-20 07:40'
updated_date: '2026-08-20 22:49'
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Implement and test the frozen/redacted import workflow state and bounded paging model.\n2. Add the SQLite-enforced read-only receipt observation seam and update the single notes.sync_state private-owner inventory/policy without schema mutation.\n3. Build the render/message-only import canvas and focused controller with named late-bound planner/executor dependencies.\n4. Integrate the retained local Library Notes route, selection/review/execution/receipt lifecycle, focus/compact behavior, refresh, cancellation/retry, and legacy Sync coexistence.\n5. Run pure/UI/backend/private-SQLite/shell gates, update the Notes guide, perform spec/quality review, and close with exact evidence.\n\nADR required: no new ADR\nADR path: backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md; backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md\nReason: TASK-16230 and TASK-16309 already define the planner, executor, approval, and private receipt boundaries; this task exposes them through production UI and adds a read-only pre-approval lookup.\n\nPlan: Docs/superpowers/plans/2026-08-20-notes-import-once-ui.md
<!-- SECTION:PLAN:END -->
