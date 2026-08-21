---
id: TASK-19003
title: Ship the Import once Notes workflow
status: Done
assignee:
  - '@codex'
created_date: '2026-08-20 07:40'
updated_date: '2026-08-21 01:24'
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
- [x] #1 Users can select one or more files or one folder; selected files require an existing-or-new destination path that is not created before approval.
- [x] #2 Review shows every planner classification, collision, confirmation, content-replacement choice, and membership choice before approval.
- [x] #3 Discovery, planning, prior-observation lookup, collision review, and cancellation create no note, folder, receipt, configuration, or private SQLite schema mutation.
- [x] #4 Execution receives the exact final immutable approved plan and reports bounded progress, cooperative cancellation, partial completion, and retryable failures truthfully.
- [x] #5 The latest import progress and receipt remain revisitable across Library navigation for the current application session; cross-process receipt browsing is not implied.
- [x] #6 Import once is available only for local Library Notes and refreshes the note list and folder tree after settlement.
- [x] #7 The retained canvas, focus restoration, and 60-column compact layout remain usable, and legacy Sync stays visible and operable.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Implement and test the frozen/redacted import workflow state and bounded paging model.\n2. Add the SQLite-enforced read-only receipt observation seam and update the single notes.sync_state private-owner inventory/policy without schema mutation.\n3. Build the render/message-only import canvas and focused controller with named late-bound planner/executor dependencies.\n4. Integrate the retained local Library Notes route, selection/review/execution/receipt lifecycle, focus/compact behavior, refresh, cancellation/retry, and legacy Sync coexistence.\n5. Run pure/UI/backend/private-SQLite/shell gates, update the Notes guide, perform spec/quality review, and close with exact evidence.\n\nADR required: no new ADR\nADR path: backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md; backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md\nReason: TASK-16230 and TASK-16309 already define the planner, executor, approval, and private receipt boundaries; this task exposes them through production UI and adds a read-only pre-approval lookup.\n\nPlan: Docs/superpowers/plans/2026-08-20-notes-import-once-ui.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the reviewed local Database Notes import-once workflow with mutation-free file/folder selection and planning, immutable approval, bounded paged review with collision/uncertain-match/update diff and membership controls, cooperative progress/cancel/retry, retained same-session progress and receipts, and post-settlement Notes/tree refresh. Added the SQLite-enforced read-only receipt lookup under the existing notes.sync_state private owner; no new storage owner or ADR. Hardened compact 60-column hierarchy, input/focus retention, truthful receipt/error copy, duplicate execution admission, off-thread cancellation settlement, privacy-safe projections, and a mutation fence that keeps legacy Sync visible but blocks concurrent Notes writers. Updated the Notes guide and testing lessons. Verification: backend/private SQLite gate 1391 passed and 4 skipped; final focused UI gate 271 passed with one known baseline modal-inventory node deselected; final Library shell 585 passed; CSS/static/diff gates green. Independent spec and UX reviews report Ready with no remaining findings. Commits: 956ae28a4, 30a44518f, 55191287d, 33897b00c, 68bad8fb0, 9f7bb99db, 3fc909ebb, e377e1361, 34be0a953.
<!-- SECTION:NOTES:END -->
