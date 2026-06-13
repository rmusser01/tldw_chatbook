---
id: TASK-60.4.2
title: Post-release write sync promotion tranche
status: Done
labels:
- post-release
- sync
- safety
- ux
priority: medium
parent_task_id: TASK-60.4
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Promote write sync only after the dry-run and audit evidence prove the user can understand authority, conflicts, recovery, and rollback before mutations are enabled.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Write sync scope references TASK-60.3 actual-use audit evidence and existing dry-run sync foundations.
- [x] #2 Mutation replay remains gated behind explicit user review, rollback, and conflict visibility.
- [x] #3 Library, Collections, Workspaces, and Settings expose consistent sync authority labels before writes are available.
- [x] #4 QA validates write-sync promotion with actual app use and non-destructive safety fixtures before completion.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add pure write-sync promotion display-state tests and module.
2. Add a read-only SyncScopeService promotion-state helper.
3. Wire shared promotion labels into Library Collections, Console workspace rail, and Settings without enabling mutation replay.
4. Run focused pytest and diff hygiene.
5. Capture actual rendered screenshots for changed visible surfaces before PR creation.
6. Update QA evidence and Backlog notes after approval.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented a read-only write-sync promotion display contract for Library Collections, Workspaces, and Settings. The shared builder turns readiness, mirror reports, conflict reports, and profile state into consistent authority, dry-run, review, conflict, rollback, and recovery labels while clamping mutation access to blocked for this tranche. Library Collections now renders selected Collection sync safety in both detail and inspector panes, Console workspace context uses the shared sync label, and Settings has a visible Sync Safety section instead of an empty-looking surface.

Actual textual-web/CDP screenshots were captured and user-approved for Settings Sync Safety, Library Collections Sync Safety, and Console Workspace Sync Safety. Focused verification passed with 34 tests and `git diff --check`; no write replay, approve-sync button, outbox drain, server mutation, or sync enablement path was added.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
`TASK-60.4.2` closes the display-only promotion tranche. Users can now see what would be required before write sync can be enabled, but all mutation paths remain intentionally unavailable.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
