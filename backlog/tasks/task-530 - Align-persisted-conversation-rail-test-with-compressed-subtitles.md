---
id: TASK-530
title: Align persisted conversation rail test with compressed subtitles
status: Done
assignee: []
created_date: '2026-07-24 19:35'
updated_date: '2026-07-24 19:47'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the Console persistence-refresh regression test aligned with the compact conversation-row subtitle contract adopted by TASK-374, without restoring redundant workspace labels.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The persistence-refresh test no longer requires the removed workspace label inside grouped rows
- [x] #2 The test still verifies auto-title selection metadata and persistence refresh behavior
- [x] #3 The focused native Console flow test passes
- [x] #4 The TASK-374 source and no-ADR decision are documented
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the failure in isolation and compare the row rendering contract with TASK-374 commit f5411ab5a.
2. Replace only the stale grouped-row workspace-label assertion while retaining persistence, auto-title, selection, metadata, and identifier checks.
3. Run the focused test, the full native Console flow module, Ruff, format, and diff checks.
4. Independently review the alignment and document verification.

ADR required: no
ADR path: N/A
Reason: This updates one stale test to an existing UI copy decision and changes no production behavior or interface.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced two stale grouped-row assertions for the redundant `Chats` workspace/group label with the retained `active session` state differentiator. The tests continue to verify persistence refresh, automatic title replacement, selected-row state, normalized age metadata, persisted conversation resume, and inspector details.

Commit `f5411ab5a` (TASK-374) intentionally compressed grouped conversation subtitles by removing redundant workspace/default-state text while preserving non-default differentiators. The focused regression cases pass 4/4 and the complete native Console flow module passes 199/199.

ADR required: no. This aligns tests with an existing presentation decision and changes no production behavior or interface.
<!-- SECTION:NOTES:END -->
