---
id: TASK-31554
title: Library note UI tests use obsolete positional row IDs
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 00:58'
updated_date: '2026-09-05 01:24'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the legacy note-row selector contract in the current Library folder-tree projection. The tree projection changed note row IDs to tree-position IDs, breaking broad current note workflows and making selectors depend on preceding folder/pager rows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Folder-tree note rows retain the established sequential library-notes-row IDs while keeping their tree-specific class and metadata.
- [x] #2 Tree-focused widget and adaptive-reader tests assert the compatible note-row identity.
- [x] #3 Affected Library note UI clusters pass without increasing wait budgets.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm flat-list and folder-tree row identity behavior and affected test contracts.
2. Preserve the established sequential note-row IDs inside the folder-tree projection, independent of folder and pager positions.
3. Update the two tree-specific selector expectations and run focused widget, adaptive-reader, shell, and notes-reader tests.

ADR required: no
ADR path: N/A
Reason: This is a compatibility bug fix within the existing Library adaptive reader and folder-tree architecture defined by ADR-086; no boundary or ownership decision changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Assigned folder-tree note buttons sequential `library-notes-row-N` IDs using a note-only counter, independent of folders and pager rows, while retaining tree classes and placement metadata.
- Updated the two tree-specific expectations to the compatible selector contract.
- Evidence: five focused tree/shell/reader workflows pass; the full Notes reader diagnostic reaches 32 passes with two unrelated residual failures and no wait-budget increase.
- ADR required: no; this is a compatibility repair under ADR-086.
<!-- SECTION:NOTES:END -->
