---
id: TASK-15508
title: Add persistent File Notes field labels
status: Done
assignee: []
created_date: '2026-08-11 23:05'
updated_date: '2026-08-11 23:13'
labels: []
dependencies: []
documentation:
  - backlog/decisions/011-chatbook-workbench-ui-system.md
modified_files:
  - tldw_chatbook/Widgets/Library/library_file_notes_workspace.py
  - Tests/UI/test_library_file_notes_workspace.py
  - backlog/tasks/task-15508 - Add-persistent-File-Notes-field-labels.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure the File Notes search and path controls retain their meaning after text entry, including the path field's current New, Move, or Restore context, without reducing compact-layout usability.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The search control keeps a persistent visible label after a query is entered.
- [x] #2 The path control keeps a persistent visible label whose copy reflects whether the field is for a new note, a new or moved note, or a deleted note restore.
- [x] #3 The labels remain visible and their neighboring inputs remain usable and keyboard reachable at 40x20 and standard desktop sizes.
- [x] #4 Focused tests cover label state changes and compact geometry, and focused static checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Characterize the current search and multi-purpose path states in the retained File Notes workspace.

2. Add focused tests for persistent copy, state transitions, keyboard reachability, and 40x20 geometry.

3. Implement compact inline labels and synchronize the path label with New, Move, and Restore context without recomposition.

4. Run focused tests, Ruff, and a self-review of the scoped diff.

5. Complete acceptance criteria and record implementation notes and verification evidence.

ADR required: no

ADR path: N/A

Reason: This is bounded UI clarity polish within the retained composition and state presentation governed by ADR-011; it does not change storage, sync policy, ownership, or a long-lived interface boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added persistent inline Search and state-aware path labels within the existing three-row control footprint. The path label now changes in place across New path, New / move path, and Restore path states; the labeled search row also hides as one unit when the Git navigator is active, and the labeled path row yields to destructive reload confirmation. Added mounted tests for entered-value persistence, state transitions, Git-mode visibility, tab reachability, and neighbor geometry at 120x40 and 40x20. Verification: 5 focused mounted tests passed, including both reload-confirmation sizes; Ruff check passed; diff check passed. The new state-sync assertion was mutation-verified by removing the sync and observing the expected failure. Ruff format check remains red on the unchanged HEAD versions of both touched files because of existing whole-file formatting drift, so no unrelated bulk reformat was included. ADR check: no new ADR; ADR-011 governs the retained workbench composition and state presentation. No reusable lesson was added because the work did not surface a new repository-wide trap.
<!-- SECTION:NOTES:END -->
