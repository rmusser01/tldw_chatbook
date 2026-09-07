---
id: TASK-19013
title: Declare WorkspaceCreateModal in the Library modal inventory
status: Done
assignee:
  - '@codex'
created_date: '2026-08-21 01:23'
updated_date: '2026-08-22 17:32'
labels:
  - library
  - testing
  - modal
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the Library modal inventory gate so it resolves and declares the existing WorkspaceCreateModal launch instead of failing before bidirectional inventory assertions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Library modal inventory resolves `create_local_workspace` to `WorkspaceCreateModal`.
- [x] #2 The `WorkspaceCreateModal` contract and any transitive modal edge are declared or explicitly excluded with a recorded reason.
- [x] #3 The full Library modal-dismissal suite reaches and passes its bidirectional inventory assertions.
- [x] #4 No production modal behavior changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the full Library modal-inventory failure on current dev and confirm WorkspaceCreateModal is already declared.
2. Replace the single retired Notes sync modal edge with the exact current lasting-sync FileOpen edge; make no production changes.
3. Run the exact inventory node, full modal-dismissal suite, Ruff/format/diff checks, and verify the production tree is unchanged.
4. Record evidence, check all acceptance criteria, and mark the task Done.

ADR required: no
ADR path: N/A
Reason: this is a test-inventory correction for existing modal behavior and introduces no architecture or runtime contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified that current dev already declares `LibraryScreen.create_local_workspace -> WorkspaceCreateModal`, includes the exact WorkspaceCreateModal dismissal contract, and discovers no undeclared transitive edge. The full inventory gate exposed one adjacent stale declaration left by the lasting Notes cutover, so the test inventory now declares `handle_library_notes_lasting_folder_requested -> FileOpen` and removes the retired legacy Sync browse edge; production code is unchanged. Verification: the exact bidirectional inventory node passed; the full `Tests/UI/test_library_modal_dismissal.py` suite passed 170 tests with one pre-existing requests dependency warning; Ruff check, Ruff format check, and `git diff --check` passed. No new ADR or lesson was required because this only reconciles a test inventory with existing runtime behavior.
<!-- SECTION:NOTES:END -->
