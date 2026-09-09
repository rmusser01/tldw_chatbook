---
id: TASK-32181
title: >-
  Library Notes: adaptive reader items_width mismatch for wide terminals now
  reachable
status: To Do
assignee: []
created_date: '2026-09-09 17:53'
labels:
  - library
  - notes
  - tests
  - follow-up
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Unblocking task-32175's row-0 wait exposed a downstream, unrelated pre-existing defect in test_library_shell.py::test_library_production_width_matrix_custom_preferences: for terminal widths 120/170/235 (all four saved_width values), the resolved neutral layout's items_width computes 64 instead of the expected 56 once a Database Notes row is opened and the adaptive reader layout is re-synced from the shell. The test previously died earlier on the flat #library-notes-row-0 wait, so this assertion was never reached before. No production code was touched by task-32175 (test-only); this is a pre-existing geometry bug in the adaptive reader layout sync path, independent of the flat-list vs folder-tree row composition.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The 12 currently-red parametrized cases (terminal_width in 120/170/235, all four saved_width values) of test_library_production_width_matrix_custom_preferences pass
- [ ] #2 Root cause of the items_width discrepancy (64 vs 56) in the wide-terminal neutral layout is identified and documented
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Discovered while verifying task-32175 (Library Notes: retire or repair flat-row tests). Reproduction: run test_library_shell.py::test_library_production_width_matrix_custom_preferences after task-32175's fix lands -- 12 of 24 parametrized cases fail with AssertionError: assert neutral.items_width == neutral_items_width (64 == 56) at the assert following screen._sync_library_notes_reader_layout_from_shell(). Narrower widths (60/80/100) pass. Not fixed here -- task-32175 is test-only.
<!-- SECTION:NOTES:END -->
