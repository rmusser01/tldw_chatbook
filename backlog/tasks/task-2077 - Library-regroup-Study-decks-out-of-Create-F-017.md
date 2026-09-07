---
id: TASK-2077
title: 'Library: regroup Study decks out of Create (F-017)'
status: Done
assignee: []
created_date: '2026-08-03 17:24'
updated_date: '2026-08-04 05:22'
labels:
  - ux-review
  - library
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Study decks sits under Create but is a Study handoff ('Continue in Study'), mis-grouped by the codebase's own admission. Evidence: LIBRARY_STUDY_HANDOFF_MODES. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Study decks is grouped with handoff/open actions, not creation,Tests updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (rail grouping only; row ids, targets, and behavior unchanged). Steps: 1. RED tests: Tests/Library/test_library_rail_state.py defaults pin gains 'study' section + study_open default; Tests/Library/test_library_shell_state.py fixed-table test expects sections [browse, create, study, ingest] with Create reduced to the three creation verbs and a Study section holding the three handoff rows; Tests/UI/test_library_shell.py section-header selector list gains #library-rail-section-header-study. 2. library_shell_state.py: move create-study/create-flashcards/create-quizzes rows (row ids unchanged -- many tests press them) into a new 'study' section between Create and Import/Export. 3. library_rail_state.py: LIBRARY_RAIL_SECTION_IDS + study_open (default True) + coerce/serialize. 4. Run shell-state/rail-state/shell/study-context/knowledge-entry/contract-layout tests + ruff.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The three Study rows (Study decks/Flashcards/Quizzes) moved from the Create section to a new 'Study' rail section between Create and Import/Export -- they are handoffs ('Continue in Study' per LIBRARY_STUDY_HANDOFF_MODES), not creation verbs. Row ids deliberately unchanged ('create-study' etc. are long-published DOM ids pressed by tests and deep links); the section_id carries the regroup. library_rail_state.py: LIBRARY_RAIL_SECTION_IDS gains 'study', new study_open preference (default True) with coerce/serialize support. Files: library_shell_state.py, library_rail_state.py, Tests/Library/test_library_shell_state.py (fixed-table test + positional section lookups updated, _create_row -> _study_row), Tests/Library/test_library_rail_state.py (defaults pin), Tests/UI/test_library_shell.py (section-header selector list), Docs/User_Guide/library.md (section list + Study rows table). Verified: targeted 33 passed; study/knowledge-entry/core-loop suites 21 passed; destination/parity/contract 225 passed + 1 skip; full test_library_shell.py 313 passed + 1 flaky-timing pass-in-isolation (test_library_shell_ingest_canvas_live_updates_without_manual_recompose, same known-flaky class). Ruff clean on changed files (1 pre-existing F401 in test_library_shell.py untouched). ADR: not required (rail grouping; row ids/targets/behavior unchanged). Commit 9c695c284.
<!-- SECTION:NOTES:END -->
