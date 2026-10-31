---
id: TASK-36528
title: File Notes 'Save failed' paints at 3.89:1 in textual-dark, and the Offline state elides to one character at 40 columns
status: To Do
created_date: 2026-10-24 00:26
labels:
- follow-up
- library
- notes
- accessibility
- tests
priority: medium
references:
- tldw_chatbook/Widgets/Library/library_file_notes_workspace.py
- Tests/UI/test_library_file_notes_workspace.py
dependencies:
- TASK-36383
- TASK-36513
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-36383 (PR #3232, 2026-10-23). `Tests/UI/test_library_file_notes_workspace.py` has 10 tests that fail on dev outside that task. The file runs on no PR lane. Re-run on dev 5b5920b8e2, all 10 failed. Six are covered elsewhere:
- the five `test_notes_authority_*` nodes and `test_wide_files_task_return_restores_database_browse_receipt` by TASK-32185 AC#4;
- `test_file_notes_linked_root_leads_with_friendly_folder_identity[60x20]` by TASK-36513.

This task is two of the other three; the third, the owned-replica failure, is filed on its own:
- **A product contrast defect.** `test_high_stakes_file_notes_states_are_legible_in_shipped_themes[(120, 40)]` fails with "textual-dark: 'Save failed' paints at 3.89:1, below 4.5:1". The error state is styled by `#file-notes-save-status.-error { color: $text; background: $error 14%; text-style: bold; }` (workspace.py:1106-1110). The test was already failing on 2026-09-11 (TASK-32185's notes).
- **The Offline state word at 40x20.** The `[(40, 20)]` case fails on `assert 'Offline' in '.'`: at 40 columns, the root status is elided to one character. TASK-36513 traced the same squeeze at 60x20 to the always-displayed 27-cell "Review recovered pairing…" button. Whether that button also explains 40x20 is not confirmed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every high-stakes File Notes save and root state paints at 4.5:1 or better in textual-dark, textual-light and high_contrast_yellow_black
- [ ] #2 Both cases of test_high_stakes_file_notes_states_are_legible_in_shipped_themes pass on dev
<!-- AC:END -->
