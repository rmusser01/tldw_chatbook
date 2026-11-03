---
id: TASK-38647
title: File Notes conflict-discard test asserts Cancel focus before the deferred focus lands
status: To Do
created_date: 2026-11-03 05:51
labels:
- follow-up
- tests
- library
priority: medium
references:
- Tests/UI/test_library_file_notes_workspace.py
- tldw_chatbook/Widgets/Library/library_file_notes_workspace.py
dependencies:
- TASK-38646
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviewing TASK-38646 (2026-11-03). `Tests/UI/test_library_file_notes_workspace.py::test_conflict_resolution_discard_keeps_cancel_first_confirmation` fails now and then at `assert cancel.has_focus` (:5964 on dev 18482c1cf7). It already failed on dev-identical code, through the scratch-profile wrapper:
- run on its own: 2 of 6;
- in a 3-test run with `-n 4`: 5 of 8;
- in the whole file with `-n 4`: 1 of 3.

**Cause.** The test waits only for `reload_confirmation_active`, then asserts focus at once. `_set_reload_confirmation` places that focus later, through `self.call_later(cancel.focus)` (library_file_notes_workspace.py:3627). This is the same family of race as TASK-38646.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test passes 10 of 10 runs on its own and 10 of 10 three-test runs with `-n 4`, each its own pytest process, because it waits for Cancel to take focus
- [ ] #2 It still fails if Cancel never takes focus after the confirmation opens (shown under a temporary source edit)
<!-- AC:END -->
