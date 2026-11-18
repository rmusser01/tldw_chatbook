---
id: TASK-41916
title: "Stale Library focus comments and a hidden-row focus during Add from files"
status: To Do
assignee: []
created_date: 2026-11-18 20:31
updated_date: 2026-11-18 20:31
labels:
- follow-up
- ui
- library
priority: low
references:
- tldw_chatbook/UI/Screens/library_screen.py
- Tests/UI/test_library_shell.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41504 (round 16): the `restore_focus_after_recompose` docstring (library_screen.py ~8082-8084) and `Tests/UI/test_library_shell.py` ~37061 still say Conversations has no row class; and on dev the armed Notes entry briefly focused a hidden `library-notes-tree-note-1` while the user was in Add from files (it healed afterwards).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The two comments match the code
- [ ] #2 The Add-from-files focus never lands on a hidden row, pinned by a test, or the notes show it cannot reach the user
<!-- AC:END -->
