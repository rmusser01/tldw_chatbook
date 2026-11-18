---
id: TASK-41601
title: "Folder files mode leaves focus on the hidden Database Notes filter, and a Notes rail entry aims at the hidden Database rows"
status: To Do
assignee: []
created_date: 2026-11-17 19:40
labels:
- follow-up
- library
- focus
- ui
priority: medium
references:
- tldw_chatbook/UI/Library_Modules/library_notes_controller.py
- tldw_chatbook/UI/Screens/library_screen.py
- Tests/UI/test_library_shell.py
dependencies:
- TASK-41129
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the libfocus stream (TASK-41129 fix round 4, 2026-11-17). It happens on base cc6006fcf5 and on head.

A probe of Notes in Folder files mode found two things:
- the Database filter and rows stay mounted but hidden, out of `screen.focus_chain`;
- after the switch to Folder files, focus is left on that hidden filter, and a Notes rail entry aims at the hidden Database rows.

**Code** (read at 212f015a91):
- **The switch.** `_show_library_file_notes` (`library_notes_controller.py:5523`) clears focus through `_evacuate_library_notes_authority_focus("database")` (`:1824`). It then hides the Database shell and calls `_restore_library_notes_authority_focus("files")` (`:5598`). That call focuses nothing when `#file-notes-search` is not yet in the focus chain (`:1836-1856`).
- **No reachability check on the Notes restore.** Neither `_library_notes_fallback_focus_target` (`:2093-2094`, the filter) nor `_restore_library_notes_focus_identity` (`:2202`) checks the focus chain. A Notes restore can therefore land on the hidden filter. The probe did not name which call landed it there.
- **The rail entry.** `_focus_library_list_entry` (`library_screen.py:11503`) maps the Notes rail row to the `library-notes-row` class (`:11513`), and the query finds the hidden Database rows (`:11531`). It then calls `set_focus(rows[0])` (`:11627`, `:11643`) with no focus-chain check. Only the empty-list fallback checks (`:11580-11585`, added by TASK-41129). TASK-41129's fix round 5 removed a Database-only gate on these rules, because it changed nothing in the cases then pinned.

**Related.** TASK-41504 covers code-driven and rail entries that land on a grip or the rail row in Database mode.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After Notes switches to Folder files, focus is on a visible Folder files control or on nothing, never on a hidden Database Notes control
- [ ] #2 A Notes rail entry in Folder files mode never focuses a hidden Database row or filter; it lands on the Folder files entry target or a documented empty-state control
- [ ] #3 Pilot pins drive both journeys and assert the focused widget id and that it is in `screen.focus_chain`; red on the code before the fix
<!-- AC:END -->
