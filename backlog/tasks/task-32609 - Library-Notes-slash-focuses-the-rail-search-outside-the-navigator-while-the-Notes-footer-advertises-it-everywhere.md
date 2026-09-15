---
id: TASK-32609
title: >-
  Library Notes: slash focuses the rail search outside the navigator while the
  Notes footer advertises it everywhere
status: To Do
assignee: []
created_date: '2026-09-15 06:38'
labels:
  - library
  - notes
  - critique-4
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 (dev 77eb2601a6), assessor A P1 (consolidated to medium -- B contradicts the general claim), persona Alex, create/edit workflow. Heuristic 7 setter (3 -> 2).

What happened. A: from the notes list, two '/' presses then typing 'abc' put abc in the rail's 'Search Library…' field while the footer read '/ find note' the whole time (A cap 44 preamble); on another attempt the '/' landed as a literal character, the field reading '/reading' and the status line 'filter: /reading · 2 results' (A caps 41, 42, 43); after Escape, '/' did nothing. B pressed '/' from the notes navigator and it worked every time -- '/' -> Very -> Enter applied the filter (B caps 06, 07, K5/K6), and B's docs check VERIFIED both '/ focuses the filter without typing a literal slash' and 'once the filter has focus / is an ordinary typeable character' (B K26). The two assessors are both right: the accelerator is state-dependent.

Cause, PROVEN. LibraryScreen.check_action gates library_notes_focus_filter on 'visible_notes and region == navigator and the focused widget is not an Input or TextArea' (UI/Screens/library_screen.py:24746-24751). Outside the navigator -- the rail after Escape, a toolbar button, the editor -- the binding is inactive and the key falls through to the screen's other '/' binding, 'focus search'. The literal-slash case is the documented Input branch of the same gate. The footer advertises '/ find note' regardless of region.

Not a wave-4 regression: the gate predates 5fd502dbac; wave 4's 32550 fixed the stale-text re-focus, a different branch. Intended behaviour is pinned by Tests/UI/test_library_notes_w4_editor.py::test_slash_on_a_filtered_list_selects_the_existing_text.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Pressing / anywhere the Notes footer advertises it focuses the notes filter and selects its existing text, and never reaches the rail search
- [ ] #2 Where / genuinely does something else, the footer says so rather than advertising find note
- [ ] #3 The key is consumed by the notes binding so it can never be inserted as a character by the same press that focuses the field
- [ ] #4 A test covers / pressed from the rail, from a toolbar button and from the editor, not only from the navigator
<!-- AC:END -->
