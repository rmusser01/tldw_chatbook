---
id: TASK-32609
title: >-
  Library Notes: slash focuses the rail search outside the navigator while the
  Notes footer advertises it everywhere
status: In Progress
assignee: []
created_date: '2026-09-15 06:38'
updated_date: '2026-09-15 17:59'
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
- [x] #1 Pressing / anywhere the Notes footer advertises it focuses the notes filter and selects its existing text, and never reaches the rail search
- [x] #2 Where / genuinely does something else, the footer says so rather than advertising find note
- [x] #3 The key is consumed by the notes binding so it can never be inserted as a character by the same press that focuses the field
- [x] #4 A test covers / pressed from the rail, from a toolbar button and from the editor, not only from the navigator
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm the gate (check_action) and the footer tier disagree only while a text field holds focus.
2. Make the navigator tier drop the '/' chip in exactly the states where the key is a literal character.
3. Pin '/' from the rail search, from a toolbar button and from the editor.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The gate and the tier disagreed about ONE thing, and it was the tier. `check_action('library_notes_focus_filter')` requires the navigator region AND a focused widget that is not an Input/TextArea; the navigator footer tier is chosen by REGION alone, and the region is computed from the notes VIEW, not from where focus is -- so with the caret in the filter box, or (after Escape) in the rail's own 'Search Library…' box, the footer still read '/ find note' while '/' was landing as a literal character and the next keystrokes were going into the rail search. That is assessor A's walk exactly, and it is also why B, who pressed '/' from the navigator with a row focused, saw it work every time.

Fix: the navigator tier drops every single printable chip while a text field holds focus -- the same transform `_library_footer_shortcuts_for_current_state` already applies to the shared Library tiers, scoped to the keys this tier owns. Escape keeps its chip because `action_library_notes_escape`'s terminal branch really does move to the rail from a focused field. No behaviour change to the key itself: the accelerator and its literal-character branch are B's documented, verified contract and the guide describes both.

AC#3 was already true and is now pinned rather than assumed -- on_key consumes '/' with stop()+prevent_default() before focusing the filter, so the press that focuses it can never also insert it.

AC#4's three call sites: from the rail (the footer no longer advertises; '/' types there), from a notes toolbar Button ('/' focuses #library-notes-filter and neither input gains a character), and from the editor (the editor tier never carried a '/' chip and check_action refuses the action there).

Files: UI/Screens/library_screen.py (`_library_notes_footer_shortcuts`, navigator branch), Tests/UI/test_library_notes_w5_kbd_focus.py, Docs/User_Guide/library/notes.md.
<!-- SECTION:NOTES:END -->
