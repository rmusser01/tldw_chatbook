---
id: TASK-32052
title: >-
  Library New-note canvas: nothing focused on entry; Enter/Down no-op; Tab
  leaves the screen
status: Done
assignee: []
created_date: '2026-09-08 18:22'
updated_date: '2026-09-08 19:59'
labels:
  - library
  - notes
  - ux
  - keyboard
  - critique-8
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pressing `n` opens the New-note canvas with the footer saying 'enter create note', but nothing has focus: Enter and Down do nothing, the first Tab walks into the top nav bar where Enter switches to Home, and 22 Tabs later focus is in the rail search box, never on Blank note. Creating the first note is mouse-only. Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 3.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Entering the New-note canvas focuses the Blank note control so Enter creates a note immediately
- [x] #2 Up/Down move between Blank note and the templates with a visible cursor
- [x] #3 Tab from any Library canvas stays inside the Library screen; the nav bar is reached only via its documented keys
- [x] #4 The footer hint on the New-note canvas is truthful for the focused control
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce live (n -> New note canvas focus, Tab escape to nav)\n2. Failing test\n3. Focus Blank note on entry; up/down between create rows; keep Tab inside Library\n4. Green, live-verify, docs
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The New-note canvas is keyboard-complete.

Entry focus: the rail-row switch now parks focus on #library-notes-create-blank, the branch the sibling CREATE_PROMPT/CREATE_SKILL entries already had. That covered the landing's `n`; Ctrl+N and the rail row pressed FROM the notes list take the RETAINED shell route, where the correct restore was immediately overwritten -- `_restore_library_notes_after_targeted_sync` replayed the pre-switch NAVIGATOR identity over the create canvas and focus landed on a notes-tree row (traced live and headlessly). Fixed at the root: that restore now refuses an identity whose region is no longer the live region, which is exactly the surface transition its own docstring says the capture does not serve.

Up/Down: `library-notes-create-row` joins `_LIBRARY_LIST_ROW_CLASSES` (that tuple's only consumer is `_move_library_list_row_focus`), and the rows gain the house focus-only left-edge bar.

Tab: LibraryScreen re-declares tab/shift+tab un-namespaced (a subclass's BINDINGS entry for a key REPLACES the inherited one) and scopes focus movement to `#screen-content`, so Tab no longer walks into the nav bar; focus already in app chrome keeps the app-wide chain. The two actions are excluded from F1 by name -- app-wide chrome is not a Library shortcut -- and added to the screen-navigation audit's universal allowlist.

Footer: 'enter create note' appears only while a create row has focus.

Files: tldw_chatbook/UI/Screens/library_screen.py, tldw_chatbook/UI/Library_Modules/screen_constants.py, tldw_chatbook/css/components/_agentic_terminal.tcss (+ regenerated bundle), Tests/UI/test_library_crit8_keyboard.py, Tests/UI/test_screen_navigation.py, Docs/User_Guide/library/notes.md.
<!-- SECTION:NOTES:END -->
