---
id: TASK-32218
title: >-
  Library Notes vocabulary: one journey passes 'Library notes', 'Library
  database', 'Folder files', 'Folder Files' and 'esc back to Database'
status: Done
assignee: []
created_date: '2026-09-10 14:54'
updated_date: '2026-09-11 02:00'
labels:
  - library
  - notes
  - copy
  - critique-9
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Five names for the two Notes sources across the rail, canvas titles, the strip and the footer (the Collections/Quick Capture/captures drift is task-32057's decision). Jordan cannot connect the word clicked to the word landed on. Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 15.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One noun per Notes source used in the rail row, canvas title, strip, empty state and footer
- [x] #2 The guide uses the same nouns
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inventory every user-visible Notes-source noun
2. Apply 'Library notes' / 'Folder files' everywhere
3. Pin with a painted-text walk in both modes
4. Update the two guide pages
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Inventory first (grep over tldw_chatbook/ and Docs/User_Guide/library/), then one noun per source everywhere a reader can see it.

Six production sites carried the drift: library_notes_canvas.py's NOTES_AUTHORITY_PREFIX ("Library notes · Library database") and three sibling literals ("Database Notes · Library database"), library_file_notes_workspace.py's authority line ("Folder Files · …"), and library_screen.py's two Escape labels (the F1 row "Back to Database notes" and the footer chip "back to Database"). The source strip, the Add-from-files canvas and the sync-roots canvas were already correct and were not touched.

Now: **Library notes** and **Folder files**, matching the strip labels the reader actually clicks. The three canvas literals were replaced by the one NOTES_AUTHORITY_PREFIX constant rather than three new copies. The empty state keeps its lowercase prose ("These notes live in the Library's own database") -- that describes where notes live and is not a name.

Docs: both guide pages rewritten to the same two nouns (Database Notes/Database notes -> Library notes, Folder Files -> Folder files, "the Library database" -> "the Library's own database", the "**Database** button" -> "**Library notes** button"), plus a stamp on each. Historical Verified-against stamps keep their original wording.

Five existing pins asserted the retired copy and were updated in place (test_library_crit8_polish_shell, test_library_notes_files_sync_journey, test_library_shell, test_library_file_notes_workspace, test_library_notes_canvas x4); none of them asserted a design decision, only the old string. New walk: Tests/UI/test_library_crit9_notes.py asserts the painted frame in both modes carries the mode's noun and none of "Library database", "Database Notes", "Folder Files", "back to Database" or a bare word "Database", and that the Folder files footer chip reads `esc back to Library notes`.

KNOWN GAP: Settings still labels the pane "Folder Files tree" (settings_appearance_defaults.py, settings_search_index.py, settings_screen.py). Those files are outside this branch's ownership, so they are left for a follow-up rather than edited here.

Files: tldw_chatbook/Widgets/Library/library_notes_canvas.py, library_file_notes_workspace.py, tldw_chatbook/UI/Screens/library_screen.py (two one-line labels), Docs/User_Guide/library/notes.md, file-notes.md, Tests/UI/test_library_crit9_notes.py + 5 pin files.
<!-- SECTION:NOTES:END -->
