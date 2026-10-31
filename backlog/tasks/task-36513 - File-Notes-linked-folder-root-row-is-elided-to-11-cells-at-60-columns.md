---
id: TASK-36513
title: File Notes linked-folder root row is elided to 11 cells at 60 columns
status: To Do
assignee: []
created_date: 2026-10-23 23:37
labels:
- follow-up
- library
- ux
priority: low
references:
- Tests/UI/test_library_file_notes_workspace.py
dependencies:
- TASK-36383
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-36383's review (2026-10-23). At 60x20, the Library File Notes root row gives the linked folder's status only 11 cells, so the user reads "Chec...otes" or "Link...otes" and cannot tell the state. The always-displayed "Review recovered pairing…" button (b5251e9a6e) takes 27 of the row's cells. With that button hidden, the status fits ("Linked · Local folder: notes", 38 cells).

QA logged the button as N-37/S-20. TASK-34000.30 AC#10 sets the button's visibility rule only for the no-folder state, so the linked row stays squeezed.

Because of this defect:
- `test_file_notes_linked_root_leads_with_friendly_folder_identity[size2]` (the 60x20 case) is red on dev.
- `test_initial_root_scan_projects_checking_authority_while_actions_are_gated` checks the summary and the tooltip instead of the painted row.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At 60x20 with a folder linked, the root row paints the state word (Checking, Linked, Offline or Warning) in full, and the folder name in full or middle-elided
- [ ] #2 The recovered-pairing review stays reachable whenever there is a recovered pairing to review
- [ ] #3 At 60x20, the row's Details and Change… controls stay on the row
- [ ] #4 test_file_notes_linked_root_leads_with_friendly_folder_identity passes at every size it is parameterized over
- [ ] #5 test_initial_root_scan_projects_checking_authority_while_actions_are_gated checks the painted root row again
<!-- AC:END -->
