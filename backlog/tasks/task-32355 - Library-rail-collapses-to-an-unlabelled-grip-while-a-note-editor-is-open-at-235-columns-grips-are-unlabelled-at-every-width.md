---
id: TASK-32355
title: >-
  Library rail: collapses to an unlabelled '--->' grip while a note editor is
  open at 235 columns; grips are unlabelled at every width
status: Done
assignee: []
created_date: '2026-09-11 06:17'
updated_date: '2026-09-11 07:44'
labels:
  - library
  - layout
  - critique-10
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
With a note open at 235 columns the rail is replaced by '--->' with no label (A cap 11; the rail returns on Escape, cap 13); three literal '<---'/'--->' runs sit in the gutters at 235x52 (B D10, cap 17). PROVEN source: library_adaptive_reader_shell.py:151-156. library.md promises a slim 'Nav' handle and the rail beside Notes at 120 columns and wider. Evidence: critique #10 snapshot (.impeccable/critique/2026-09-11T06-13-11Z__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 1f3184655b, captures under the session scratchpad crit10/A and crit10/B).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 At 120 columns and wider the rail stays beside the note editor, or the collapse is documented as intended and the handle is labelled 'Nav'
- [x] #2 Every pane grip carries a label or a footer hint
- [x] #3 Pinned
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Decide rule-vs-preference for the rail beside a note editor from the config evidence.\n2. Label the grips Nav/Items above 5 cells, guillemet below.\n3. Docs + pinned painted assertions at 235/100/60.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The rail-beside-editor branch is settled by evidence, not by preference: the collapse is the Notes WORK-FIRST session (`library_notes_controller.py:2379-2386`, `NotesWorkSessionPhase.ACTIVE -> library_open=False`), a deliberate rule that activates ONLY at 120+ columns of editor width (`Tests/UI/test_library_notes_work_session.py` pins 119 -> INACTIVE, 120/121 -> ACTIVE) and is cancelled for the rest of the visit by one manual expand (MANUAL_LIBRARY_EXPAND -> MANUALLY_CANCELLED). The profile's persisted `[library.reader] library_open = true` says the user's preference is OPEN, so the collapse is the rule, not stored state. `library.md:273`'s '120 columns and wider the rail stays beside it' is about the FILE NOTES workspace, not the note editor; the page now states the editor rule where the rail is described. AC#1 is therefore taken via its second branch, the labelled handle.

AC#2: the handle carried its name only in `_name`/`tooltip`, neither of which a terminal paints. The column is five cells wide and 20-45 rows tall, so `LibraryAdaptiveReaderPaneGrip.render` now spells the name DOWNWARDS above the arrows -- 'Nav' for the Library pane (the guide's own name for this handle; 'Library' would not fit), the pane's own label for the items pane ('Items', 'Prompts', 'Skills', 'Folderfiles'). A horizontal label cannot hold any of those at five cells, which is why the plan's 'Nav >' / 'Items >' shape was not used. The arrow rows are untouched, so the 35/65-per-cent pins still hold.

Live at 235x52, 100x30 and 60x24 on the seeded profile: both handles paint their name, including on the rail a note editor collapsed.

Files: tldw_chatbook/Widgets/Library/library_adaptive_reader_shell.py, Tests/UI/test_library_crit10_layout.py, Docs/User_Guide/library.md.
<!-- SECTION:NOTES:END -->
