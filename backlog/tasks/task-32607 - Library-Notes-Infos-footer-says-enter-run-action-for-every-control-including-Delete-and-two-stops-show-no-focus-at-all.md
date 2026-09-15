---
id: TASK-32607
title: >-
  Library Notes: Info's footer says enter run action for every control including
  Delete, and two stops show no focus at all
status: To Do
assignee: []
created_date: '2026-09-15 06:38'
labels:
  - library
  - notes
  - critique-4
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 (dev 77eb2601a6), assessor A P1, personas Sam and Jordan, edit workflow.

What happened. Everywhere else in the editor the footer names the focused control ('enter save note', 'enter use in Console', 'enter undo delete'). Inside Info, eleven consecutive Tab presses produced the same generic chip for '‹ Notes', Keywords, Copy, Export Markdown, Export text and Delete (A cap 11 plus 11 Tab probes), and two of those stops showed no focus indicator anywhere on screen (A caps 12, 13). A keyboard user pressing Enter in Info is choosing blind between 'copy to clipboard' and 'delete this note'. The ANSI decode also shows Delete's label rendered grey #a5a5a5 against its siblings' white #e1e1e1, so the destructive control reads as the disabled one.

Cause, PROVEN. Of the four Notes footer tiers, navigator, preview and editor are wrapped in _with_library_notes_focus_chip (UI/Screens/library_screen.py:8224, :8236, :8256); the context tier -- Info -- is not (:8241-8245), so every Info stop renders the tier's literal ('enter', 'run action') from LIBRARY_NOTES_CONTEXT_SHORTCUTS (:1553-1556). Wave 4's task-32537 (#2683) added the chip to Preview and left this tier alone.

Docs contradicted: notes.md says 'The footer names whichever editor control has focus as an enter … chip, so focus is never unaccounted for.'
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every Info control names itself on the footer when focused, in the same grammar the editor tier uses
- [ ] #2 Every Info tab stop paints a visible focus indicator that survives a monochrome capture
- [ ] #3 Delete in Info is styled as destructive rather than dimmer than its neighbours
- [ ] #4 A test pins the focus chip for each Info control by name, so a new control cannot ship chip-less
<!-- AC:END -->
