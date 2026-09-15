---
id: TASK-32616
title: >-
  Library Notes: the open note's list row keeps the old title at the moment a
  first-timer checks the save
status: To Do
assignee: []
created_date: '2026-09-15 06:41'
labels:
  - library
  - notes
  - critique-4
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 (dev 77eb2601a6), assessor A P2 and assessor B D20 (PARTIAL), persona Jordan, create workflow. Both assessors hit it independently; A names it the emotional valley of the whole first journey.

What happened. Jordan types a title, the editor heading reads 'My first note' and the status reads 'Saved 22:50' -- and the only durable artefact on screen, the row in the list beside them, still reads 'Untitled · now' (A caps 06, 07). It corrects the moment Escape leaves the editor (A cap 08). B saw the same shape later in the journey: after changing a title to a new value the list row still showed the old one while Info was open (B cap 58). For a first-timer the question is never 'did the widget update', it is 'is my writing safe', and the screen answers it two ways at once.

Cause PROVEN and pinned: notes.md documents the refresh as deliberately skipped while the title field holds focus. The reasoning is sound for a rename; applying it to a note whose title has never been anything, and to a row that is the user's only save receipt, is the part both assessors dispute. This is a design decision to revisit, not a bug to fix blind.

Compounding, same pane, same moment: two 'Next:' instructions disagree with each other -- the list's 'Library notes · Ready · Next: Create a note or add from files.' beside the editor's 'Empty note — type to keep it · Next: Start typing.' (A cap 04) -- and after typing the editor's still reads 'Next: Start typing.' (A cap 05). Open riders 32513/32514 hold the twice-painted save state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The currently open note's own row reflects its committed title, or is marked as the open note so the stale label reads as a pointer rather than a contradiction
- [ ] #2 The focus guard that skips the refresh still holds for every other row
- [ ] #3 One pane never shows two Next instructions that disagree
- [ ] #4 The decision is recorded either way, with the reasoning, so critique 5 does not re-litigate it
<!-- AC:END -->
