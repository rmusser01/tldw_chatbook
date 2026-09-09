---
id: TASK-32126
title: >-
  Library Notes empty state never renders: the tree projection short-circuits it once Agent_Lessons exists, and the guide and code disagree on the copy
status: To Do
assignee: []
created_date: '2026-09-08 21:39'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - onboarding
  - docs
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PROVEN: the canvas yields the empty Static only when `tree_projection is None`, and a brand-new profile always has a tree because the `Agent_Lessons` folder is seeded (documented in notes.md). Result: Notes (0) shows one row, '▸ Agent_Lessons', and nothing else. The guide promises 'No notes yet. Create one to see it here.'; `_EMPTY_NOTES_COPY` in library_notes_state.py says 'No notes yet. Create your first note.'; neither renders. A first-time user's first question is what Agent_Lessons is and whether they made it. Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A profile with zero notes shows an empty-state line with a call to action even when seeded folders exist
- [ ] #2 Agent_Lessons carries a one-line gloss in place, or is hidden until it contains a note
- [ ] #3 The guide and the code use the same empty-state copy
- [ ] #4 Covered by a test on a zero-note projection that contains the seeded folder
<!-- AC:END -->
