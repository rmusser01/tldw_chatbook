---
id: TASK-32133
title: >-
  Library Notes editor: Escape is refused silently when a save is blocked, and a blank note reads Saved before a character is typed
status: To Do
assignee: []
created_date: '2026-09-08 21:39'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - copy
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Evidence assessor: with a title failing validation ('Title begins or ends with whitespace…'), Escape does nothing and says nothing; 'Discard new note' had already disappeared so the note had no exit except fixing the title. Design assessor: a new Blank note shows status 'Saved' and a list row 'Untitled' before anything is typed, and is then discarded if abandoned. Both halves are dishonest in opposite directions. Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A refused Escape states why and what to do ('Fix the title to leave, or Discard')
- [ ] #2 A fresh blank note shows a draft state until its first save actually lands
- [ ] #3 Covered by tests
<!-- AC:END -->
