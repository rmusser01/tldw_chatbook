---
id: TASK-32131
title: >-
  Library Notes slash accelerator inserts itself into the filter it focuses
status: To Do
assignee: []
created_date: '2026-09-08 21:39'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - keyboard
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PROVEN live (parent + evidence assessor): pressing `/` in the Notes list focuses the filter and types '/', so 'Reading' becomes '/Reading' and the query is wrong. The footer advertises '/ find note' and the guide lists it as 'Focus the note filter'. Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Pressing / focuses the filter and leaves its content unchanged
- [ ] #2 Covered by a test
<!-- AC:END -->
