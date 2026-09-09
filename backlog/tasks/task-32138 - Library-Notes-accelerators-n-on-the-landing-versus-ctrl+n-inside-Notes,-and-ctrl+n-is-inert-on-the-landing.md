---
id: TASK-32138
title: >-
  Library Notes accelerators: n on the landing versus ctrl+n inside Notes, and ctrl+n is inert on the landing
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
Parent verified live: the Library landing footer advertises 'n new note' and Ctrl+N does nothing there; the Notes canvas footer advertises 'ctrl+n new note'. `test_library_notes_bindings_are_inactive_outside_notes_workflow` pins the inactivity, so this is a pinned decision the critique disagrees with: the same action has two keys depending on the canvas. Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One accelerator for New note is advertised and works on the landing and inside Notes, or both keys work in both places
- [ ] #2 The pinning test records the decision
<!-- AC:END -->
