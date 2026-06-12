---
id: TASK-103
title: Define a shell-wide pane-focus keyboard convention for workbench screens
status: To Do
assignee: []
created_date: '2026-06-11 20:43'
labels:
  - ux
  - keyboard
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Personas workbench (UX-E2) wanted pane-jump keys to cycle focus between the library list, work area, and inspector, but no house pattern exists: chat_screen.py and notes_screen.py define no pane-focus bindings at all, and ctrl+left/ctrl+right collide with Input/TextArea word-navigation (the keys are consumed inside text fields, making screen-level cycling inconsistent). Rather than invent a one-off convention on one screen, define a single shell-wide convention (key choice, wrap order, focus-steal rules) and apply it to Console, Notes, and Personas together.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A documented pane-focus key convention exists for workbench screens (keys, cycle order, text-input interaction),Personas/Notes/Console workbench screens implement the convention consistently,The chosen keys do not conflict with Input/TextArea default bindings or app-level bindings
<!-- AC:END -->
