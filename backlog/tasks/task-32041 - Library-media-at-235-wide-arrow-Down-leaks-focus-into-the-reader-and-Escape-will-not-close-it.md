---
id: TASK-32041
title: >-
  Library media: at 235 wide, arrow-Down leaks focus into the reader and Escape
  will not close it
status: To Do
assignee: []
created_date: '2026-09-08 14:36'
labels:
  - library
  - media
  - ux
  - accessibility
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #7 P1 (lead). At 235x52, the first arrow-Down off a focused Media list row defocuses the list into the reader pane instead of advancing the list cursor; four Escapes then neither close the reader nor move a cursor; a Console round-trip strands Library on a provider 'Get started' card that needs Home->Library to recover. The same gestures work correctly at 100x30, so this is a width-dependent focus-routing bug. Keyboard navigation of the list is the core interaction.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At 235x52, arrow Down/Up move the Media list cursor (visible focus cue) while the list owns focus, rather than leaking focus to the reader
- [ ] #2 The Escape ladder returns focus predictably from the reader back to the list at every supported width
- [ ] #3 A Console -> Library round-trip lands on the Media list, not a Get-started card
- [ ] #4 Painted pins assert cursor movement at 235x52 (parity with the 100x30 behaviour that already works)
<!-- AC:END -->
