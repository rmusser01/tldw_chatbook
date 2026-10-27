---
id: TASK-37020
title: >-
  Console: the avatar painter re-mounts the portrait inside a collapsed
  Character section on every expression change
status: To Do
assignee: []
created_date: '2026-10-26 13:55'
labels:
  - console
  - performance
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the TASK-35970 review (2026-10-26). TASK-35970 stopped the collapsed Character browser from recomposing on every send, but in a character chat with a portrait the avatar painter still replaces the portrait widget inside the collapsed section body on every expression change. A streaming send with reactions on changes the expression twice (idle, speaking, idle), and each replacement removes and mounts the portrait, requesting layouts for a widget no one can see. Measured in a mounted probe (the real Console with a character card that has idle and speaking images, Character collapsed): two expression changes made 8 layout requests on the portrait holder (ClickableAvatarBox#console-character-avatar and its AutoImage child) inside the hidden body. The behaviour predates TASK-35970; dev 0d7e418db1 does the same. The geometry path already skips a collapsed body; the paint path does not.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A send in a character chat with a portrait and reactions on, with the Character section collapsed, causes no layout request inside the Character section
- [ ] #2 Opening the section after such a send shows the current portrait and reaction caption from its first painted frame
- [ ] #3 Verified live in a character chat with a portrait (real app, real model)
<!-- AC:END -->
