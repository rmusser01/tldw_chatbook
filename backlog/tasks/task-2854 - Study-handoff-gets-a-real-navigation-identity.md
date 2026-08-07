---
id: TASK-2854
title: Study handoff gets a real navigation identity
status: To Do
assignee: []
created_date: '2026-08-07 01:10'
labels:
  - library
  - study
  - navigation
  - ux
  - uat-2026-08-06
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Library UAT 2026-08-06 (LIB-06, observed at dev `6ffa56516`).

Rail Study ▸ "Study decks (0) / opens Study" first opens a Library-local staging canvas (so the
gloss is false for the first click), whose "Continue in Study" lands a full Study screen
(Dashboard/Paths/Flashcards/…) that has NO tab in the tab bar — the bar still highlights
"⌃3 Library" — Escape is dead, and the footer offers no way back. Keyboard users are stranded on
a screen the navigation model claims doesn't exist. This directly violates the "no hidden mystery
navigation" principle.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Study screen reached from Library has a truthful navigation identity: either its own tab highlight or a visible breadcrumb naming where you are and how to get back
- [ ] #2 The tab bar never highlights Library while a non-Library screen is displayed
- [ ] #3 Escape (or an advertised key) returns from the Study screen to the Library staging canvas
- [ ] #4 The rail gloss no longer promises "opens Study" for a click that opens a Library staging canvas
- [ ] #5 Live TUI verification of the full round trip Library → staging → Study → back
<!-- AC:END -->
