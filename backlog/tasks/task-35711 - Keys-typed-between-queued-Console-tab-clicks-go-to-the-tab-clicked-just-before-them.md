---
id: TASK-35711
title: >-
  Keys typed between queued Console tab clicks go to the tab clicked just before
  them
status: To Do
assignee: []
created_date: '2026-10-20 05:19'
labels:
  - console
  - tabs-sessions
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When two Console tab clicks queue behind a busy screen, the keys typed between and after them reach the composer in no fixed order relative to the presses, because keys and presses wait in the same screen queue. TASK-35640 pinned only what holds in every order: no key stays in the tab you started from, none is lost, and order is kept. Live, a key typed after clicking Chat 3 sometimes landed in Chat 2, so the user ends on Chat 3 with their text in Chat 2. Textual stamps every Key and Click event with its creation time, which could let each key be assigned to the click it followed rather than to whichever press is handled first. Worth doing if that outcome proves confusing in use.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With two tab clicks queued behind a busy Console, each key goes to the draft of the tab clicked most recently before that key was typed, whatever order keys and presses are handled in
- [ ] #2 A mounted test drives the keys-first, presses-first and interleaved orders and passes under the plain pytest command
- [ ] #3 The Console User Guide's tab-switch typing paragraph states the stronger guarantee
<!-- AC:END -->
