---
id: TASK-32572
title: 'Console: Tab does not cycle the Get started card''s buttons'
status: To Do
assignee: []
created_date: '2026-09-14 22:44'
labels:
  - console
  - rider
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by group 4 while delivering task-32555, which made the Get started card's actions real bordered buttons with a visible focus cue. The cue only helps if focus can move: Tab does not cycle the card's buttons, so a keyboard-only user on a no-provider profile cannot reach 'Write a note in Library' or a detected-server action without the mouse. Pre-existing, outside 32555's ACs, and now more visible because the buttons look focusable. The card can carry three actions (Set up provider, Write a note in Library, and Use detected <provider> when a loopback server is found).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Tab and Shift+Tab cycle every action the card is currently showing, including the detected-server action when present
- [ ] #2 The focus cue task-32555 added marks the focused one at each stop
- [ ] #3 Driven live on a no-provider profile, at 235x52 and 100x30, with a capture per stop
<!-- AC:END -->
