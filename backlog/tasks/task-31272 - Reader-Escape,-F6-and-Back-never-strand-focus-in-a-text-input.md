---
id: TASK-31272
title: Reader Escape, F6 and Back never strand focus in a text input
status: To Do
assignee: []
created_date: '2026-09-04 13:54'
labels:
  - library
  - media-ux
  - a11y
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 P1: leaving the Reader takes three Escapes and the second lands inside the rail's Search input (A cap 20); `s` typed at the F6 rail stop lands in `Search Library…` (B cap_79); F6's content stop is real (first candidate in `_MEDIA_WORKBENCH_FOCUS_TARGETS`) but invisible after the task-31221 outline suppression, so a keyboard user cannot tell it landed (B cap_57); Escape does not close the More menu though the label promises `close more` (B cap_106); `‹ Back` flips `_library_media_view` to list without changing the visible Reader, so ]/[ die on identical pixels (A cap 25/31/39). Eight distinct `esc …` labels were seen live.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Escape from the Reader focuses the loaded list row, never an Input
- [ ] #2 The F6 content stop has a visible focus treatment that does not paint over content (painted-text test proves content still paints; a focus test proves the treatment)
- [ ] #3 Escape closes the More menu
- [ ] #4 In the three-pane shell, Back either is removed or makes list mode visible (dimmed Reader or a parked label) so the key-map change is explained on screen
- [ ] #5 Distinct `esc …` labels reduced to four or fewer, table recorded in the notes
- [ ] #6 Live-verified
<!-- AC:END -->
