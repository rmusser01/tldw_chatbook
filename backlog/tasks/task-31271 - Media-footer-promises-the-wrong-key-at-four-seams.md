---
id: TASK-31271
title: Media footer promises the wrong key at four seams
status: To Do
assignee: []
created_date: '2026-09-04 13:54'
labels:
  - library
  - media-ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 P1: (a) after Escape closes Find the footer still says `esc close find` because `_library_media_escape_label` reads the DOM before the recompose lands (A cap 08/23, B cap_21); (b) right after `s` the footer promises `space toggle selection` while Space is a no-op unless a row is focused and, with focus on the pane grip, collapses the Library pane (A cap 31→32, B cap_69) — `_toggle_library_media_select_mode` never moves focus to a row; (c) on the last item of a set the chip still reads `] next in set` although that ] is the completion gesture (B cap_50); (d) l/c/t are real Reader keys with show=False and `t` arms delete unadvertised (B cap_97). The footer is the surface's main trust instrument.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After Escape closes Find, the footer no longer shows `esc close find`
- [ ] #2 Pressing s focuses the first media row so the Space chip is true immediately; Space never falls through to the pane grip
- [ ] #3 On the last item of a review set the ] chip names the completion gesture (e.g. `] finish review`) instead of `next in set`
- [ ] #4 l, c and t are advertised in the Reader footer set (t at minimum, since it arms delete)
- [ ] #5 Each seam has a pinning test
<!-- AC:END -->
