---
id: TASK-31981
title: 'Library: a blocked action explains itself only on mouse hover, not inline'
status: To Do
assignee: []
created_date: '2026-09-07 22:48'
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
Critique #6 P1, both assessors. Clicking a disabled `○ Generate` in the reader's Analysis pane produces a byte-identical screen; the reason (`No analysis provider is configured.`) appears only on mouse hover, unreachable by a keyboard-first user. The same silence covers `○ Analyze` in select mode. The Export gate already does this correctly two clicks away: `No destination chosen` is printed directly beneath the blocked `○ Export bundle (.zip)`. The product contract says explain why unavailable actions are unavailable, and the design doc says use recovery callouts instead of silent disabled controls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A blocked Generate/Analyze action shows its reason as an always-visible line adjacent to the control, not only in a hover tooltip
- [ ] #2 The reason names a next step where one exists (e.g. Set a provider in Settings)
- [ ] #3 The pattern matches the existing Export gate's inline-reason treatment
- [ ] #4 A painted pin asserts the reason text is present on the screen with no hover
<!-- AC:END -->

## Renumbering provenance

Filed as TASK-31977 during critique #6's fix wave; renumbered to TASK-31981 because a concurrent session landed its own TASK-31977 on dev first (2026-08-21 owner rule, TASK-19601: older arrival keeps the id). No other task references this one.
