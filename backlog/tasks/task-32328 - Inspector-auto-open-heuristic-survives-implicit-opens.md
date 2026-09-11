---
id: TASK-32328
title: >-
  Inspector auto-open heuristic survives implicit opens
status: To Do
assignee:
  - '@zcode'
created_date: '2026-09-10 12:00'
labels:
  - console
  - ux
  - rail-ux-review
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX review B4. The 118-128 column Inspector auto-open band (_should_open_standard_width_inspector, chat_screen.py ~8298-8329) is permanently suppressed once ANY right_open preference is stored. Use the existing explicit-marker pattern (ADR-043 left_open_explicit) so only explicit Inspector toggles disable the heuristic.

Filed from the 2026-09-10 Console rail UX review (review item B4).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Storing an implicit right_open preference (e.g. via reveal logic) no longer permanently disables the auto-open heuristic
- [ ] #2 An explicit user toggle of the Inspector rail still disables auto-open
- [ ] #3 Existing rail preference serialization round-trips unchanged for existing configs (no migration break)
- [ ] #4 Unit tests cover both paths (implicit store keeps heuristic; explicit toggle kills it)
<!-- AC:END -->
