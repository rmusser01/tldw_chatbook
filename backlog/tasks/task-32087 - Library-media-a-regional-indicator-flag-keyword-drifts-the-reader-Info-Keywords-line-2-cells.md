---
id: TASK-32087
title: >-
  Library media: a regional-indicator flag keyword drifts the reader Info
  Keywords line +2 cells
status: To Do
assignee: []
created_date: '2026-09-08 20:48'
labels:
  - library
  - media
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #8 gap in task-32044. task-32044 made the LIST-ROW keyword-reason suffix flag-pair-aware (never paints a half-flag), but the reader's Info tab renders the raw keyword on its 'Keywords:' line, and a regional-indicator flag pair there still drifts the row frame +2 cells (line measures 237 vs 235; the right border lands at col 236 vs 234). The flag-pair width miscount lives on this second surface, uncovered by 32044.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A keyword containing a regional-indicator flag emoji does not drift the reader Info 'Keywords:' line frame width or overpaint neighbouring content, at 235x52 and 100x30
- [ ] #2 The Info 'Keywords:' rendering uses the same flag-pair-aware width handling task-32044 applied to the list suffix (or an equivalent), rather than mis-measuring the pair
- [ ] #3 A pin paints a flag-keyword Info line and asserts the frame stays aligned
<!-- AC:END -->
