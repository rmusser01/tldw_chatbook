---
id: TASK-24611
title: >-
  Reorder Inspect rail sections to decision order and default-collapse review
  sections
status: To Do
assignee: []
created_date: '2026-08-30 00:55'
labels:
  - console
  - ux
  - inspector
  - critique-2026-08-29
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The rail mounts its sections in DOM order rather than decision order: Changed files, a post-hoc review artefact, sits third, above everything describing the current send, while the one gather-evidence-before-sending control is the last widget in the last section and reachable only past the fold. Measurement found 11 bounded sections plus 17 boundary anchors, with only 2 or 3 visible without scrolling at 120 columns and below.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Section order follows authority, then what the user can change, then live state, then after-the-fact review
- [ ] #2 Changed files and Session Settings default to collapsed
- [ ] #3 The library-search control is reachable without scrolling at 120 columns
- [ ] #4 Per-section collapse state persists across turns and session switches
<!-- AC:END -->
