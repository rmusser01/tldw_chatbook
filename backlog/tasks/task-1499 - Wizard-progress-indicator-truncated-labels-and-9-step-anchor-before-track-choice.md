---
id: TASK-1499
title: >-
  Wizard progress indicator: truncated labels and 9-step anchor before track
  choice
status: Done
assignee: []
created_date: '2026-07-31 00:22'
updated_date: '2026-07-31 01:38'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX UAT: Welcome shows nine dots with clipped labels ('Notes sy', 'Appeara', 'Protect') before the user picks a track, anchoring perceived effort at 9 when the recommended path is 4.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No progress label truncates at 120 cols
- [ ] #2 Welcome either shows the quick-track count or numbers-only until a track is chosen
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Container defaults to TRACK_QUICK matching the preselected Welcome radio (Step 1 of 4 anchor); titles shortened (Notes/Style/Protect) under the 8-char progress budget with a guard test.
<!-- SECTION:NOTES:END -->
