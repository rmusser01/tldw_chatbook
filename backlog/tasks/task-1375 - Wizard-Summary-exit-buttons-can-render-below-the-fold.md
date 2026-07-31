---
id: TASK-1375
title: Wizard Summary exit buttons can render below the fold
status: Done
assignee: []
created_date: '2026-07-30 10:03'
updated_date: '2026-07-31 00:23'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
With tall summary content (e.g. a long wrapped config path) the Done/Start-chatting buttons fall below the visible step area at 40 rows; focus reaches them but they are not visible. Ensure the actions row stays pinned or the step scrolls to it. Found during PR-1095 UAT.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Summary exit buttons visible at 120x40 with a 200-char config path
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Superseded by TASK-1495: root cause is the non-scrolling step viewport (affects key input and discovery too, not just long config paths).
<!-- SECTION:NOTES:END -->
