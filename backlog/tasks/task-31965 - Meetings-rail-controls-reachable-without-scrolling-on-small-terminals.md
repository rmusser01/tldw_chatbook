---
id: TASK-31965
title: 'Meetings: rail controls reachable without scrolling on small terminals'
status: To Do
assignee: []
created_date: '2026-09-07 05:49'
labels:
  - ui
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
With the app stylesheet loaded, the Meetings rail's Start/Pause/Stop row sits below the fold at 120x32 and smaller (the Sources block above it takes 12+ rows). This predates TASK-31826, which made the rail scrollable so everything is at least reachable. Decide the rail's small-terminal information architecture (hoist the controls above Sources, collapse the probing lines, or a compact mode) and pin it with pilots at 100x30 and 80x24.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Start is visible without scrolling at 100x30 and 80x24 with the bundled stylesheet
- [ ] #2 The voice controls and the learning offer remain reachable at those sizes
- [ ] #3 Pilots at both sizes assert the layout (not only 160x45)
<!-- AC:END -->
