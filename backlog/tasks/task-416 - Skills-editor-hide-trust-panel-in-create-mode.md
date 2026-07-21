---
id: TASK-416
title: Skills editor - hide trust panel in create mode
status: To Do
assignee: []
created_date: '2026-07-21 15:18'
labels:
  - skills
  - ux
  - trust
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P1 from the 2026-07-21 Skills UX/NNG review (verified live). The create-skill editor renders the trust panel with 'Trust: trusted' and Unlock / Review changes / Approve buttons for a skill that does not exist on disk yet. Nothing has been trusted; the state line is false. After first save the panel correctly shows the real state. NNG heuristic 1 (visibility of system status).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The trust panel is not rendered (or shows an honest pre-save placeholder) while creating a skill that has never been saved,After the first successful save the real trust state renders as today,Covered by tests
<!-- AC:END -->
