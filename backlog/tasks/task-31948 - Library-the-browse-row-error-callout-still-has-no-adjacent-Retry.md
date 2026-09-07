---
id: TASK-31948
title: Library - the browse-row error callout still has no adjacent Retry
status: To Do
assignee: []
created_date: '2026-09-07 08:25'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR G Task 3 concern (2026-09-05): PR G gave the Media load path a failure callout with a Retry beside it, but #library-canvas-error still paints a message with no action next to it, so the only recovery from a failed browse row is to leave the surface and come back.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A browse error a refetch can clear carries a Retry adjacent to the message
- [ ] #2 Retry re-runs the same fetch and clears the callout on success
- [ ] #3 A painted pin covers the callout and its Retry
<!-- AC:END -->
