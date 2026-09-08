---
id: TASK-32042
title: 'Library: Conversations select mode is a degraded, truncated copy of Media''s'
status: To Do
assignee: []
created_date: '2026-09-08 14:36'
labels:
  - library
  - conversations
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #7 P1. Conversations' select-mode toolbar renders 'Selec' (truncated 'Select all'), a bare unlabeled marker, and an awkwardly wrapped '0 selected', and its footer omits the space/s hints Media shows. It is the same feature as Media's clean, labelled select mode but inconsistent and less legible. Share the select-toolbar treatment so the two do not diverge.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Conversations select mode shows full, untruncated action labels at supported widths, matching Media's grammar
- [ ] #2 The select toolbar is built from the shared treatment rather than a divergent per-canvas copy
- [ ] #3 Painted pins assert the Conversations select actions are legible (no mid-word truncation) at 235x52 and 100x30
<!-- AC:END -->
