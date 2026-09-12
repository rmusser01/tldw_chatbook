---
id: TASK-32304
title: >-
  Library below 64 columns — Collections, Conversations and Skills give the
  whole stage to an empty work pane
status: To Do
assignee: []
created_date: '2026-09-11 00:54'
labels:
  - library
  - ux
  - critique-9
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Media fixed this for itself with `list_first_when_empty` (task-32065): with nothing open the narrow stage shows the list, not an empty reader. Collections, Conversations and Skills still hand the entire stage to their empty work pane at 60x24, so the user sees a blank canvas with the list hidden behind a hop. Observed while live-verifying task-32225 (critique-9 shell branch, PR #2585).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At 60x24 with nothing open, Collections, Conversations and Skills show their list, matching Media's rule
<!-- AC:END -->
