---
id: TASK-457
title: >-
  Console first-feedback cluster follow-ups: cold-send optimistic echo, rail
  pressed state, inspector-during-run
status: To Do
assignee: []
created_date: '2026-07-22 02:20'
labels:
  - console
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remaining pieces of the task-351 first-feedback finding (Console UX review j4-first-feedback-latency-cluster) not covered by the warm send-echo fix. Sub-symptom (a)-cold: on a cold provider the user's message still waits on the readiness probe before appearing — needs an optimistic user-append BEFORE resolve_for_send with an honest block/error row on failure (a real blocked-send behaviour change). Sub-symptom (b): rail conversation rows are plain Buttons with no pressed/loading feedback, so a slow/failed open reads as a dead click. Sub-symptom (c): the Inspector toggle during a run shows no immediate acknowledgment.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Cold-provider first send echoes the user's message before the readiness probe resolves; a not-ready provider shows an honest block/error row instead of a silently-dropped message,Rail conversation rows show a pressed/loading acknowledgment on click,Inspector toggle acknowledges immediately (within ~100ms) even during an active run
<!-- AC:END -->
