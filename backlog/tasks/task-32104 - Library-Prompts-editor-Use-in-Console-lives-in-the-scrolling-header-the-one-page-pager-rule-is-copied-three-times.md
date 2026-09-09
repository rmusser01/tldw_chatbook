---
id: TASK-32104
title: >-
  Library Prompts editor: 'Use in Console' lives in the scrolling header; the
  one-page pager rule is copied three times
status: To Do
assignee: []
created_date: '2026-09-08 22:42'
labels:
  - library
  - prompts
  - cleanup
  - critique-8
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the task-32074/32067 reviews (PR #2528): 'Use in Console' sits on its own row inside the scrolling editor header, so it leaves the viewport at the bottom of a long prompt (making it fixed means restructuring the editor shell); the hide-when-one-page pager rule is spelled out in `library_media_canvas.py`, `library_conversations_canvas.py` and `library_prompts_canvas.py` with one shared `single_page` flag — a fourth surface needs a fourth copy. Rider from the critique-8 fix wave reviews (plan Docs/superpowers/plans/2026-09-08-library-crit8-wave.md; wave PRs #2519 #2523 #2524 #2525 #2528 #2531).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 'Use in Console' stays reachable without scrolling, or the guide states the scroll position where it lives
- [ ] #2 The one-page pager rule is one helper used by all three canvases
<!-- AC:END -->
