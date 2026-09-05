---
id: TASK-31577
title: Media DB search - keyword leg escapes LIKE metacharacters asymmetrically
status: To Do
assignee: []
created_date: '2026-09-05 03:23'
labels:
  - library
  - media-ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
search_media's opt-in keyword leg escapes LIKE metacharacters, so a keyword week_3 only matches the literal query week_3 while the title and content legs go through FTS tokenisation and match week 3. The two legs disagree on the same query (wave 4 PR C Task 5 unpinned edge).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The behaviour for _ and % in queries is documented and consistent across legs
- [ ] #2 Tests pin week_3 and week 3 against a week_3 keyword
<!-- AC:END -->
