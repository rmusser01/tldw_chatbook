---
id: TASK-166
title: Library repeat-visit snapshot-fetch cost
status: To Do
assignee: []
created_date: '2026-07-11 22:02'
labels:
  - follow-up
  - library
  - performance
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Since screen instance caching was removed (freeze fix, PR #595), LibraryScreen.on_mount runs _refresh_local_source_snapshot on every visit rather than once per cache lifetime, so repeat Library visits re-fetch instead of showing an instant cached view. Consider a short-lived snapshot memo keyed by scope to restore instant repeat visits without reintroducing screen caching.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Repeat Library visits within a short window avoid a redundant full snapshot fetch,No stale data shown beyond the memo window
<!-- AC:END -->
