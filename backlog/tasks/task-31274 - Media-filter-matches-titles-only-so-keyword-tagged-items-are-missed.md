---
id: TASK-31274
title: Media filter matches titles only so keyword-tagged items are missed
status: To Do
assignee: []
created_date: '2026-09-04 13:54'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 P2: filtering the list by `day2` — a keyword on four seeded rows — produced `Media (0)` and `No media matched ‘day2’.` (B cap_15). The filter appears to match titles only (cause suspected, not traced). The user's stated sequential-review scenario is a tag/keyword-filtered browse, so a keyword miss undercuts `Review these` over a tag scope, and the empty state does not say what was searched.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The fields the filter searches are documented in the task notes after a code trace
- [ ] #2 Keyword matches are included in the filter, or an explicit `keyword:` syntax exists and the input placeholder says so
- [ ] #3 The empty state names what was searched (e.g. `No media matched “day2” in titles or keywords`)
- [ ] #4 Tests pin keyword matching and the copy
<!-- AC:END -->
