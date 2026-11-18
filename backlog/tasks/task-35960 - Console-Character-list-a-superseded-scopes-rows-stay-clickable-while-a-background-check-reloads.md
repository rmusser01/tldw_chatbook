---
id: TASK-35960
title: >-
  Console Character list: a superseded scope's rows stay clickable while a
  background check reloads
status: To Do
assignee: []
created_date: '2026-10-21 01:31'
labels:
  - console
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Since TASK-33620.15.6 a background Console Character display check that finds the scope changed reloads inside one finite callback without first entering the Refreshing state. The previous scope's list therefore stays visible and clickable for the whole reload (groups, unavailable details and the fence pair), where before only the short scope probe ran with it showing. After a character pick (invalidate_scope) or a profile or database switch, that list belongs to the old character or profile. Nothing opens across profiles, because activation carries its own data-profile fence, but the user can click a row that no longer belongs to what they are looking at, and the click then does nothing or opens the old context. The rows should read as stale while the displayed scope is known to be out of date.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 While the displayed scope fingerprint is cleared or no longer matches the ambient scope, the list's rows are visibly marked stale and activating one does not open the old scope's conversation
- [ ] #2 A background check that finds the scope unchanged leaves the rows fully interactive with no visible flicker
- [ ] #3 Tests cover a character pick and a profile switch during a held background reload
<!-- AC:END -->
