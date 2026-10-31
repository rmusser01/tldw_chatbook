---
id: TASK-38422
title: 'Console: a partly blocked Evidence row raises no Inspect rail badge'
status: To Do
assignee: []
created_date: '2026-10-31 03:31'
labels:
  - console
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Since TASK-33621.11 the Inspect rail badge reads each Inspector row's status, not its words. An Evidence row with some references blocked ('Some evidence needs review: 1 blocked') is status ready because the next send is not blocked, so it raises no badge at all. Before, the word 'blocked' in that copy raised 'blocked'. Partial evidence problems now have no signal on the closed rail. Consider a distinct row status or code, such as 'review', with its own badge.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A partly blocked Evidence row shows a distinct, non-blocking signal on the closed Inspect handle, or the owner decides none is wanted
- [ ] #2 The signal comes from the row's status or code, never its copy
- [ ] #3 A test drives a partly blocked bundle through the real Inspector rows and rail badge
<!-- AC:END -->
