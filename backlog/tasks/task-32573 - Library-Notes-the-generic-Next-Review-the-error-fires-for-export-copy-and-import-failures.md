---
id: TASK-32573
title: >-
  Library Notes: the generic "Next: Review the error" fires for export, copy and
  import failures
status: To Do
assignee: []
created_date: '2026-09-14 22:44'
labels:
  - library
  - notes
  - rider
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by group 4. Several Notes failure paths fall back to a generic next-action clause, 'Next: Review the error', which names no action a reader can take — there is nowhere in the app called 'the error'. It fires for export, copy and import failures, all of which have a real next action (retry the export to a different destination, copy again, reopen the import receipt). The wave's rule elsewhere (task-32534, task-32545) is that a row states its category when the cause cannot be named, and names a control when it can.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Export, copy and import failures each name a next action that exists on screen
- [ ] #2 Where no action can be named, the line states the failure's category rather than 'Review the error'
- [ ] #3 A pin drives each of the three real routes and asserts the shipped clause
<!-- AC:END -->
