---
id: TASK-32352
title: >-
  Library Collections: the rail row opens 'Quick Capture', its empty state
  blames unset filters and names an absent action, and its pager is not disabled
  at 0 of 0
status: To Do
assignee: []
created_date: '2026-09-11 06:16'
labels:
  - library
  - collections
  - critique-10
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The rail row 'Collections (0)' opens a canvas titled Quick Capture whose only message is 'No captures match this scope. Clear filters or save a URL with Quick Capture.' (A caps 16/58); Previous/Next render enabled at '0-0 of 0' while the identical Trash pager renders disabled (B D6 cap 52). Overlaps the open product decision task-32057. Evidence: critique #10 snapshot (.impeccable/critique/2026-09-11T06-13-11Z__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 1f3184655b, captures under the session scratchpad crit10/A and crit10/B).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One name for the feature in the rail and on the canvas
- [ ] #2 A true empty state that does not mention filters unless one is set and does not cite an action absent from the screen
- [ ] #3 The pager is disabled at 0 of 0 like Trash's
<!-- AC:END -->
