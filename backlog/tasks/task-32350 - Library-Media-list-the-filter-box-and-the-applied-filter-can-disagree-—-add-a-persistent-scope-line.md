---
id: TASK-32350
title: >-
  Library Media list: the filter box and the applied filter can disagree — add a
  persistent scope line
status: To Do
assignee: []
created_date: '2026-09-11 06:15'
labels:
  - library
  - media
  - ux
  - critique-10
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The filter box keeps a draft while the list shows the last applied query and the count reads 'Media (9)' with nothing saying which filter is live (A caps 41/47; the draft-vs-applied model is deliberate, the missing scope line is not). Evidence: critique #10 snapshot (.impeccable/critique/2026-09-11T06-13-11Z__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 1f3184655b, captures under the session scratchpad crit10/A and crit10/B).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A scope line under the Media header states the applied filter, type and sort and the count as N of M, with a Clear action
- [ ] #2 Clearing the applied filter also clears the box
- [ ] #3 Pinned
<!-- AC:END -->
