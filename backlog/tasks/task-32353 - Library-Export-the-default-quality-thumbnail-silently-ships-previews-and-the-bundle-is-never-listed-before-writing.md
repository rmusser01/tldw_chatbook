---
id: TASK-32353
title: >-
  Library Export: the default 'quality: thumbnail' silently ships previews, and
  the bundle is never listed before writing
status: To Do
assignee: []
created_date: '2026-09-11 06:16'
labels:
  - library
  - export
  - critique-10
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The export canvas defaults to 'quality: thumbnail — keeps a small preview image instead of the full file', rendered at the same weight as 'sort'; nothing lists the items or estimates the size before Export bundle (A cap 50). Evidence: critique #10 snapshot (.impeccable/critique/2026-09-11T06-13-11Z__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 1f3184655b, captures under the session scratchpad crit10/A and crit10/B).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Export defaults to full fidelity
- [ ] #2 The canvas shows what the bundle will contain (item count, fidelity, estimated size) before the button is pressed
- [ ] #3 Pinned
<!-- AC:END -->
