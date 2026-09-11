---
id: TASK-32359
title: 'Library rail: active and focused rows are visually identical'
status: To Do
assignee: []
created_date: '2026-09-11 06:18'
labels:
  - library
  - accessibility
  - critique-10
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Both the active destination and the focused row render bold + underline with backgrounds three RGB units apart (B D9, caps 16/09 .ansi). Focus elsewhere is shown by shape; the rail is the exception. Evidence: critique #10 snapshot (.impeccable/critique/2026-09-11T06-13-11Z__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 1f3184655b, captures under the session scratchpad crit10/A and crit10/B).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The focused rail row is distinguishable from the active row by a glyph or shape, not colour alone
- [ ] #2 Pinned with a painted assertion
<!-- AC:END -->
