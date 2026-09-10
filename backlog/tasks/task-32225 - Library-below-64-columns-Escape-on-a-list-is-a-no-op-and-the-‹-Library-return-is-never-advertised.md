---
id: TASK-32225
title: >-
  Library below 64 columns: Escape on a list is a no-op and the '‹ Library'
  return is never advertised
status: To Do
assignee: []
created_date: '2026-09-10 14:55'
labels:
  - library
  - layout
  - keyboard
  - critique-9
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
In the single-stage layout Escape on a list does nothing and the footer never names the return control. Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 24.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Escape (or an advertised key) returns to the rail stage; the footer names it
<!-- AC:END -->
