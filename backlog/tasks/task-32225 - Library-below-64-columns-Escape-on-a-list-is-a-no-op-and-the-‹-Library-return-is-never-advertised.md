---
id: TASK-32225
title: >-
  Library below 64 columns: Escape on a list is a no-op and the '‹ Library'
  return is never advertised
status: In Progress
assignee: []
created_date: '2026-09-10 14:55'
updated_date: '2026-09-10 17:20'
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing test at 60x24: the footer advertises 'esc focus rail', the rail pane is closed, and Escape moves nothing.
2. Add the narrow single-stage context to _library_route_shortcuts_for_current_state ('esc back to Library') and route Escape to the same seam the '< Library' control uses.
3. Docs + live-verify at 60 columns.
<!-- SECTION:PLAN:END -->
