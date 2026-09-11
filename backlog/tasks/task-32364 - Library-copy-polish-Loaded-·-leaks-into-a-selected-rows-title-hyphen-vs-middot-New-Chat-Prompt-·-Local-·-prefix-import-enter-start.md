---
id: TASK-32364
title: >-
  Library copy polish: 'Loaded ·' leaks into a selected row's title, hyphen vs
  middot, 'New Chat', 'Prompt · Local ·' prefix, import 'enter start'
status: To Do
assignee: []
created_date: '2026-09-11 06:19'
labels:
  - library
  - copy
  - critique-10
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Selecting a media row prefixes its title with 'Loaded ·' (A caps 39/47); conversation rows use '5 messages - 16m' where every other list uses '·' and the untitled conversation reads 'New Chat' (A cap 54); every Prompts row begins 'Prompt · Local ·' and 'System + User' is schema-speak (A cap 57); the import footer says 'enter start' but the first Enter only validates (A caps 07/08). Evidence: critique #10 snapshot (.impeccable/critique/2026-09-11T06-13-11Z__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 1f3184655b, captures under the session scratchpad crit10/A and crit10/B).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Status words never prefix a title
- [ ] #2 One separator glyph across lists
- [ ] #3 The import footer names what Enter does at each step
<!-- AC:END -->
