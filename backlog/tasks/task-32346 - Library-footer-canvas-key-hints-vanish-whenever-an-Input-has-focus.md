---
id: TASK-32346
title: 'Library footer: canvas key hints vanish whenever an Input has focus'
status: To Do
assignee: []
created_date: '2026-09-11 06:14'
labels:
  - library
  - ux
  - critique-10
  - keyboard
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After submitting a search (or typing in any Library Input) the footer reads 'typing in field | F6 next pane' while the canvas keys u (use Library context in Console), o (open evidence) and / still work and are named nowhere; keyboard-only users lose their key map at the moment of use (A caps 14/15; B marks the footer-hint claim contradicted). Not pinned by name. Evidence: critique #10 snapshot (.impeccable/critique/2026-09-11T06-13-11Z__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 1f3184655b, captures under the session scratchpad crit10/A and crit10/B).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The canvas verbs stay visible while an Input has focus, prefixed by the field state (e.g. 'typing in field · esc leaves field | u … | o …')
- [ ] #2 When width is short, 'F6 next pane' is dropped before any canvas verb
- [ ] #3 Pinned on Search/RAG and the Media list
<!-- AC:END -->
