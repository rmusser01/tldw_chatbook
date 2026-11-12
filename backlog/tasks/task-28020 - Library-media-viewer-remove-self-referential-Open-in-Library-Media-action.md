---
id: TASK-28020
title: Library media viewer - remove self-referential Open in Library Media action
status: To Do
assignee: []
created_date: '2026-09-02 04:11'
labels:
  - library
  - media-ux
dependencies: []
references:
  - >-
    .impeccable/critique/2026-09-02T04-00-36Z__tldw-chatbook-ui-screens-library-screen-py.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The viewer action Open in Library Media posts NavigateToScreen(media), which aliases back to library (library_screen.py:31174-31195; screen_registry.py:229) - a round-trip to the screen the user is already on. Delete it, or repoint it if a distinct destination was intended.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The action is removed or navigates somewhere distinct from the current view
- [ ] #2 Action-row tests are updated
<!-- AC:END -->
