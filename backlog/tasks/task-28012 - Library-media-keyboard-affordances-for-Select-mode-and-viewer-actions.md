---
id: TASK-28012
title: Library media - keyboard affordances for Select mode and viewer actions
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
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Select mode is a mouse-only header button; Space on a row does nothing (live-tested) and no footer hint advertises any key. The viewer five-button action row (Edit, Use in Console, Read it later, Open in Library, Delete) has no accelerators - every action is a Tab-walk. Add a key to toggle row selection with a footer hint, and accelerator keys for the viewer actions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Row selection can be entered and toggled from the keyboard, advertised in the footer
- [ ] #2 Viewer actions have bound keys shown in the footer or help panel
- [ ] #3 Existing mouse paths are unchanged
<!-- AC:END -->
