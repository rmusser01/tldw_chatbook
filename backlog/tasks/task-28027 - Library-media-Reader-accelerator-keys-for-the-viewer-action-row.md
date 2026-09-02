---
id: TASK-28027
title: Library media Reader - accelerator keys for the viewer action row
status: To Do
assignee: []
created_date: '2026-09-02 06:57'
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
Split from task-28012 (which delivered select-mode keyboard access on the LIST). The Reader's action row (Find / Read later / Use in Console / More, with Edit metadata / Open original / Open manager / Move to trash under More) has no accelerator keys - every action is a Tab-walk (Alex persona red flag from the 2026-09-01 critique). Give the common Reader actions bound keys, advertised in the footer or F1 help, without stealing the printable keys the search/filter inputs need. Note existing Reader keys already taken: / (focus search), F6 (pane), ] / [ (next/prev item, task-28005), enter (next match when searching, task-28011).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The common Reader actions (at least Find, Read later, Use in Console, Move to trash) have bound keys
- [ ] #2 The keys are advertised in the viewer footer or F1 help and gated to the Reader
- [ ] #3 A focused search/filter input still receives those printable keys
<!-- AC:END -->
