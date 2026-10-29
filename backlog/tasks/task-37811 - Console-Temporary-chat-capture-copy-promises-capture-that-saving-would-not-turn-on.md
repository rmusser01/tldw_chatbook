---
id: TASK-37811
title: 'Console: Temporary-chat capture copy promises capture that saving would not turn on'
status: To Do
assignee: []
created_date: '2026-10-29 17:30'
labels:
  - console
  - copy
  - needs-owner
dependencies: []
references:
  - >-
    backlog/tasks/task-33621.17 -
    Console-Temporary-chats-stop-every-send-on-an-error-framed-capture-gate-and-its-default-Save-Send-never-sends.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-33621.17 gives a Temporary chat Capture Off until the chat is saved (owner decision 2026-10-27). Saving hands Capture back to whatever the chat would otherwise resolve to. Three pieces of copy say that saving turns capture on:
- the composer menu row 'Save chat (enables capture)';
- the menu hint 'records no traces until saved';
- the User Guide line that saving turns capture on.

That is only true when the saved chat resolves Capture On. With global capture Off, or with a stored conversation Capture Off, saving changes nothing and the copy promises traces that never come. The wording was owner-specified, so the owner decides whether the suffix should depend on what the chat will resolve to once saved.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner's decision on the conditional wording is recorded on this task
- [ ] #2 The menu row, its hint and the User Guide match what saving resolves to: global capture On, global capture Off, and a stored conversation Capture Off
- [ ] #3 A test pins the menu copy for each of the three cases
<!-- AC:END -->
