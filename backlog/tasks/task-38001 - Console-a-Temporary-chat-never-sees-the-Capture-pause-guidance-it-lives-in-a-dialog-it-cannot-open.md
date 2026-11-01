---
id: TASK-38001
title: >-
  Console: a Temporary chat never sees the Capture pause guidance; it lives in a
  dialog it cannot open
status: To Do
assignee: []
created_date: '2026-10-30 05:25'
labels:
  - console
  - copy
dependencies: []
references:
  - >-
    backlog/tasks/task-37811 -
    Console-Temporary-chat-capture-copy-promises-capture-that-saving-would-not-turn-on.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The only in-app copy that tells a Temporary chat what Capture On does (each send pauses on the Save chat to capture this send card for This conversation, or only the next send for Next send) is the Temporary guidance in the Trace screen's Exchange Capture Policy dialog. That branch renders only when the dialog has no saved conversation to target, and the Trace screen never opens for a Temporary chat: pressing y in one says 'The active conversation has no persisted trace yet.' (seen live while closing TASK-37811). The dialog a Temporary chat can reach, Trace Privacy from the Conversation Inspector's Capture settings, has no Temporary guidance. Two reviews of TASK-37811 reworded the unreachable copy; a user choosing Capture On in a Temporary chat still gets no warning that its sends will pause.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 In a Temporary chat, the dialog reached from the Conversation Inspector's Capture settings says that Capture On for This conversation pauses each send until Save & Send or Send without capture, and that Next send pauses only that send
- [ ] #2 The unreachable Exchange Capture Policy Temporary branch is either removed or made reachable, and the task notes say which and why
- [ ] #3 A mounted test opens the dialog from a real Temporary chat and pins the guidance; the User Guide matches
<!-- AC:END -->
