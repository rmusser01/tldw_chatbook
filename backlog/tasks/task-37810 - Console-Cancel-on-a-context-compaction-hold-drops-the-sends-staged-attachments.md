---
id: TASK-37810
title: 'Console: Cancel on a context-compaction hold drops the send''s staged attachments'
status: To Do
assignee: []
created_date: '2026-10-29 17:30'
labels:
  - console
  - composer
dependencies: []
references:
  - >-
    backlog/tasks/task-33621.18 -
    Console-Unsent-turn-needs-attention-shelf-outlives-a-turn-that-was-sent-and-its-clipped-Rest-refills-the-answered-prompt.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by code reading during TASK-33621.18's pre-merge review. It is pre-existing and was not reproduced by a running test.

A send held at the context-compaction threshold (TASK-34350) takes the composer's staged attachments with it. The runtime records no unsent turn for that hold, so nothing keeps those attachments. The hold card's Cancel (`_cancel_context_compaction_hold`) puts the text back in the composer and gives the attachments back nowhere.

TASK-33621.18 found the same shape on the Temporary-chat capture pause and fixed it there. The runtime now records the capture pause's unsent turn held off the shelf (`card_held` on the refusal). The card's Cancel then returns the text and the attachments through `return_turn_recovery_to_draft`, and an accepted card action drops the turn. The compaction hold could reuse that path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Cancel on a context-compaction hold puts the held send's text and its staged attachments back in the composer
- [ ] #2 While the hold card is on screen, the unsent-turn shelf still does not offer the held send
- [ ] #3 Compact and send, and Send without compacting, leave no unsent turn behind once the send is accepted
- [ ] #4 A mounted test stages an image, crosses the threshold, cancels the hold, and fails before the fix
<!-- AC:END -->
