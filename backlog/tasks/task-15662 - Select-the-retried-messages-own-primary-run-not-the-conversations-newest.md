---
id: TASK-15662
title: 'Select the retried message''s own primary run, not the conversation''s newest'
status: To Do
assignee: []
created_date: '2026-08-11 21:30'
labels:
  - console
  - agents
  - db
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`_previous_primary_run_id` selects the newest non-superseded primary run for the WHOLE CONVERSATION rather than the run tied to the message being retried. PR 3a-1 Task 4 made this SAFE (the supersede statement now skips any row that is not in a terminal status, so a live wrong-target is left alone rather than destroyed) but the selection itself is still coarser than it should be: retrying an older failed message can still target a newer message's run record.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Retrying a message supersedes the run that produced THAT message, not merely the conversation's newest primary
- [ ] #2 A conversation with several failed messages retried out of order leaves each run tied to its own message
- [ ] #3 The Task 4 terminal-status guard remains in place and its tests still pass
- [ ] #4 A test constructs two failed messages with distinct primary runs and fails when the selection falls back to newest-in-conversation
<!-- AC:END -->
