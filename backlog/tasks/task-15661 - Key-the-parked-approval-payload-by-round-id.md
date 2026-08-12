---
id: TASK-15661
title: 'Key the parked approval payload by round id (fleet F7)'
status: To Do
assignee: []
created_date: '2026-08-11 21:30'
labels:
  - console
  - agents
  - approvals
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The parked approval payload lives in a single slot, so two approval rounds that are parked at the same time overwrite each other. This is pre-existing for sibling sub-agents within one turn and is already documented in the file as an accepted limitation, but cross-turn survivors (PR 3a-1) widen the window in which two rounds can be parked together. Key the payload by round id so each parked round keeps its own.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Two rounds parked at the same time each retain their own payload
- [ ] #2 Answering one parked round does not alter or clear the other's payload
- [ ] #3 The accepted-limitation comment in the source is removed rather than reworded
- [ ] #4 A test parks two rounds concurrently and fails when the slot is shared again
<!-- AC:END -->
