---
id: TASK-25714
title: Conversation database failure gives recovery advice that cannot work
status: To Do
assignee: []
created_date: '2026-08-31 05:07'
labels:
  - console
  - ux-review
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When the conversation database cannot be opened, Console tells the user to restart and check the app log. Restarting cannot fix an on-disk fault, and the underlying exception is never written to the log, so the instruction leads nowhere. In the observed case a single corrupt index made the whole product unusable while table data and foreign keys were intact and one REINDEX restored it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The underlying database exception is written to the app log before the user is told to consult it
- [ ] #2 Console distinguishes a repairable integrity fault from an unrecoverable one and says which it is
- [ ] #3 A repairable fault offers an in-app repair action rather than only a restart suggestion
<!-- AC:END -->
