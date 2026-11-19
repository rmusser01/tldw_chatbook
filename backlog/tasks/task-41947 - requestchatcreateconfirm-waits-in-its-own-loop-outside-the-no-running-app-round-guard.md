---
id: TASK-41947
title: "`request_chat_create_confirm` waits in its own loop, outside the no-running-app round guard"
status: To Do
assignee: []
created_date: '2026-11-19 07:30'
labels:
- follow-up
- console
- bug-suspected
priority: medium
references:
- tldw_chatbook/Chat/console_interrupt_rounds.py
dependencies:
- TASK-41637
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41630..41655 follow-up burn-down (2026-11-18/19). Checked on dev d0486978dd unless stated; paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** TASK-41637 made an approval round fail closed when no app is running and no view can answer it (`run_round`). `request_chat_create_confirm` has its own wait loop outside `run_round`, so a chat-create confirmation with no running app may still wait forever.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A chat-create confirmation requested with no running app and no answerable view settles as refused promptly (test, red first)
- [ ] #2 One with an answerable view still waits for the answer
<!-- AC:END -->
