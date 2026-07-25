---
id: TASK-636
title: Investigate new user session CPU hang recursive asyncio task creation
status: To Do
assignee: []
created_date: '2026-07-25 18:00'
labels:
  - followup
  - uat
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UAT 2026-07-25 (scratchpad/uat-new-user): one of two clean new-user sessions escalated to a 99-100% CPU hang for 14+ minutes after entering Settings-RAG. Stack sample (preserved in the UAT evidence dir) shows recursive asyncio task creation with NO capture-related frames - a separate mechanism from the task-627 mouse-capture leak, explicitly NOT fixed by it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Root cause identified from the preserved stack sample + repro attempt
- [ ] #2 Fix or mitigation lands with a regression test
- [ ] #3 New-user session survives Settings-RAG entry without CPU escalation
<!-- AC:END -->
