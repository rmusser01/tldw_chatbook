---
id: TASK-182
title: >-
  Console generation feedback: progress indicator, clear composer on accepted
  send, do not persist error text as assistant messages
status: To Do
assignee: []
created_date: '2026-07-12 02:47'
labels:
  - ux
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Core-loop UAT 2026-07-11: between send and first token (30-90s on local models) the Assistant row is completely empty and the composer retains the sent text, implying the message was not sent. Provider failures are stored as assistant message content (polluting history and future model context); a failed first send also sets first_send_completed and each failure accrues another junk saved conversation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A visible generating state exists between send and first token
- [ ] #2 Composer clears (or visibly locks) once the send is accepted
- [ ] #3 Provider errors render as system/status rows and are not persisted as assistant message content
- [ ] #4 A failed first send does not permanently dismiss the onboarding card
<!-- AC:END -->
