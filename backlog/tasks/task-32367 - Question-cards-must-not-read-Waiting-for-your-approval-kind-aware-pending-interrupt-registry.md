---
id: TASK-32367
title: Question cards must not read Waiting for your approval (kind-aware pending-interrupt registry)
status: To Do
assignee: []
created_date: '2026-09-11 01:55'
labels:
  - console
  - approvals
dependencies: []
priority: medium
---

## Description

`has_pending_approval_round` covers all five interrupt kinds (including the question card) while the inspector counts mounted approval cards only, so the transcript activity line can say "Waiting for your approval" while a question card is up. Lane B's minimum fix (R15) corrected the false invariant comment in chat_screen.py; the registry itself is still kind-blind. Found by lane B's final review (task-32345 area).

Source: approval-card / MCP-permissions fix wave 2026-09-10/11 (plan `Docs/superpowers/plans/2026-09-10-approval-card-fix-wave.md`, review snapshot `.impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md`); rider recorded in the lane ledger, not fixed in the wave.

## Acceptance Criteria

- [ ] A pending question card never produces the "Waiting for your approval" activity line; it produces copy that names a question
- [ ] The pending-interrupt registry exposes the interrupt kind to the activity classifier and the inspector count agrees with it
- [ ] A test pins one question card + one approval card → the line names the approval, and a lone question card → the question copy
