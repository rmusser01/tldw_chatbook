---
id: TASK-32276
title: >-
  Show a waiting-for-approval state instead of Thinking or Generating while a
  card is pending
status: To Do
assignee: []
created_date: '2026-09-10 19:11'
labels:
  - console
  - approvals
  - ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
With an approval card armed, the assistant row reads 'Thinking... <elapsed>', the status strip says 'Run: Agent running.', and the inspector says 'Live work: Generating...' and 'Run: Recovery required'. The activity table in UI/Console_Modules/agent.py has no waiting state; 'Waiting for approval' copy already exists unused in console_send_authority_summary.py. Source: live approval-card / MCP-permissions UX review 2026-09-10 against dev 3315241674 (snapshot .impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 While a round waits on the user, the assistant activity line, the status strip run chip and the inspector summary all say the run is waiting for the user's approval, with elapsed time.
- [ ] #2 'Recovery required' and 'Generating...' never render for a healthy pending approval.
- [ ] #3 Tests cover the pending-approval rendering of all three surfaces.
<!-- AC:END -->
