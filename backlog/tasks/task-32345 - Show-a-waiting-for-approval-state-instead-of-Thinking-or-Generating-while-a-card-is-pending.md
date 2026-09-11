---
id: TASK-32345
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

## Renumbering provenance

<!-- SECTION:PROVENANCE:BEGIN -->
This task was filed as TASK-32276 on 2026-09-10 at 12:19 PT (19:19 UTC) in the
approval-card / MCP-permissions UX review wave (PR #2574), which makes it the
OLDER arrival against the TASK-32276 that `dev` minted later the same day
("Connect Console archive search and exact conversation resumption").

It renumbered to TASK-32345 anyway. The bare 2026-08-21 owner rule of TASK-19601
quoted in `scripts/check_backlog_task_ids.py` (older arrival keeps the id) is
superseded by the 2026-09-08 refinement recorded in
`backlog/docs/lessons-backlog-hygiene.md`: **landed-keeps-id trumps
older-keeps-id -- a task already on origin/dev never renumbers; the unmerged
side moves regardless of timestamps, because renumbering landed ids breaks
external references.** The dev-side TASK-32276 is merged and cited from
`backlog/decisions/147-conversation-archive-and-exact-resume.md`,
`backlog/docs/lessons-testing-evidence.md` and two plan/QA records; this task
was cited only from its own unmerged plan. `dev` applied the same refinement
earlier on 2026-09-10 when it renumbered its archive-lifecycle task to
TASK-32300.

Renumbered 2026-09-10. Commit messages on the wave branches
`approval-wave-a/b/c` written before this date that cite `task-32276` refer to
THIS task; the dev-side TASK-32276 keeps the id.
<!-- SECTION:PROVENANCE:END -->
