---
id: TASK-32344
title: First agent send after restart stalls ~70 s with no activity copy
status: To Do
assignee: []
created_date: '2026-09-10 19:10'
labels:
  - console
  - agents
  - mcp
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On a warm profile, the first Console agent send after an app restart showed an empty assistant row and 'Run: Agent running.' for about 70 seconds before the provider was called; a later send in the same instance took about 5 seconds. Nothing on screen said what the app was waiting on. The built-in MCP server spawn/discovery is the suspected cause; not traced. Source: live approval-card / MCP-permissions UX review 2026-09-10 against dev 3315241674 (snapshot .impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause of the delay is identified and either removed or bounded by a visible timeout with a stated reason.
- [ ] #2 While pre-provider setup runs, the assistant row or status strip says what is happening (for example connecting tools) instead of staying blank.
- [ ] #3 A diagnostic event or test pins the pre-provider setup time budget.
<!-- AC:END -->

## Renumbering provenance

<!-- SECTION:PROVENANCE:BEGIN -->
This task was filed as TASK-32275 on 2026-09-10 at 12:19 PT (19:19 UTC) in the
approval-card / MCP-permissions UX review wave (PR #2574), which makes it the
OLDER arrival against the TASK-32275 that `dev` minted later the same day
("Build searchable conversation archive review and bulk recovery").

It renumbered to TASK-32344 anyway. The bare 2026-08-21 owner rule of TASK-19601
quoted in `scripts/check_backlog_task_ids.py` (older arrival keeps the id) is
superseded by the 2026-09-08 refinement recorded in
`backlog/docs/lessons-backlog-hygiene.md`: **landed-keeps-id trumps
older-keeps-id -- a task already on origin/dev never renumbers; the unmerged
side moves regardless of timestamps, because renumbering landed ids breaks
external references.** The dev-side TASK-32275 is merged and cited from
`backlog/decisions/147-conversation-archive-and-exact-resume.md`,
`backlog/docs/lessons-testing-evidence.md` and two plan/QA records; this task
was cited only from its own unmerged plan. `dev` applied the same refinement
earlier on 2026-09-10 when it renumbered its archive-lifecycle task to
TASK-32300.

Renumbered 2026-09-10. Commit messages on the wave branch `approval-wave-a`
written before this date that cite `task-32275` refer to THIS task; the
dev-side TASK-32275 keeps the id.
<!-- SECTION:PROVENANCE:END -->
