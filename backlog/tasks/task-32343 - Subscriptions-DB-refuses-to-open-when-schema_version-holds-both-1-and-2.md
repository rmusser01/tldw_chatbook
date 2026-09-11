---
id: TASK-32343
title: Subscriptions DB refuses to open when schema_version holds both 1 and 2
status: To Do
assignee: []
created_date: '2026-09-10 19:10'
labels:
  - db
  - watchlists
  - bug
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Startup crashes with 'Unsupported subscriptions schema version' when the schema_version table contains rows 1 and 2. The fresh-create path inserts 2 with INSERT OR IGNORE while the migration path deletes then inserts; a scratch profile reached the two-row state after an abnormal exit. The failure is a raw traceback with no recovery hint and blocks the whole app. Source: live approval-card / MCP-permissions UX review 2026-09-10 against dev 3315241674 (snapshot .impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A schema_version table containing the current version alongside older rows opens and is normalised to the current version instead of raising.
- [ ] #2 The sequence that produces the two-row state is reproduced by a test and can no longer occur.
- [ ] #3 An unsupported version fails at boot with an actionable message naming the file, not a raw traceback.
<!-- AC:END -->

## Renumbering provenance

<!-- SECTION:PROVENANCE:BEGIN -->
This task was filed as TASK-32274 on 2026-09-10 at 12:19 PT (19:19 UTC) in the
approval-card / MCP-permissions UX review wave (PR #2574), which makes it the
OLDER arrival against the TASK-32274 that `dev` minted later the same day
("Repair workspace archive recovery and session close clarity").

It renumbered to TASK-32343 anyway. The bare 2026-08-21 owner rule of TASK-19601
quoted in `scripts/check_backlog_task_ids.py` (older arrival keeps the id) is
superseded by the 2026-09-08 refinement recorded in
`backlog/docs/lessons-backlog-hygiene.md`: **landed-keeps-id trumps
older-keeps-id -- a task already on origin/dev never renumbers; the unmerged
side moves regardless of timestamps, because renumbering landed ids breaks
external references.** The dev-side TASK-32274 is merged and cited from
`backlog/decisions/147-conversation-archive-and-exact-resume.md`,
`backlog/docs/lessons-testing-evidence.md` and two plan/QA records; this task
was cited only from its own unmerged plan. `dev` applied the same refinement
earlier on 2026-09-10 when it renumbered its archive-lifecycle task to
TASK-32300.

Renumbered 2026-09-10. Commit messages on the wave branches
`approval-wave-a/b/c` written before this date that cite `task-32274` refer to
THIS task; the dev-side TASK-32274 keeps the id.
<!-- SECTION:PROVENANCE:END -->
