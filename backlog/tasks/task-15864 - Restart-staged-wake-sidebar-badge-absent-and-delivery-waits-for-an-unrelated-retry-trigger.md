---
id: TASK-15864
title: >-
  Restart-staged wake: sidebar badge absent and delivery waits for an unrelated
  retry trigger
status: To Do
assignee: []
created_date: '2026-08-13 21:44'
labels:
  - fleet
  - console
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR 3a-2 Task 7 live restart (scenario 5): with a durable fleet_unseen mark surviving SIGKILL and an owed run in the ledger, the fresh app rendered NO ◈ badge on the marked conversation's sidebar browser row (Task 4's restart claim covers a fresh screen's first read from the DB — live restart contradicted it on the sidebar surface, possibly because no session tab existed for the conversation). Opening the marked conversation view-cleared the mark and created the session, but the seeded wake did NOT deliver on open — the retry-trigger list (composer poke, terminal transitions, drains, mount) omits session-open, so the wake sat pending until a composer keystroke. Also observed and worth a ruling: a wake deferred while its conversation is being VIEWED view-clears the mark while the ledger still owes the wake — a restart in that window leaves an owed, unmarked run the mount-claim will never seed (same shape as the unmarked mid-run orphan, which is by-design per the corrected spec §3).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After a restart with a staged wake, the marked conversation's sidebar row shows the ◈ badge before the conversation is opened
- [ ] #2 Opening a marked conversation (creating its session) is a wake retry trigger — delivery does not wait for an unrelated keystroke
- [ ] #3 A ruling is recorded (fix or documented limit) for the owed-but-unmarked window: wake deferred in a viewed conversation, then restart
<!-- AC:END -->
