---
id: TASK-32274
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
