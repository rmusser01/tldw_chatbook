---
id: TASK-22225
title: >-
  v48 policy seeding: skip deleted conversations
status: To Do
assignee: []
created_date: '2026-08-24'
labels:
  - database
  - migration
priority: low
dependencies: []
---

## Description

Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22225).

`DB/ChaChaNotes_DB.py:5953-5970`: the v48 bump seeds
`console_conversation_library_policy` with one row per conversation via
`INSERT ... SELECT id FROM conversations` with no `WHERE deleted = 0` — O(all
conversations ever) inserts inside the boot version-bump transaction, permanently storing
rows for tombstoned conversations.

## Acceptance Criteria

- [ ] The seeding migration (current version at fix time) excludes deleted conversations; a fresh-migration test proves it
- [ ] Existing over-seeded rows are cleaned or explicitly documented as inert
- [ ] Migration remains self-contained (the TASK-21441 lesson)
