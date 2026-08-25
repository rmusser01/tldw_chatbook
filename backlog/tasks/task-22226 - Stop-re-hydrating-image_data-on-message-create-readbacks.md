---
id: TASK-22226
title: >-
  Stop re-hydrating image_data on message-create readbacks
status: To Do
assignee: []
created_date: '2026-08-24'
labels:
  - database
  - chat
priority: low
dependencies: []
---

## Description

Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22226).

Pre-existing shape (the select list grew since the pin). `Chat/chat_persistence_service.py:
1324-1382` re-reads the just-written message up to three times per create (feedback +
citation paths), each via `get_message_by_id` (`DB/ChaChaNotes_DB.py:10965`) which
hydrates the `image_data` BLOB — MBs copied per image message persist.

## Acceptance Criteria

- [ ] Create-path readbacks use a projection without BLOB columns, or reuse already-known values
- [ ] Measured before/after on an image message persist
- [ ] No change to what callers receive (shape-compatible)
