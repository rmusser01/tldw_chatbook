---
id: TASK-2300
title: Watchlists Selects render empty option lists
status: To Do
assignee: []
created_date: '2026-08-04'
labels:
  - watchlists
  - bug
  - uat-2026-08-04
dependencies: []
priority: high
---

## Description (the why)

UAT (2026-08-04, live tmux, fresh profile): two Selects on the Watchlists
screen open with no options at all. The Items tab's status filter opens an
empty floating overlay (a bare border, nothing selectable), and the New Rule
form's condition Select displays "No items". Both are dead controls; the
items filter one compounds TASK-2301 into ingested/ignored items being
unreachable. Likely one root cause (option population), possibly interacting
with the recently-added PruneSafeSelect guard — diagnose before fixing.

UAT findings F30 (critical), F36.

## Acceptance Criteria (the what)

- [ ] The Items status filter opens a populated option list covering every
      item status the backend can produce, and picking one filters the list.
- [ ] The New Rule condition Select offers the real condition vocabulary.
- [ ] The root cause is identified and recorded in the task notes (including
      whether PruneSafeSelect was involved), with a regression test that
      fails when option population breaks again.
- [ ] Verified live in a real terminal, not only under pytest.
