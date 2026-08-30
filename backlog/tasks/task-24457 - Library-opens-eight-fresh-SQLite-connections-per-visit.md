---
id: TASK-24457
title: Library opens eight fresh SQLite connections per visit
status: To Do
assignee: []
created_date: '2026-08-29'
labels:
  - performance
  - library
  - database
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Each visit to the Library screen opens 8 new SQLite connections through
`DB/private_sqlite.py::_connect_registered_sqlite`. Every one of them pays the full connection
setup cost: `PRAGMA journal_mode=WAL`, `PRAGMA synchronous=NORMAL` and
`PRAGMA foreign_keys=ON`, plus the file open itself.

The connections are not pooled or reused across visits, so the cost is paid again on every
navigation to the screen.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Repeat visits to the Library screen open materially fewer new SQLite connections than the first visit
- [ ] #2 Connection lifetime and thread-affinity rules are respected -- no connection is shared across threads in a way the DB layer forbids
- [ ] #3 The private-path policy enforcement in `_connect_registered_sqlite` still runs for every logical database open
- [ ] #4 Library screen data loads correctly on first and repeat visits
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
NOT IMPLEMENTED in the 2026-08-29 review pass.

Root cause established and shared with task-24452: the 8 connections per visit are not a leak in
the DB layer, they are a consequence of the app constructing a NEW `LibraryScreen` on every visit
(verified live -- three visits, three distinct instance ids), each building its own handles
through `DB/private_sqlite.py::_connect_registered_sqlite`.

That means the fix is either screen instance reuse (task-24452's architectural change) or an
app-scoped connection cache of the kind task-24456 used for config reads. The latter is the
tractable one and is independent of the architecture question, but it must preserve the
private-path policy enforcement `_connect_registered_sqlite` performs on every logical open, and
must respect SQLite thread affinity -- neither of which was verified in this pass.
<!-- SECTION:NOTES:END -->
