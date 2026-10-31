---
id: TASK-36962
title: >-
  Console: a failed Library-activity durable read shows no activity until the
  conversation's activity revision moves
status: To Do
assignee: []
created_date: '2026-10-26 03:21'
labels:
  - console
  - performance
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the TASK-33620.15.4 review (2026-10-25). The off-loop Library-activity read (library_activity.py _read_durable_rows/_durable_read) caches () after a failed read under its (database, conversation, activity revision) key, so a transient failure keeps the activity footers and the control bar's trajectory rows empty until that conversation's activity revision moves. The base re-read on any projection token move (a selection or active-path change). Caching the failure is deliberate: dropping it would let a persistently failing read restart on every projection the landing triggers, the same self-sustaining loop the review found in the appearance re-read. A fix needs a bounded retry (for example, retry a failed key once after a delay or on the next token move), not uncached failures.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A transient failed durable read is retried and its rows render without the activity revision moving
- [ ] #2 A read that keeps failing starts at most a bounded number of re-reads per run (no read or publication per poll tick)
- [ ] #3 A deterministic test pins both, red on the pre-fix head
<!-- AC:END -->
