---
id: TASK-33166
title: Close DreamsDB per-thread connections on close - fd-sentinel root cause
status: To Do
assignee: []
created_date: '2026-09-29 19:02'
labels:
  - dreams
  - tech-debt
  - testing
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The test-suite fd-growth sentinel (Tests/conftest.py, warn-only, threshold 200) has been nudged over by Dreams test fixtures across two PRs (#2817 measured 197/200 at HEAD; #2890 fixtures pushed runs to ~295). Root cause is not the fixtures: DreamsDB.close() closes only the CALLING thread connection while the thread-local idiom keeps one connection per worker thread that ever touched the DB, and each connection also pins private_sqlite admission state. Fix close() to sweep ALL thread-local connections (registry of weakrefs or equivalent), keeping the Library_Collections_DB idiom intact; re-baseline the sentinel afterwards. Filed from the Dreams Phase 2 final-review deferred triage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 DreamsDB.close() releases every thread-local connection it created, verified by an fd/conn-count test,fd-growth sentinel returns below threshold on the Dreams suites without loosening it,Library_Collections_DB-pattern behavior (thread-local reads) unchanged
<!-- AC:END -->
