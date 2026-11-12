---
id: TASK-40578
title: >-
  Console: a chained trace replacement validates every link twice while
  holding the write lock
status: To Do
assignee: []
created_date: '2026-11-12 23:24'
labels:
  - console
  - trace
  - performance
dependencies:
  - TASK-38298
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The chained trace replacement (TASK-38298, ADR-337 amendment 2026-11-06) is written in the IMMEDIATE dispatch-binding transaction. Inside it, the request writer validates the replacement once (every link's range read, plus the check that no live row lies between links), and the replacement writer validates it again before writing. The work is linear in the replaced span. Measured 2026-11-12 on a loaded machine, binding a 1,999-row chain (8 links) took about 73 ms of CPU (109 ms wall) at the median, mostly these reads. At the 256-link cap (about 65,000 rows) the same work may hold the database write lock for seconds, blocking every other writer. Measure it, and consider validating each link once.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Binding a chain is measured at about 2,000 rows and at the 256-link cap, wall and CPU, with the share spent validating links
- [ ] #2 If the duplicate validation is removed or narrowed, every link is still validated once inside the binding transaction, and the tests that pin per-link revalidation stay green
- [ ] #3 The measurements and the decision are recorded in the task notes
<!-- AC:END -->

## Identity

Filed 2026-11-12 from the PR #3515 review (finding F4). A sweep of every ref's task paths and every worktree found a maximum of 40546, so this task takes 40578.
