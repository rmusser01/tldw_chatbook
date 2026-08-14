---
id: TASK-16198
title: 'Fix dev-red knowledge_entry test: real network egress in teardown'
status: To Do
assignee: []
created_date: '2026-08-14 03:05'
labels:
  - tests
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The knowledge_entry suite fails on pristine dev with a network-egress teardown error naming real remote IPs (`104.18.3.115:443`, `104.18.2.115:443`) — reproduced byte-for-byte on clean base `c3ed2854a` during TASK-15471. This is the egress-guard (TASK-15211 programme: tests must not reach the network; the guard now blocks by default) catching a genuine leak: something in the knowledge_entry path opens a real connection during teardown. Find the egress source, stub or gate it, and make the suite green under the guard. Absent from known-red batch task-15766. Surfaced during TASK-15471 (per-click I/O off-loop, PR #1625 merged `172ada448`) and its concurrency review; evidence in the session review record.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The knowledge_entry suite passes on a pristine checkout with the egress guard active
- [ ] #2 The egress source is named in the notes (what connects, from where, why in teardown)
- [ ] #3 No weakening of the egress guard itself
<!-- AC:END -->
