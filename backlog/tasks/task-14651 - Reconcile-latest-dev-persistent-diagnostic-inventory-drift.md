---
id: TASK-14651
title: Reconcile latest-dev persistent diagnostic inventory drift
status: To Do
assignee: []
created_date: '2026-08-09 21:05'
labels:
  - testing
  - baseline
  - security
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-2118 final verification reproduced the persistent-diagnostic architecture failure on exact origin/dev f6911b37b after removing TASK-2118's sole branch-owned LLM_API_Calls.py digest delta. The generated-versus-stored baseline differs across 16 unrelated owner entries while persistent sink topology remains unchanged. Review those current-dev diagnostic changes under ADR-029 and reconcile only the accepted baseline; TASK-2118 must not bless them.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every generated-versus-stored owner delta on the recorded dev incident baseline is reviewed under ADR-029
- [ ] #2 The checked inventory is reconciled without changing production behavior or unrelated sink topology
- [ ] #3 The focused persistent-diagnostic architecture test passes after the reviewed refresh
<!-- AC:END -->
