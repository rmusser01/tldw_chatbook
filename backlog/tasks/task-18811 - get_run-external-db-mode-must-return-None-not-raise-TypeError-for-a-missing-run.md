---
id: TASK-18811
title: >-
  get_run external-db mode must return None, not raise TypeError, for a missing
  run
status: To Do
assignee: []
created_date: '2026-08-19 14:47'
labels:
  - research
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
LocalResearchService.get_run promises dict-or-None, but its external-db branch only catches KeyError: an injected db that returns None for a missing run makes _as_local_run(None) raise dict(None) TypeError out of a lookup API. The new external-mode lease path (claim_run -> get_run) inherits this, so a missing run can surface as TypeError instead of the service's not-found contract. Found during PR #1822's external review round and adjudicated then as 'found, not fixed: worth its own task'. The path-backed branch is correct; only the external-db branch needs the guard.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 get_run in external-db mode returns None when the injected db returns None for a missing run, matching the path-backed branch,The external-mode claim_run path resolves a missing run to its documented not-found error rather than a TypeError,Regression test covers an external db double whose get_run returns None
<!-- AC:END -->
