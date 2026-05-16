---
id: TASK-12.2
title: 'Phase 5.2: Active server auth live status'
status: To Do
assignee: []
created_date: '2026-05-16 00:00'
labels:
  - product-maturity
  - phase-5-server-parity
dependencies: []
parent_task_id: TASK-12
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose active-server, auth, reachability, and recovery state in the running app while preserving local mode usability.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 QA walkthrough verifies local mode, missing server, auth-required, unreachable, and ready server states are understandable in the running app.
- [ ] #2 Focused regression evidence proves local mode does not require server credentials and server mode does not silently fall back to local writes.
- [ ] #3 Repo-tracked QA evidence exists under Docs/superpowers/qa/product-maturity/phase-5/.
- [ ] #4 P0/P1 findings are fixed or explicitly accepted according to the spec severity policy.
<!-- AC:END -->
