---
id: TASK-17663
title: 'Watchlists: destination action-outcome test red on dev (collections param)'
status: To Do
assignee: []
created_date: '2026-08-18'
labels:
  - watchlists
  - test-health
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_destination_shells.py::test_destination_action_buttons_explain_their_outcome[watchlists_collections]` fails on clean origin/dev — verified 2026-08-18 in a detached baseline worktree at `dd1a82146` (solo run, same failure on a task-17656/17660 branch that touches neither watchlists nor this test's fixtures). Found during task-17660's collateral run. Needs the usual decide-by-reproducing pass: either a watchlists collections action button stopped explaining its outcome (real copy/tooltip regression) or the pin is stale against an intended change from a recent watchlists/destination merge.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 The failing parameterization is green on dev, with the regression fixed or the pin updated to the intended contract (decided by reproducing the surface live first)
- [ ] #2 The task records which merge introduced the red
<!-- AC:END -->
