---
id: TASK-2331
title: Runs toolbar Refresh reloads and Re-run gives feedback
status: To Do
assignee: []
created_date: '2026-08-04'
labels:
  - watchlists
  - bug
dependencies: []
priority: medium
---

## Description (the why)

Observed live during UAT batch-2 verification and confirmed in review
(`runs_pane.py:396-397`): the Runs toolbar's Refresh button only re-arms the
action buttons (`_update_action_buttons`) — it never reloads the runs list.
"Re-run source" does run the check but gives no visible feedback that it
started or finished (compounded by Refresh being dead). Pre-existing before
the batch-2 branch; now more visible since run rows carry real accounting.

## Acceptance Criteria (the what)

- [ ] Refresh reloads the runs list (and the selected run's detail) from the
      backend.
- [ ] Re-run source shows an immediate acknowledgment and a completion signal
      (success and failure), consistent with TASK-2309's check-now feedback.
- [ ] Both behaviors have discriminating tests.
