---
id: TASK-3504
title: Console auto-retrieve zero-result outcome is fully silent
status: To Do
assignee: []
created_date: '2026-08-07 20:37'
labels:
  - console
  - rag
dependencies:
  - TASK-3170
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-3170's Task 8 (Console send-path auto-retrieve injection) deliberately made a zero-result auto-retrieve outcome clear its in-flight 'Retrieving...' placeholder without staging the manual run's blocking recovery card -- correct, since staging that card would lock the composer until the user found the un-stage button. But the accepted trade-off means a zero-result auto-retrieve is now completely silent: the send proceeds with no evidence and nothing tells the user retrieval ran and found nothing, unlike the manual chip run (which shows the recovery card) or a failed/timed-out auto-retrieve (which shows a quiet notice). A user relying on auto-retrieve for grounding has no way to know a given send went out ungrounded because the query matched nothing, versus because retrieval succeeded. The existing degraded-notice seam (used for failed/timeout outcomes) is the natural place to add one more quiet notify call for the empty case.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A zero-result auto-retrieve outcome shows a quiet, non-blocking notice (reusing the existing degraded-notice seam) distinguishing 'retrieval ran and found nothing' from a silent send
- [ ] #2 The notice does not block the send and does not stage the manual run's recovery card
- [ ] #3 A regression test confirms the notice fires exactly on the zero-result auto-retrieve path and not on the already-covered failed/timeout/success paths
<!-- AC:END -->
