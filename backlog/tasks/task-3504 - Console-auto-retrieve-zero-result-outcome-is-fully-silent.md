---
id: TASK-3504
title: Console auto-retrieve zero-result outcome is fully silent
status: Won't Do
assignee: []
created_date: '2026-08-07 20:37'
updated_date: '2026-08-22'
labels:
  - console
  - rag
dependencies:
  - TASK-3170
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
This narrow transient-notice proposal is superseded by the accepted
per-conversation Library-control design for PR #1933. That design persists a
bounded device-local zero-match disclosure on the sent turn and gives retrieval
failure a blocking Retry / Send once without Library / Cancel gate. Implementing
this task independently would create conflicting behavior, so it is closed as
Won't Do rather than marked Done.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A zero-result auto-retrieve outcome shows a quiet, non-blocking notice (reusing the existing degraded-notice seam) distinguishing 'retrieval ran and found nothing' from a silent send
- [ ] #2 The notice does not block the send and does not stage the manual run's recovery card
- [ ] #3 A regression test confirms the notice fires exactly on the zero-result auto-retrieve path and not on the already-covered failed/timeout/success paths
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Disposition (2026-08-22): PR #1933's replacement programme owns zero-match and
blocked/dependency-unavailable behavior together. The original acceptance
criteria remain unchecked because this transient-notice design was deliberately
not implemented.
<!-- SECTION:NOTES:END -->
