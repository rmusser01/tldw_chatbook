---
id: TASK-32815
title: Guard Console footer focus queries during empty-stack shutdown
status: To Do
assignee: []
created_date: '2026-09-18 18:40'
labels:
  - ui
  - console
  - test-health
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-32813 consumer qualification observed a Console archive case complete its interaction assertions and then fail during shutdown: a deferred footer refresh called _console_rail_focus_active after the final screen had been popped. The same exact case passed on isolated rerun. The implicated ChatScreen helper is unchanged from the saved head; this is a sibling path to TASK-32297, which guards the environment poll.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Deferred footer focus checks treat an empty screen stack as inactive and do not raise during shutdown.
- [ ] #2 A deterministic regression reproduces the observed footer path without depending on a timing-sensitive full-app failure.
- [ ] #3 Targeted footer and archive checks pass, with the original incident and remaining lifecycle limits recorded.
<!-- AC:END -->
