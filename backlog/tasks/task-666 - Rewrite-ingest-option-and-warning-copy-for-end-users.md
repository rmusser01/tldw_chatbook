---
id: TASK-666
title: Rewrite ingest option and warning copy for end users
status: To Do
assignee: []
created_date: '2026-07-26 03:26'
labels:
  - ingest
  - ux
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The per-type options panel titles the collapsible with a raw dump of internal field names and values, the missing-tooling hints read as truncated sentences, and a path that cannot be found offers a Retry action when the only fix is to correct the path. A first-time user cannot tell what any of it means.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The options panel title describes the settings in plain language rather than internal field names
- [ ] #2 Missing-tooling hints read as complete sentences and name the install command
- [ ] #3 A not-found path offers a correction affordance rather than Retry
- [ ] #4 Retry remains available for failures that are genuinely retryable
<!-- AC:END -->
