---
id: TASK-31632
title: >-
  Library media - one recovery callout for load failures with the reason and
  Retry adjacent
status: To Do
assignee: []
created_date: '2026-09-05 06:18'
labels:
  - library
  - media-ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #5 P1: three load-failure sentences (Couldn't load page 1., Couldn't load media. Check the local Library and retry., Library source services unavailable; retry Library later.) render as bare text with no reason, while the only Retry sits 34 rows below in the pager and the service wall's sole control, Continue, leaves Library for Home. The service wall is a 5-second source-snapshot timeout collapsed into one static string by a bare except, so a transient failure reads as an indefinite outage and never self-heals.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every Media load failure renders in one recovery callout with a tinted state border that names what failed, why, and what to do, with Retry inside the callout
- [ ] #2 A snapshot timeout is distinguished from a hard failure and re-tries on return to Library
- [ ] #3 Continue either dismisses in place or is renamed to what it does
- [ ] #4 Tests cover the three failure paths
<!-- AC:END -->
