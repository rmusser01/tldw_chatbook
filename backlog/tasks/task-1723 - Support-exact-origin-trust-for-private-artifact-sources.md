---
id: TASK-1723
title: Support exact-origin trust for private artifact sources
status: To Do
assignee: []
created_date: '2026-08-01 16:06'
labels:
  - security
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Acquisition threads trusted_origins as a frozenset of hostnames, which grants trust to that host on any scheme and port -- including across a redirect hop. It is fixture-only today, but a self-hosted or LAN artifact source (a likely TASK-596 case) would use the same seam and inherit the widened trust. The parallel TASK-595 branch added trusted_private_origins: an exact scheme/host/effective-port allowlist enforced on every hop, alongside the existing hostname trust. Adopt it when a real private source lands; not before.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A private source pinned to one origin is not trusted after a redirect to a different port or scheme
- [ ] #2 Default-port normalization is handled so https://h and https://h:443 are the same origin
- [ ] #3 Existing hostname-based trusted_origins behavior is unchanged for current callers
<!-- AC:END -->
