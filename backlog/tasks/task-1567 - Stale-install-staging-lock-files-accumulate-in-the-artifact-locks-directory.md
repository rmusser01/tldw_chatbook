---
id: TASK-1567
title: Stale install-staging lock files accumulate in the artifact locks directory
status: To Do
assignee: []
created_date: '2026-08-01 01:57'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in TASK-595's final review: service.py's _install_staging_lease_key uses a fresh mkdtemp name per install and lease files are never unlinked, so one <sha256>.lock accumulates in locks/ per install, permanently. Needs cleanup on release (or a sweep in reconcile) without reintroducing the GC-vs-live-install race the lease exists to prevent.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Repeated installs do not grow the locks directory without bound
- [ ] #2 Cleanup cannot delete a lock a live installer still holds
<!-- AC:END -->
