---
id: TASK-1695
title: Adopt the descriptor + source-map contract for per-file artifact URLs
status: To Do
assignee: []
created_date: '2026-08-01 07:02'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reconciliation item 2: the parallel TASK-595 branch passes an explicit credential-free source map alongside the exact descriptor, so per-file URLs never need to enter the frozen descriptor schema. Adopt that contract in acquisition.py, replacing the current hard CatalogError refusal of multi-file artifacts. Supersedes TASK-1693 (descriptor schema v2), which should be closed as unnecessary if this lands.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Multi-file artifacts provision successfully with per-file URLs supplied by the caller's source map
- [ ] #2 No per-file url field is added to the frozen ArtifactFile schema
- [ ] #3 TASK-1693 is closed as superseded
<!-- AC:END -->
