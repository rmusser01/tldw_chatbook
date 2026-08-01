---
id: TASK-1694
title: >-
  Adopt the service-owned payload-subtree finalization seam from the parallel
  TASK-595 branch
status: To Do
assignee: []
created_date: '2026-08-01 07:02'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reconciliation decision (Docs/superpowers/reviews/2026-08-01-task-595-duplicate-implementation-reconciliation.md, item 1): port the download-stage API from codex/task-595-managed-downloads-v2 (service.py +653, test_service.py +528, in the GitHub/tldw_chatbook clone) — a marked, contained stage whose payload/ subtree is verified and RENAMED into the immutable destination, with resume metadata held outside that subtree. Retarget acquisition._install_artifact at the finalization seam instead of install(consume_source=True). This removes the sibling-sidecar workaround and structurally eliminates the class of bug where a retryable install failure destroys the resumable download. Do this BEFORE TASK-596 builds on the layer.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Verified payload files are promoted by renaming the stage's payload subtree; no second staging copy occurs for remote acquisition
- [ ] #2 Resume metadata cannot reside inside what gets promoted
- [ ] #3 A retryable install/finalization failure leaves the durable partial resumable (regression test)
- [ ] #4 The ported stage tests from the parallel branch pass unmodified in intent
<!-- AC:END -->
