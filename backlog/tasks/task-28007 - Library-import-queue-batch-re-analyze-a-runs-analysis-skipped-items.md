---
id: TASK-28007
title: Library import queue - batch re-analyze a run's analysis-skipped items
status: To Do
assignee: []
created_date: '2026-09-02 04:10'
labels:
  - library
  - media-ux
dependencies: []
references:
  - >-
    .impeccable/critique/2026-09-02T04-00-36Z__tldw-chatbook-ui-screens-library-screen-py.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When Analyze-after-import is on but the provider is unready, every imported row records analysis skipped with a reason (honest receipts) but there is no batch remediation - Retry re-imports rather than re-analyzing, so a 40-item run means 40 manual regenerations or a full re-import. Add a run-level Analyze-these-N action on completed runs that contain analysis-skipped rows. Builds on the viewer Generate seam (same provider resolution and persistence).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A completed import run with analysis-skipped items offers a single action that generates analyses for all of them
- [ ] #2 Per-item success and failure are reported in the same receipt style as import rows
- [ ] #3 Items that already have an analysis are not overwritten without an explicit choice
<!-- AC:END -->
