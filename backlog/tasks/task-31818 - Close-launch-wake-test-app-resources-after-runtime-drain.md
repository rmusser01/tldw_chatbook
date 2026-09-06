---
id: TASK-31818
title: Close launch-wake test app resources after runtime drain
status: Done
assignee:
  - '@codex'
created_date: '2026-09-06 05:58'
updated_date: '2026-09-06 06:17'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The complete launch-wake file passes behaviorally but retains roughly 357 descriptors; close its exact test app resources without weakening wake or reuse assertions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The complete launch-wake file retains no test SQLite descriptors after normal runtime/controller cleanup and exact database closure.
- [x] #2 Shared app-owner cleanup remains explicitly imported and closes no foreign/global resources; existing hydration and resource controls remain green.
- [x] #3 Full affected verification, static checks, independent review and evidence pass without production, GC or threshold changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Reason: test-only lifecycle cleanup using existing APIs. 1. Preserve the complete launch-wake 11-test descriptor baseline and native attribution. 2. Reuse TASK31816 exact app/runtime/database cleanup, extracting the local hydration fixture into an explicitly imported shared fixture only if required to avoid duplicate teardown. Bind capture to the importing test module and exact builder-created apps. 3. Preserve all wake/reuse/late-owner assertions and existing error/cancellation cleanup ordering. 4. Run complete launch-wake, hydration and resource-control files with native attribution, then every affected importer, lint/format/review and checkpoint.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Extracted the existing hydration app-owned runtime/auxiliary cleanup into an explicitly imported shared fixture bound to request.module._build_test_app. Launch-wake now uses it without changing wake/reuse behavior; current provider owner assertion is retained. Four RED isolation/order/error/cancellation controls now pass. Root verified the full prior importer set plus launch-wake and controls: 576 passed in 236.96s, three dependency warnings, zero Darwin F_GETPATH retained SQLite lines or FD-growth warning (/private/tmp/tldw-31818-all-importers-final.xml). New targeted launch/hydration/control set: 77 passed. Full lint, changed-region formatting, diff checks and independent root review pass. No production change, foreign/global cleanup, GC or threshold relaxation. ADR required: no; existing lifecycle APIs only.
<!-- SECTION:NOTES:END -->
