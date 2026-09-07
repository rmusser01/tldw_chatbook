---
id: TASK-527
title: Use file-backed SQLite stores in the shared UI app harness
status: Done
assignee: []
created_date: '2026-07-24 19:17'
updated_date: '2026-07-24 19:21'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep full-app UI tests on isolated temporary databases whose schema survives the services' multiple SQLite connections, so schedules, research, and writing screens exercise their real empty states instead of failing on missing tables.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The shared UI app builder uses per-app temporary files for subscriptions, research, and writing databases
- [x] #2 Operational reads from all three stores see their initialized schemas
- [x] #3 The Schedules workbench loads an empty queue without a missing-table error
- [x] #4 The full screen-navigation and Schedules workbench modules pass
- [x] #5 The merge-base failure and no-ADR decision are documented
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the schedules failure and trace the missing table through the shared app builder and SQLite connection lifecycle.
2. Add a harness regression that performs empty reads through subscriptions, research, and writing services.
3. Replace only unsupported :memory: path patches with files inside the builder's existing per-app temporary directory.
4. Run the regression, screen-navigation and Schedules workbench modules, Ruff, format, diff checks, and independent review.
5. Document merge-base evidence and the no-ADR decision before completion.

ADR required: no
ADR path: N/A
Reason: This corrects test-harness database paths without changing production schemas, ownership, or runtime storage policy.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Changed the shared full-app UI builder to place subscriptions, research, and writing SQLite stores under its already-unique temporary directory instead of using :memory:. Added an operational regression that reads through WatchlistProjection, LocalResearchService, and LocalWritingService, proving each service sees the schema created on its separate initialization connection.

Before the fix, the schedules loader failed with sqlite3.OperationalError: no such table: subscriptions; the new regression reproduced the same failure immediately. The merge base is masked earlier by its production-path readonly database defect, but inspection confirms the same unsupported :memory: patches are present there. Verification: full screen-navigation plus Schedules workbench modules pass (93 passed); Ruff, format, and diff checks pass. Independent review approved with no findings.

ADR required: no. All paths remain isolated test-only temporary files; production schemas, ownership, and runtime storage policy are unchanged.

Modified: Tests/UI/test_screen_navigation.py and this task file.
<!-- SECTION:NOTES:END -->
