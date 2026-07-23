---
id: TASK-399.3
title: A2 Project search and reconcile file notes
status: To Do
assignee: []
created_date: '2026-07-23 14:23'
labels:
  - notes
  - search
  - filesystem
dependencies:
  - TASK-399.2
documentation:
  - >-
    Docs/superpowers/specs/2026-07-22-file-backed-notes-authority-recovery-design.md
  - backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md
parent_task_id: TASK-399
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Provide a scalable repairable read model of the linked root that follows external filesystem changes near real time while disk remains authoritative.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Initial inventory exposes deterministic paged path metadata before body indexing and never reads all bodies before first tree availability.
- [ ] #2 Strict UTF-8/optional-BOM and configured-size eligibility, filename/path authority, opaque-frontmatter exclusion from editable body/FTS, and present/missing/tombstoned/offline transitions preserve identity while suppressing every stale body/snippet.
- [ ] #3 File FTS is maintained by an idempotent retryable worker outside projection commits; an FTS failure cannot block browsing or later file commands.
- [ ] #4 Filename/path results remain available while body indexing is incomplete or unavailable.
- [ ] #5 Watchdog events and the visible bounded polling fallback feed the same hash-based reconciliation path in packaged builds.
- [ ] #6 A settled external edit reaches projection and eligible FTS visibility within 2 seconds p95 on the fixed healthy runner.
- [ ] #7 Warm first-page search over the 5,000-file fixture completes within 200 ms p95 across at least 30 samples, with stale query generations unable to publish.
- [ ] #8 Watcher storms, overflow, Git bulk changes, unreadable/offline transitions, queue bounds, and polling CPU gates pass.
<!-- AC:END -->
