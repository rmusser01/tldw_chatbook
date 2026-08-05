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
Own confirmed read-only root activation and provide a scalable, repairable read model that follows external filesystem changes near real time while disk remains authoritative.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After a fresh successful A1 preview and explicit confirmation, activation acquires coordinator election, durably persists one root in a crash-resumable activating state, and leaves source files unchanged.
- [ ] #2 Activation captures watcher events before its bounded initial inventory, drains those events through the same reconciliation path, rescans after overflow or root-identity change, and atomically publishes only a consistent root/projection generation.
- [ ] #3 Initial inventory exposes deterministic paged path metadata before body indexing and never reads all bodies before first tree availability.
- [ ] #4 Strict UTF-8/optional-BOM eligibility uses the fixed 8,000,000-raw-byte pre-read guard and 2,000,000-character projection/FTS ceiling without treating the separate 200,000-character interactive ceiling as an indexing limit; filename/path authority, opaque-frontmatter exclusion from editable body/FTS, and present/missing/tombstoned/offline transitions preserve identity while suppressing every stale body/snippet.
- [ ] #5 File FTS is maintained by an idempotent retryable worker outside projection commits; an FTS failure cannot block browsing, reconciliation, or later file commands.
- [ ] #6 Every body-projection and FTS batch checks free space on the actual store volume and preserves the 256 MiB recovery floor plus its next bounded batch reservation; when admission fails, metadata/path publication continues while new body/FTS writes pause under a visible storage-low status.
- [ ] #7 Filename/path results remain available while body indexing is incomplete, paused for capacity, or unavailable, and eligible indexing resumes only after the floor and one batch reservation are healthy.
- [ ] #8 Watchdog is a declared core dependency included in wheel and packaged-app artifacts; installed-build smoke tests prove that Watchdog events and the visible bounded polling fallback feed the same hash-based reconciliation path.
- [ ] #9 A settled external edit reaches projection and eligible FTS visibility within 2 seconds p95 on the fixed healthy runner.
- [ ] #10 Warm first-page search over the 5,000-file fixture completes within 200 ms p95 across at least 30 samples, with stale query generations unable to publish.
- [ ] #11 Watcher storms, overflow, Git bulk changes, unreadable/offline transitions, queue bounds, and polling CPU gates pass.
- [ ] #12 The activation entry point, persistent monitoring, and Files-source publication remain behind the default-off A release gate until A4 validates A0-A4 together.
<!-- AC:END -->
