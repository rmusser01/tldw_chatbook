---
id: TASK-36004
title: Own keyword index jobs refresh recovery and revocation
status: To Do
assignee:
  - '@codex'
created_date: '2026-10-21 14:31'
labels: []
dependencies:
  - TASK-36002
  - TASK-36003
documentation:
  - Docs/superpowers/plans/2026-10-20-workspace-project-keyword-index.md
  - backlog/decisions/240-workspace-project-context-and-folder-bound-index.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make project indexing an application-owned optional lifecycle with safe incremental refresh and cancellation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Enable, Refresh, Pause, Resume, Disable and Clear compare expected scope/job state before effects; delayed actions cannot resurrect revoked enablement or work.
- [ ] #2 One durable writer publishes only compatible batches under cross-process authority fences; detach/exclusion/retarget/clear races cannot publish stale data.
- [ ] #3 Incremental changes preserve unreadable or unfinished subtrees; restart stays Interrupted until explicit Resume and quotas report Partial coverage.
- [ ] #4 Revocation precedes cleanup; physical process-tree and SQLite owner retirement are observed even without a Console view or on cleanup failure.
<!-- AC:END -->
