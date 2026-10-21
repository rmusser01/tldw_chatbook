---
id: TASK-36004
title: Own keyword index jobs refresh recovery and revocation
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-21 14:31'
updated_date: '2026-10-21 16:27'
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implement approved K3 single app-owned index service/runtime with exact-scope/job CAS, short registry-before-store publication fences, incremental durable frontier and whole-binding quota accounting. Capture verified admitted data/profile/restore authority outside fences under retained execution lease; ordinary preference saves remain compatible while switch/restore/rollback invalidate. Add small native lifetime helper with capability inherited at process creation; exact old claim replacement requires fresh exclusive reacquisition and actual writer-thread DB closure/scanner retirement. Preserve base-interpreter isolated bootstrap and no background restart Resume. Revoke before cancellation/cleanup; keep stable control epoch and lease objects through Clear, including obsolete namespace accounting. TDD real stores and two-process late-publication/action/crash races; required K1/K2 covering checks, no-Console app shutdown/profile qualification, scoped lint/diff and independent review. ADR required: yes, existing accepted ADR240 implements these exact retirement/privacy boundaries with ADR101/102/126/174/228; native implementation detail documented there before code, no new ADR. No Library mutations, semantic work, new dependencies or full suite.
<!-- SECTION:PLAN:END -->
