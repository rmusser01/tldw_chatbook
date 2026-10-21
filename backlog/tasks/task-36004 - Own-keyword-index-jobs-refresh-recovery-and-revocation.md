---
id: TASK-36004
title: Own keyword index jobs refresh recovery and revocation
status: Done
assignee:
  - '@codex'
created_date: '2026-10-21 14:31'
updated_date: '2026-10-21 18:08'
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
- [x] #1 Enable, Refresh, Pause, Resume, Disable and Clear compare expected scope/job state before effects; delayed actions cannot resurrect revoked enablement or work.
- [x] #2 One durable writer publishes only compatible batches under cross-process authority fences; detach/exclusion/retarget/clear races cannot publish stale data.
- [x] #3 Incremental changes preserve unreadable or unfinished subtrees; restart stays Interrupted until explicit Resume and quotas report Partial coverage.
- [x] #4 Revocation precedes cleanup; physical process-tree and SQLite owner retirement are observed even without a Console view or on cleanup failure.
- [x] #5 Explicit Clear reclaims validated v2 derived cache bytes across old root and authority generations so a normally admitted quota can be reused; epoch/claim history and stable native lease objects survive, while unproven legacy or insufficient-headroom cleanup remains Off with a truthful retryable outcome.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implement approved K3 single app-owned index service/runtime with exact-scope/job CAS, short registry-before-store publication fences, incremental durable frontier and whole-binding quota accounting. Capture verified admitted data/profile/restore authority outside fences under retained execution lease; ordinary preference saves remain compatible while switch/restore/rollback invalidate. Add small native lifetime helper with capability inherited at process creation; exact old claim replacement requires fresh exclusive reacquisition and actual writer-thread DB closure/scanner retirement. Preserve base-interpreter isolated bootstrap and no background restart Resume. Revoke before cancellation/cleanup; keep stable control epoch and lease objects through Clear, including obsolete namespace accounting. TDD real stores and two-process late-publication/action/crash races; required K1/K2 covering checks, no-Console app shutdown/profile qualification, scoped lint/diff and independent review. ADR required: yes, existing accepted ADR240 implements these exact retirement/privacy boundaries with ADR101/102/126/174/228; native implementation detail documented there before code, no new ADR. No Library mutations, semantic work, new dependencies or full suite.

AC5 refinement from source audit: explicit binding-wide Clear inventories validated v2 namespaces, acquires stable native lifetimes in deterministic order outside fences, preserves control epochs/claim tombstones/lease names, and reclaims derived payload with bounded incremental vacuum configured for fresh caches. Process smaller namespaces first; enforce journal/temp-inclusive whole-binding headroom. Qualify admitted near-capacity old-root/prior-authority cleanup and reusable quota, DB/control identity, concurrent Enable/stale Clear and cleanup retry; unproven legacy or insufficient headroom remains Off truthfully. No existing disposal callable was found; ADR240 records this choice before cleanup code.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented app-owned folder-bound lifecycle and runtime admission, schema-v2 incremental membership, inherited native retirement capability, exact-scope/job management and binding-wide in-place Clear with conservative physical quota headroom. Registry-before-store publication and new-unit admission reject remote revocation; delayed cancellation targets only captured runs. Commits 7f5255e270 and485c23cf7e; independent spec/quality review and scoped fix review approved. Targeted amended service/store54passed plus test-helper3passed; native/owner/recovery30passed earlier on unchanged paths. Scoped lint/format/diff pass; inherited warning,16baseline E721 and Windows-only/POSIX-unexecuted limits retained. See Docs/superpowers/qa/2026-10-20-workspace-project-context/keyword/task-3-report.md, task-3-review.md, task-3-re-review-1.md and k3-evidence.json. Existing ADR240 (backlog/decisions/240-workspace-project-context-and-folder-bound-index.md), ADR101/102/126/174/228 apply; refinements recorded before code. No Library/tool/UI/semantic changes. Known minor historical run retention goes to whole-branch review.
<!-- SECTION:NOTES:END -->
