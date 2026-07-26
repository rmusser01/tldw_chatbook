---
id: TASK-837
title: Cache workspace folder-root lookups off the file-tool hot path
status: To Do
assignee: []
created_date: '2026-07-26 16:20'
labels:
  - tools
  - performance
  - workspaces
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
allowed_file_roots opens a fresh WorkspaceDB connection on every file-tool invocation (read/list/write), including the zero-binding case. Correctness is unaffected (fail-safe design), but tight agent tool loops pay a per-call SQLite open. Add a short-TTL cache or per-run memoization keyed by workspace id, preserving the call-time existence re-check for the folders themselves (existence must stay live; only the binding LIST may be cached).

Source: workspace folder-roots train final review (spec 2026-07-26-settings-workspaces-category-design.md), deferred-minor triage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 File-tool invocations no longer open a new WorkspaceDB connection per call in steady state
- [ ] #2 Folder existence is still verified live at call time
- [ ] #3 Binding add/remove/toggle invalidates or bypasses the cache within one run
<!-- AC:END -->
