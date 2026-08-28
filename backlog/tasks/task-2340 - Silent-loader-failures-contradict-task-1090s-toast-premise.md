---
id: TASK-2340
title: Silent loader failures contradict task-1090s toast premise
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-04'
updated_date: '2026-08-28 15:45'
labels:
  - watchlists
  - error-recovery
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UAT batch-2 review measured that `WatchlistsCollectionsScreen._load_source_rows_for_tree`
(:1535 at the time) logs its failure at debug and never notifies — while the
**Done** task-1090 lists that method among handlers "left at debug,
deliberately" on the premise that their failure surfaces as a toast. That
premise is now measured false for it. The new structural loader contract
(batch 2, PR #1348) covers async loaders only ("except handler guarding an
awaited read that logs at debug must notify in that handler"), so synchronous
loaders sit outside its enforcement by construction.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- SECTION:ACCEPTANCE_CRITERIA:BEGIN -->
- [x] #1 Failures in `_load_source_rows_for_tree` and `scoped_source_rows` emit a
      fixed, app-authored error toast with `markup=False`, while preserving the
      existing debug log and safe empty-list fallback.
- [x] #2 Synchronous-loader failure toasts contain no exception text, source
      names, URLs, watchlist names, or other dynamic content.
- [x] #3 The other synchronous entries from task-1090 are reconciled with current
      code: `_resolve_breadcrumb_labels` has no failure handler and
      `_refresh_feeds_region_for_scope` no longer exists.
- [x] #4 A structural test rejects new synchronous debug-swallow handlers unless
      they notify with literal `severity="error"` and `markup=False` in that
      exact handler or the handler-level key is listed with a non-empty,
      current exemption reason; discovered and exempt inventories must match
      exactly so stale and missing exemptions both fail.
- [x] #5 Mounted regressions prove both affected loaders retain their empty
      fallback and emit exactly one markup-disabled error toast without
      leaking sentinel exception content.
- [ ] #6 Focused Watchlists tests, modified-file Ruff lint and format checks, and
      `git diff --check` pass.
<!-- SECTION:ACCEPTANCE_CRITERIA:END -->
<!-- AC:END -->

## References

- Design: `Docs/superpowers/specs/2026-08-28-watchlists-synchronous-loader-failure-toasts-design.md`
- Plan: `Docs/superpowers/plans/2026-08-28-watchlists-synchronous-loader-failure-toasts.md`

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add mounted red regressions for both synchronous source loaders and a handler-level AST contract with exact exemptions.
2. Add the two fixed severity=error, markup=False notifications through _notify_watchlists while preserving debug logs and empty fallbacks.
3. Run the focused failure-policy and source-scope modules, modified-file Ruff lint/format, and git diff --check.
4. Complete TASK-2340 acceptance criteria and implementation notes only after the final focused gate passes.

ADR required: no
ADR path: N/A
Reason: bounded error reporting within the existing Watchlists screen; no schema, ownership, service, runtime, dependency, or long-lived UX boundary changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added fixed markup-disabled error toasts to both live synchronous source loaders while preserving debug diagnostics and empty fallbacks. Added mounted regressions plus an exact handler-level AST contract with documented lifecycle/preference-write exemptions. ADR required: no; ADR path: N/A; this is a bounded screen error-reporting repair. Focused verification is pending its final combined completion gate.
<!-- SECTION:NOTES:END -->
