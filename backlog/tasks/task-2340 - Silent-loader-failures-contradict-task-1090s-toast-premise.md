---
id: TASK-2340
title: Silent loader failures contradict task-1090s toast premise
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-04'
updated_date: '2026-08-28 14:55'
labels:
  - watchlists
  - error-recovery
dependencies: []
priority: medium
---

## Description (the why)

UAT batch-2 review measured that `WatchlistsCollectionsScreen._load_source_rows_for_tree`
(:1535 at the time) logs its failure at debug and never notifies — while the
**Done** task-1090 lists that method among handlers "left at debug,
deliberately" on the premise that their failure surfaces as a toast. That
premise is now measured false for it. The new structural loader contract
(batch 2, PR #1348) covers async loaders only ("except handler guarding an
awaited read that logs at debug must notify in that handler"), so synchronous
loaders sit outside its enforcement by construction.

## Acceptance Criteria (the what)

- [ ] Failures in `_load_source_rows_for_tree` and `scoped_source_rows` emit a
      fixed, app-authored error toast with `markup=False`, while preserving the
      existing debug log and safe empty-list fallback.
- [ ] Synchronous-loader failure toasts contain no exception text, source
      names, URLs, watchlist names, or other dynamic content.
- [ ] The other synchronous entries from task-1090 are reconciled with current
      code: `_resolve_breadcrumb_labels` has no failure handler and
      `_refresh_feeds_region_for_scope` no longer exists.
- [ ] A structural test rejects new synchronous debug-swallow handlers unless
      they notify with `markup=False` in the same handler or are listed with a
      non-empty, current exemption reason.
- [ ] Mounted regressions prove both affected loaders retain their empty
      fallback and emit exactly one markup-disabled error toast without
      leaking sentinel exception content.
- [ ] Focused Watchlists tests, modified-file Ruff lint and format checks, and
      `git diff --check` pass.

## References

- Design: `Docs/superpowers/specs/2026-08-28-watchlists-synchronous-loader-failure-toasts-design.md`
