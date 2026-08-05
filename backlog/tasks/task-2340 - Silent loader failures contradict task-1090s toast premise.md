---
id: TASK-2340
title: Silent loader failures contradict task-1090s toast premise
status: To Do
assignee: []
created_date: '2026-08-04'
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

- [ ] `_load_source_rows_for_tree`'s failure surfaces to the user (toast,
      markup=False, type-only) or its silence is re-justified in writing
      against task-1090's own rule.
- [ ] The other synchronous entries in task-1090's 15-handler list are
      re-checked against the same premise: `scoped_source_rows`,
      `_resolve_breadcrumb_labels`, `_refresh_feeds_region_for_scope` —
      each either notifies, is exempted with a current justification, or is
      fixed.
- [ ] The structural loader contract (or a sibling) enforces whatever rule
      emerges for synchronous loaders, so the next silent loader cannot
      land unnoticed.
