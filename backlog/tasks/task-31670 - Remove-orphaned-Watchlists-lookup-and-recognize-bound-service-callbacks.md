---
id: TASK-31670
title: Remove orphaned Watchlists lookup and recognize bound service callbacks
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:08'
updated_date: '2026-09-05 18:26'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the production caller audit after atomic collection creation retired the old lookup and service wiring began passing bound callbacks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The unused case-insensitive lookup is removed while atomic collection reuse behavior stays covered.
- [x] #2 The caller audit recognizes exact WatchlistBundleService bound callbacks and rejects unrelated objects with colliding method names.
- [x] #3 Affected Watchlists collection and subscription tests and scoped static checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Extract the existing AST scanner into a test-local helper without changing its rules; add positive bound-callback and unrelated-object negative regressions and reproduce the audit failure. 2. Recognize service-bound method references supplied as call arguments using the existing service provenance predicate, without generic attribute/text matching. 3. Delete get_watchlist_by_name_ci, whose only caller was intentionally replaced by atomic create_with_sources in commit1bba027f55. 4. Run the complete Tests/Watchlists/test_watchlists_collections_screen.py and Tests/Subscriptions/test_watchlist_bundle_service.py, test_watchlist_opml_service.py, test_local_watchlists_service.py plus scoped lint/format checks; diagnose any separately observed toolbar failure before expanding scope. 5. Record evidence, request parent review, mark done, scoped commit. ADR required: no. ADR path: N/A. Reason: routine dead-code removal and test audit repair, preserving existing atomic service contracts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Removed the genuinely orphaned get_watchlist_by_name_ci method; commit1bba027f55 had replaced its sole caller with atomic create_with_sources(return_existing), whose behavior remains unchanged. The test-local AST scanner now counts exact service-bound callbacks in positional/keyword call arguments. Two positive cases failed before repair; two unrelated-object negatives passed. The full caller audit originally identified exactly the orphan and update_sources; now passes. Final gate: pytest Tests/Watchlists/test_watchlists_collections_screen.py Tests/Subscriptions/test_watchlist_bundle_service.py Tests/Subscriptions/test_watchlist_opml_service.py Tests/Subscriptions/test_local_watchlists_service.py -o addopts= -n2;225passed106.09s, report /private/tmp/tldw-31670-31680-watchlists-green.xml. A separately tracked fixture startup race was diagnosed and repaired without altering runtime reconciliation. Scoped Ruff lint, changed-range formatting, diffcheck and parent independent review passed. ADR required:no; routine dead-code/audit repair, atomic contracts unchanged.
<!-- SECTION:NOTES:END -->
