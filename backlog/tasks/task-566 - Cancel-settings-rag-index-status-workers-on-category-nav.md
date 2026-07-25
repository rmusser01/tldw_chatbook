---
id: TASK-566
title: Cancel settings-rag-index-status workers on category nav
status: Done
assignee:
  - '@claude'
created_date: '2026-07-25 07:57'
updated_date: '2026-07-25 16:16'
labels:
  - settings
  - rag
  - tech-debt
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The exclusive worker group settings-rag-index-status (SP3-era) lets a stale off-thread status fetch land its callback after the user navigates away from Library/RAG — including, post-541, a re-index confirm modal appearing over an unrelated category. 541 reviews rated it pre-existing/non-blocking; wants a cancel-group-on-nav sweep in _select_category.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Leaving the Library/RAG category cancels in-flight index-status workers
- [x] #2 No modal or status write can land after nav-away (regression test)
- [x] #3 Re-entry still triggers a fresh fetch
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add regression tests: leaving Library/RAG cancels the settings-rag-index-status worker group; a stale callback landing after nav-away no-ops (no status write, no first-run refresh, no re-index modal); re-entry still triggers a fresh fetch.
2. In _select_category leave-LIBRARY_RAG branch, call self.workers.cancel_group(self, "settings-rag-index-status").
3. Guard _apply_library_rag_index_status to no-op when the active category is no longer LIBRARY_RAG.
4. Guard the modal-push branch of _decide_reindex_confirmation the same way (the non-modal dispatch-straight-through branch is unaffected).
5. Run the RAG profile region test file green.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two-layer fix, since the exclusive @work group cancellation is best-effort: cancelling a Textual Worker cancels its wrapping asyncio Task, but a thread body that has already started (loop.run_in_executor) keeps running to completion regardless and still calls back via call_from_thread -- so the callback-side guard is what actually prevents UI intrusion, not the cancel call alone.

1. _select_category's existing leave-LIBRARY_RAG branch now also calls self.workers.cancel_group(self, "settings-rag-index-status") (guarded by is_mounted, matching _refresh_library_rag_index_status's existing guard shape, since self.workers routes through self.app which a not-yet-mounted screen does not have).
2. _apply_library_rag_index_status (the status-write target of all three workers in the group: category-show, 't' test, and the Save-path reindex-confirm fetch) now no-ops when _active_category_id() is not LIBRARY_RAG -- skips the cache write, the Static text update, and the first-run panel refresh.
3. _decide_reindex_confirmation's modal-push branch (state == "built") is guarded the same way; the non-modal "nothing built to lose, dispatch save directly" branch is deliberately NOT guarded -- that save already reflects the user's deliberate Save click and has nothing intrusive to show. When the guard fires on the built branch, the save is silently dropped (not auto-confirmed, not shown out of context) since there is no one left on screen to confirm the destructive re-index warning.

Tests added to Tests/UI/test_settings_rag_profile_region.py (all sync-constructed, no full pilot mount needed -- `is_mounted` turned out to be a plain `_is_mounted` instance flag, and `workers`/`app` are both fakeable at the class level the same way the existing `fake_app` fixture already does): test_leaving_library_rag_cancels_the_index_status_worker_group (AC1), test_apply_library_rag_index_status_no_ops_after_navigating_away + test_stale_reindex_confirm_worker_landing_after_nav_away_skips_the_modal (AC2), test_reentering_library_rag_after_leaving_still_fetches_fresh_status (AC3, passed even pre-fix -- confirms the existing re-entry refetch was never broken by this change). First attempt at the AC1/AC3 tests used a full DestinationHarness pilot mount and was flaky (WaitForScreenTimeout / NoMatches on the third navigation, both from the same test) -- replaced with the sync-constructed style used throughout this file once the is_mounted/workers fakeability was confirmed, which is both fast (~1s for the 4 new tests) and deterministic.

Full Tests/UI/test_settings_rag_profile_region.py: 115 passed. Full Tests/UI/test_settings_configuration_hub.py (shared _select_category/_apply_library_rag_index_status/_decide_reindex_confirmation call sites) also re-run clean, confirming no regressions from the new guards.
<!-- SECTION:NOTES:END -->
