---
id: TASK-566
title: Cancel settings-rag-index-status workers on category nav
status: Done
assignee:
  - '@claude'
created_date: '2026-07-25 07:57'
updated_date: '2026-07-25 16:46'
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
Review follow-up (Important, caught live during PR review, same bug class as this task): `_apply_rag_test_category_result` calls the now-guarded `_apply_library_rag_index_status` but its OWN `self.app.notify("RAG check: ...")` was still unconditional -- a stale 't' test-category worker landing after nav-away would still toast the RAG check summary over an unrelated category. Guarded with the same `_active_category_id() is not LIBRARY_RAG: return` check, placed right before the notify call (after the state/preview-summary computation, matching the narrow-guard philosophy already used for `_decide_reindex_confirmation`'s modal-push branch). Regression test: test_stale_test_category_worker_landing_after_nav_away_skips_the_toast (RED confirmed pre-fix: notify fired with the stale category still showing; GREEN after). Full Tests/UI/test_settings_rag_profile_region.py: 117 passed.
<!-- SECTION:NOTES:END -->
