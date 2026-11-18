---
id: TASK-41766
title: "Five Media tests query a widget right after their wait returns, while a recompose has it unmounted"
status: To Do
created_date: 2026-11-18 16:14
labels:
- follow-up
- architecture-review-2026-10-25
- library
- tests
- flaky
priority: low
references:
- Tests/UI/test_library_media_reader_scroller_resolution.py
- Tests/UI/test_library_media_reader_flow.py
- Tests/UI/test_library_per_click_recompose_t21116.py
- Tests/UI/test_library_review_round_t21116.py
- Tests/UI/test_library_shell.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Each of these tests waits for state or for a selector, then calls `query_one` on a widget that a recompose has not mounted yet or has just replaced, so under load it raises `NoMatches`.

**The scroller test.**
- `Tests/UI/test_library_media_reader_scroller_resolution.py::_open_raw_view_ready` (`:110`) queries `#library-media-viewer-content` straight after `loaded_id` settles.
- `::test_restore_progress_resolves_the_real_scroller_and_applies_the_saved_offset` (`:305`) failed there in TASK-36952.3.6's marker run. In that slice's review it failed 2 of 9 runs on the head and 0 of 9 on the base. The last 6 alternating rounds passed on both sides, and the review found no mechanism in the move.
- The base showed the same failure in the sibling capture test.

**The reader-flow test.**
- `Tests/UI/test_library_media_reader_flow.py::test_detail_failure_keeps_items_usable_and_reader_retryable` (`:759`) queries `#library-media-reader-retry` right after the retryable state settles.
- It failed 5 of 15 runs on the head and 2 of 15 on the base, and 1 of 8 on each side over the last 8 alternating rounds, with the same `NoMatches` on both sides. This is from the 3.6 review (2026-11-17, chain over dev 5e3c8b2bd4).

**Status at dev 908e9840aa.** Both waits are unchanged. TASK-41127 fixed this shape for `#library-media-reader-find`: `_load_row_0` and `_open_first_reader_row` now wait for the Reader chrome. These two remain.

**Three t21116 tests the lane PR dropped** (fix round 1, 2026-11-18, dev 908e9840aa, load 33-97). Each waits with `_wait_for_selector` (`Tests/UI/test_library_shell.py:3930-3949`) and then queries a widget in the same tick. After a match the helper pauses once and re-queries; when the re-query finds nothing, mid-recompose, it returns the stale first match instead of waiting again, so the caller's next `query_one` can raise `NoMatches`:
- `Tests/UI/test_library_per_click_recompose_t21116.py::test_media_row_open_latency_probe` (`:384`) passed 2 of 3: `NoMatches '#library-media-row-0'` at load 78, where its loop presses row 0 again right after waiting for it;
- `::test_media_viewer_substate_escape_is_viewer_scoped` (`:145`) passed 6 of 7: `NoMatches '#library-media-reader-more'` inside `test_library_shell.py::_disclose_media_viewer_more_actions` (`:3960`), which waits for `#library-media-viewer` and then queries the More button;
- `Tests/UI/test_library_review_round_t21116.py::test_late_media_detail_arrival_cannot_clobber_the_trash_view` (`:71`) passed 6 of 7: `NoMatches '#library-media-trash-open'` right after waiting for that selector.

The reader-flow tests that wait one `pilot.pause()` after a press belong to TASK-41505.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each test waits, with a bounded deadline, for the widget it queries
- [ ] #2 Each passes 10 of 10 at -n 4 under load, under the private-profile marker if it stops at RecoveryRequired locally, with the load recorded
- [ ] #3 Their assertions are unchanged
- [ ] #4 `_wait_for_selector` does not return a node that its own settle re-query no longer finds
<!-- AC:END -->
