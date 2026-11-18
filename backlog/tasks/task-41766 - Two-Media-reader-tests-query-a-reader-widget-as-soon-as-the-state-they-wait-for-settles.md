---
id: TASK-41766
title: "Two Media reader tests query a reader widget as soon as the state they wait for settles, before it mounts"
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
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Both tests wait for state, then call `query_one` on a widget that the next recompose mounts, so under load they raise `NoMatches`.

**The scroller test.**
- `Tests/UI/test_library_media_reader_scroller_resolution.py::_open_raw_view_ready` (`:110`) queries `#library-media-viewer-content` straight after `loaded_id` settles.
- `::test_restore_progress_resolves_the_real_scroller_and_applies_the_saved_offset` (`:305`) failed there in TASK-36952.3.6's marker run. In that slice's review it failed 2 of 9 runs on the head and 0 of 9 on the base. The last 6 alternating rounds passed on both sides, and the review found no mechanism in the move.
- The base showed the same failure in the sibling capture test.

**The reader-flow test.**
- `Tests/UI/test_library_media_reader_flow.py::test_detail_failure_keeps_items_usable_and_reader_retryable` (`:759`) queries `#library-media-reader-retry` right after the retryable state settles.
- It failed 5 of 15 runs on the head and 2 of 15 on the base, and 1 of 8 on each side over the last 8 alternating rounds, with the same `NoMatches` on both sides. This is from the 3.6 review (2026-11-17, chain over dev 5e3c8b2bd4).

**Status at dev 908e9840aa.** Both waits are unchanged. TASK-41127 fixed this shape for `#library-media-reader-find`: `_load_row_0` and `_open_first_reader_row` now wait for the Reader chrome. These two remain. The reader-flow tests that wait one `pilot.pause()` after a press belong to TASK-41505.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each test waits, with a bounded deadline, for the widget it queries
- [ ] #2 Each passes 10 of 10 at -n 4 under load, under the private-profile marker if it stops at RecoveryRequired locally, with the load recorded
- [ ] #3 Their assertions are unchanged
<!-- AC:END -->
