---
id: TASK-41773
title: "Seven Library tests are red on dev because their fakes lack an attribute the code under test started reading later"
status: To Do
created_date: 2026-11-18 16:14
labels:
- follow-up
- architecture-review-2026-10-25
- library
- tests
priority: low
references:
- Tests/UI/test_library_multiselect_media.py
- Tests/UI/test_review_set_walker.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Each of these tests builds a `SimpleNamespace` fake and calls a real method on it. The method later started reading an attribute the fake does not have. Five of the tests fail on the `AttributeError` before their assertions run; in the other two the code catches the `AttributeError` and fails closed, so they fail on their own assertions. The Phase 4 media lane PR's discovery runs found the first six red on 2026-11-18 at dev 47a05413cb (the multiselect three in 3 of 3 runs, the walker three in 1 of 1) and left them off the lane; they listed the seventh, the walker's toast test, as a non-retarget assertion red. The multiselect three were re-checked at dev 908e9840aa the same day, with the same error.

**Three multiselect tests.** These fail with `AttributeError: 'types.SimpleNamespace' object has no attribute '_library_media_editor_dirty'`:
- `Tests/UI/test_library_multiselect_media.py::test_row_press_normal_mode_opens_viewer` (`:191`);
- `::test_type_filter_change_exits_select_mode_and_notifies_discard` (`:1182`);
- `::test_single_delete_arm_supersedes_stale_receipt` (`:2248`).

The seam came from 05b078e1c2 (2026-10-20, TASK-34000.22). The three were already red at 08b3c9d337, the series' original base, according to the 3.2 review and the 3.3 report. The series retargeted them receiver-only. They are also listed in TASK-34000.100, which only open PR #3423 adds; at filing, #3423 is conflicting and not armed. If TASK-34000.100 lands first, this task's notes cite its commit for the multiselect three, and AC #1 then covers the walker four.

**Four review-set walker tests.** Three use `_walker_fake` (`Tests/UI/test_review_set_walker.py:35`) and one uses `_entry_fake` (`:378`). Both fakes predate 9feec051c1 (2026-09-19, TASK-32804.4), which made the review-set readers go through `_active_review_set_snapshot`, and neither fake has it.
- `::test_review_footer_at_last_follows_the_loaded_item_not_the_cursor` (`:1349`) raises `AttributeError: ... '_active_review_set_snapshot'`.
- `::test_active_review_progress_exposes_the_live_index_and_total` (`:1392`) raises the same error.
- `::test_review_set_active_reflects_the_service` (`:1296`) gets `False`, because `LibraryScreen._review_set_active` catches the same `AttributeError` and fails closed.
- `::test_create_drops_the_reviewing_toast_the_banner_already_carries` (`:1513`, added in 3a5ebe6257 on 2026-09-07) fails `assert not ['Reviewing 2 items.']`: `_active_review_set_banner` catches the missing `_active_review_set_snapshot` and returns `None`, so `_create_and_open_review_set` sends the toast the banner would have replaced. It was red 3 of 3 at dev 908e9840aa on 2026-11-18 (serial, through the scratch wrapper). A throwaway copy that binds the real `_active_review_set_snapshot` onto the same fake passed, so the test has been red since 9feec051c1 (2026-09-19).

**One more broken-fake retarget, owned elsewhere.** The lane PR's fix round 1 (2026-11-18) also found `Tests/UI/test_library_screen_reuse.py::test_on_screen_suspend_stops_every_timer_in_isolation` (a 3.3 retarget) red with `AttributeError: 'LibraryScreen' object has no attribute '_artifacts_navigation'`: its bypassed screen never had the attribute that 40a1b99576 (2026-09-20) made `on_screen_suspend` read. TASK-41713 (open #3621) already owns it, so it is not counted here.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the seven tests passes on dev, its fake given what the method now reads, with its assertions unchanged
- [ ] #2 Each is shown red before the fix and green after, by name
- [ ] #3 The seven run on the UI PR gate, or the reason they do not is recorded
<!-- AC:END -->
