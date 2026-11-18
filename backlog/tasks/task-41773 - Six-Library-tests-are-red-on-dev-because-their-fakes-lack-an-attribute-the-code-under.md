---
id: TASK-41773
title: "Six Library tests are red on dev because their fakes lack an attribute the code under test started reading later"
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
Each of these tests builds a `SimpleNamespace` fake and calls a real method on it. The method later started reading an attribute the fake does not have, so the tests fail before their assertions run. The Phase 4 media lane PR's discovery runs found all six red (2026-11-18, dev 47a05413cb, 3 of 3) and left them off the lane. The multiselect three were re-checked at dev 908e9840aa the same day, with the same error.

**Three multiselect tests.** These fail with `AttributeError: 'types.SimpleNamespace' object has no attribute '_library_media_editor_dirty'`:
- `Tests/UI/test_library_multiselect_media.py::test_row_press_normal_mode_opens_viewer` (`:191`);
- `::test_type_filter_change_exits_select_mode_and_notifies_discard` (`:1182`);
- `::test_single_delete_arm_supersedes_stale_receipt` (`:2248`).

The seam came from 05b078e1c2 (2026-10-20, TASK-34000.22). The three were already red at the series' base, 08b3c9d337, according to the 3.2 review and the 3.3 report. The series retargeted them receiver-only. They are also listed in TASK-34000.100, which open PR #3423 adds; that task is not on dev at filing. Whichever change lands first fixes them.

**Three review-set walker tests.** These use `_walker_fake` (`Tests/UI/test_review_set_walker.py:35`), which predates 9feec051c1 (2026-09-19, TASK-32804.4). That commit added `_active_review_set_snapshot`.
- `::test_review_footer_at_last_follows_the_loaded_item_not_the_cursor` (`:1349`) raises `AttributeError: ... '_active_review_set_snapshot'`.
- `::test_active_review_progress_exposes_the_live_index_and_total` (`:1392`) raises the same error.
- `::test_review_set_active_reflects_the_service` (`:1296`) gets `False`, because `LibraryScreen._review_set_active` catches the same `AttributeError` and fails closed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the six tests passes on dev, its fake given what the method now reads, with its assertions unchanged
- [ ] #2 Each is shown red before the fix and green after, by name
- [ ] #3 The six run on the UI PR gate, or the reason they do not is recorded
<!-- AC:END -->
