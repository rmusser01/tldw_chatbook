---
id: TASK-41769
title: "The phase-C media region census is red on dev: four media @on rows are unclassified"
status: To Do
created_date: 2026-11-18 16:14
labels:
- follow-up
- architecture-review-2026-10-25
- library
- tests
priority: low
references:
- Tests/UI/test_library_phase_c_region_ownership.py
- tldw_chatbook/UI/Screens/library_screen.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_library_phase_c_region_ownership.py::test_the_census_covers_every_media_row_on_the_screen_exactly_once` fails serially at dev 908e9840aa (2026-11-18). The census has three sets: migrated, deferred canvas-origin and permanent screen. Four media `@on` rows on `LibraryScreen` are in none of them:
- `handle_library_media_take_note`, added in 47f5bbea32 (2026-10-09);
- `handle_library_media_link_undo`, added in e4680be1a9 (2026-10-19);
- `handle_library_media_edit_field_changed` and `handle_library_media_analysis_text_changed`, added in 05b078e1c2 (2026-10-20).

The census file was last edited on 2026-09-09, and no PR lane runs it. The TASK-36952.3.4 review found it red at that slice's base, 14bd55bcf3, and at its head. The four rows predate the series (2026-10-09 to 2026-10-20), so the test was red before the series too. While it is red, the census cannot catch the next unclassified row, and catching one is its only job.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the four rows is classified by where its message is posted, with the reason recorded beside it
- [ ] #2 The census test passes on dev
- [ ] #3 The census test runs on a PR lane, or the reason it does not is recorded
<!-- AC:END -->
