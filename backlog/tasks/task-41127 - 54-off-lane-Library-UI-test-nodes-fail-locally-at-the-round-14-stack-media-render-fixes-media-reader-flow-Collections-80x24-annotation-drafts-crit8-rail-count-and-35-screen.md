---
id: TASK-41127
title: "54 off-lane Library UI test nodes fail locally at the round-14 stack: media render fixes, media reader flow, Collections 80x24 annotation drafts, crit8 rail count and 35 screen-navigation nodes"
status: To Do
created_date: 2026-11-15 02:20
labels:
- follow-up
- library
- tests
- dev-red
priority: medium
references:
- Tests/UI/test_library_media_render_fixes.py
- Tests/UI/test_library_media_reader_flow.py
- Tests/UI/test_library_collection_reader_journeys.py
- Tests/UI/test_library_crit8_collections_row.py
- Tests/UI/test_screen_navigation.py
- Tests/UI/test_library_prompt_mixins.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** None of these nodes is in the UI census or any workflow, so they can stay red indefinitely; the round-14 Library streams recorded several as red on their base. Under the plain command they hit the local RecoveryRequired trap, so their own failures are only visible with a profile marker.

**Evidence.** The five files at `-n 4` through run.sh with a scratch plugin stamping `bootstrap_profile` on unmarked items (no repository edit; load 44-55): 54 failed, 334 passed. No failing node is a census entry; the 101 census nodes of `test_screen_navigation.py` passed.
- `test_library_media_render_fixes.py` x9: `test_escape_labels_are_one_of_four` (`:1284`, `('armed delete', 'cancel')`), the analyze/generate reason nodes x8 ("OpenAI is not ready (Missing API key)" where "No analysis … Providers & Models." is expected, `:1956`, `:1988`, `:2036`).
- `test_library_media_reader_flow.py` x7: page-three anchor, select-mode settlement, `t` delete confirm, edit-metadata routing (`NoMatches '#library-media-reader-find'` / `'#library-media-reader-more'`), and `test_any_non_markdown_item_with_content_says_why_rendered_is_absent[plaintext|video]` (`:3146`, empty notice). TASK-23145 owns an older scroller-seam failure in this file; these are different.
- `test_library_collection_reader_journeys.py::test_annotation_drafts_survive_disclosures_modes_and_capture_return[textual-dark-size1|textual-light-size1]` (80x24): "Expected readable #library-collections-freeform-note-save".
- `test_library_crit8_collections_row.py::test_rail_count_never_falls_back_to_the_unfiltered_total_after_a_visit`.
- `test_screen_navigation.py` x37: 2 are TASK-23149's fake-row nodes; the other 35 include `WorkspaceProbe` lacking `cancel_path_task` / `cancel_structural_wait`, lambdas taking one argument where `library_screen.py:11078` passes two, `LibraryScreen` lacking `_sync_collections_panel`, and many route tests landing on `ChatScreen`/`Screen` instead of `LibraryScreen`, which may be an artifact of the shared bootstrap profile rather than the code.
- Also off-lane and intermittent: `test_library_prompt_mixins.py::test_literal_history_restore_keeps_kind_definition_and_guards` (red in 1-3 of 3 params on base 26b2e677dd, TASK-40293's notes).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each node is run under its own profile (a private-profile marker where it reads config) and classified as test drift, product defect, or shared-profile artifact, with the evidence
- [ ] #2 Drift is re-pinned with the commit that moved the behaviour named; a product defect gets a test red before its fix
- [ ] #3 Every node that then passes runs on a PR lane (UI census tail entries with measured times)
<!-- AC:END -->
