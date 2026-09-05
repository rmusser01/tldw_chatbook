---
id: TASK-31249
title: Library UI test debt on dev - six pre-existing failures nobody owns
status: To Do
assignee: []
created_date: '2026-09-04 04:59'
labels:
  - library
  - tests
  - tech-debt
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Eight Library UI tests fail identically on clean dev (verified at 59d987015d and e4652f9d37, each file run in its own process) and no open task owns them. PR #2367's body noted them as pre-existing and the wave-3 landing hit them again; every media PR touching these files now has to re-verify them by hand to tell its own breakage from the baseline, which is how a real break (fixed in PR #2369) nearly hid among them. Suspected origins, unconfirmed: #2364 (Personas demand-mount, task-31215) for the `#library-media-edit` group; an unbound `MediaReadingScopeService` fake (`active_authority`) for the deep-link test.

Failures (`pytest --tb=line`, Tests/UI):
- test_library_per_click_recompose_t21116.py::test_media_viewer_substate_escape_is_viewer_scoped -- `#library-media-edit never mounted within 30.0s` (test_library_shell.py:3686 helper)
- test_library_per_click_recompose_t21116.py::test_open_item_by_id_media_is_canvas_scoped -- `assert [LibraryScreen()] == []` (a whole-screen recompose where a canvas-scoped one is pinned)
- test_library_per_click_recompose_t21116.py::test_export_open_from_media_is_canvas_scoped -- LibraryRail identity changed (rail was recomposed)
- test_library_review_round_t21116.py::test_viewer_substate_escape_refreshes_the_footer_shortcut_set -- `#library-media-edit never mounted within 30.0s`
- test_library_shell.py::test_library_starter_deep_link_opens_hidden_collection_or_note_route -- worker `AttributeError: 'SimpleNamespace' object has no attribute 'active_authority'` in library_collections_capture_controller.adopt_active_authority
- test_library_choice_strips.py::test_media_type_strip_works_in_both_layouts -- `compact class never reached True at (100, 30)`
- test_library_shell.py::test_library_shell_media_viewer_inplace_search_preserves_identity_focus_and_parse_count -- after a submitted search and a Next press in Rendered mode the `#library-media-viewer-content-markdown` widget is a new instance (in-place match navigation rebuilt the body); plain `LibraryHarness`
- test_library_shell.py::test_library_shell_media_viewer_inplace_search_chrome_paints_above_content -- the content body lays out at `Region(x=83, y=53, height=3)` below the 48-row viewport after a rendered-mode search, so the document heading never paints; plain `LibraryHarness` (no production stylesheet). Its retired 18-row-cap assertion was updated in wave-4 PR A; the layout failure remains
<!-- SECTION:DESCRIPTION:END -->

Wave 4 (2026-09-04) additions, all confirmed pre-existing on clean dev/base before each PR:
- test_library_ingest_canvas.py::test_progress_detail_paints_below_row…[size0|size1] -- geometry failures on D base 573a5854cd and every later head (`_QueuePanelHost` mounts only the queue panel)
- test_library_ingest_retry_last (registry-ticks flake) -- reproduces byte-identically on the base
- test_library_shell.py::test_library_media_durable_mutation_gates_and_refreshes_applied_scope[False] and ::test_library_note_compact_deep_link_intent_opens_notes_stage[context2-#library-note-body-editor-False] -- fail in the whole-file run on dev 91757b61e9-equivalent head e97b8fa736, pass in isolation (order-dependent)
- test_library_shell.py test_library_note_* group (11) -- red in whole-file runs, not on this census before (PR #2400 final review)
- generated_stylesheet test and the notes never-mount class (169 notes tests never mount in whole-file runs) -- carried from PR A's whole-file baseline (226 failing / 825)
- Test residue to clean: reader-shell colour assertion is satisfied by :hover (bold is the real cue); the Find-position test never asserts "Match 2" after Next; full-row equality in the join test is brittle; the search AC#1 typing stage is not exercised (class set only on submit); decorative assertion test_library_media_trash.py:2615; Docs/superpowers/qa/library-reader-grip-polish-2026-09 README/geometry.json/_capture_grips_impl.py:128 still describe accent end-caps as the shipped focus treatment (add a retirement note)

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the six tests passes on dev, or is rewritten/removed with the reason recorded in this task (no bare skip markers)
- [ ] #2 The root cause of the `#library-media-edit never mounted` group is identified and recorded, whether the fix lands in production code or in the test contract
- [ ] #3 test_library_shell.py, test_library_per_click_recompose_t21116.py, test_library_review_round_t21116.py and test_library_choice_strips.py run green in separate processes on dev
<!-- AC:END -->
