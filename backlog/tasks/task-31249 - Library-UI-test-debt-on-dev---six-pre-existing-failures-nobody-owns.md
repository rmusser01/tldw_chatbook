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

Wave 5 (2026-09-05..07) additions -- each established on clean dev, not on a media branch:
- Tests/UI/test_library_recompose_ratchet.py::test_library_screen_whole_screen_recompose_count_is_ratcheted -- RED on dev a4ef89a30: the Library surface has 66 whole-screen recompose sites against a ratchet of 63 (found during PR I Task 2). PR I's helper removed one (65 with it), PR J removed none. Someone owns draining the remaining sites or raising the ratchet with a recorded reason; until then it is a known dev red that every media PR has to discount
- Tests/UI/test_library_ingest_canvas.py::test_backend_switch_failure_restores_persisted_server_controls -- fails ~1 in 4 on dev and on PR J alike: `NoMatches: No nodes match '#label' on SelectCurrent(...)` raised inside `Mount` dispatch, i.e. a Textual `Select` reads its `#label` before the child mounts under load. Fix shape: wait on the Select's mounted state before driving it (PR J verification, 2026-09-06)
- Tests/UI/test_library_media_trash.py::test_media_trash_back_and_escape_restore_distinct_media_return[escape] -- load-sensitive: failed once at ~18 s while other suites ran, 0/6 when run alone on PR J and on dev. Its return-to-Media wait times out under load; anchor the wait on the mounted return state rather than a fixed clock (PR J verification, 2026-09-06)
- Tests/UI/test_mcp_workbench.py::test_test_tool_preview_escape_revokes_nonce_through_mounted_binding -- a flaky race on Escape -> unmount (`KeyError: No text-area--gutter key in COMPONENT_CLASSES`): failed twice in PR #2451's Fast Lane and once locally in a py3.11 minimal venv, then passed on identical heads. Not Library code at all (ToolTestApp is a standalone ConsolidatedCSSApp with no Library imports) but it lands in the same Fast Lane runs; same class as the ingest-canvas Select flake. Needs a settle/wait before Escape or a guard in the preview teardown (2026-09-06)
- Tests/UI/test_library_entry_compose_once.py::test_library_graduation_announcement_survives_reconcile_and_same_route_replace and ::test_library_notes_recompose_does_not_steal_newer_focus[reconcile|replace] -- red on dev 5f12507c13, i.e. after #2414 (perf/library-reuse-31521); found while landing PR F (2026-09-05)
- Environment fact for anyone comparing runs: whole-file `Tests/UI/test_library_shell.py` has been ~226-230 red in this environment since at least 2026-09-05 (the Notes-test block, ~32 s per test) and is identical across dev commits and branch heads. A whole-file pass/fail COUNT therefore proves nothing about a diff -- comparisons must use the failing NAME sets

Wave 7 (2026-09-07, the media series' `origin/dev` reconciliation merge, 306 commits) -- each proven on an isolated `origin/dev` worktree with its own venv at MATCHING Python (3.14.2) and an identical 106-package `uv pip list`, both trees verified to resolve their own `tldw_chatbook`:
- Tests/UI/test_library_screen_reuse.py::test_on_screen_suspend_stops_every_timer_in_isolation -- RED on dev 0bb00beaf: `on_screen_suspend` now calls `self._unavailable_navigation.clear_character_return(self)`, and this test builds its screen with `LibraryScreen.__new__`, which skips the `__init__` line that creates that attribute. `AttributeError: 'LibraryScreen' object has no attribute '_unavailable_navigation'`. dev never touched this file. Fix shape: seed `_unavailable_navigation` in the fixture alongside the `_media_state`/`_ingest_state`/`_prompts_state` seeds already there
- Tests/UI/test_library_modal_dismissal.py::test_library_modal_inventory_matches_declared_edges_bidirectionally -- RED on dev 0bb00beaf: `unresolved modal constructor in supported presenter: LibraryScreen._present_library_skills_import_choice_if_needed (SkillImportChoiceModal(snapshot.candidates))`. The inventory's AST resolver cannot resolve dev's construction shape. Not media
- Tests/UI/test_screen_navigation.py -- 32 failed / 110 passed on dev 0bb00beaf, and the SAME 32 names on the merged wave-7 branch. This is a large, unowned dev-side regression in its own right, much wider than the ~30 churning failures the wave-6 merge recorded
- Tests/Architecture/test_library_modules_size_ratchet.py::test_budget_is_not_left_slack_after_a_move[library_conversations_controller.py] -- RED on dev: the file shrank 1738 -> 1686 without its row being lowered, 52 slack against a 50 tolerance
- Tests/Architecture/test_library_modules_size_ratchet.py::test_controller_does_not_grow_past_its_budget[library_media_browse_controller.py] -- the standing dev red, fourth consecutive Library wave, now **649 vs a pin of 371** (410 at the wave-6 merge). Dev's creep has more than tripled the overshoot. Deliberately NOT re-pinned by any Library wave: no Library extraction has touched the file, and raising it from a passing branch would launder dev-side debt
- The Media controller's own exclusion debt (89 fixture-shape exclusions and 7 named move candidates) is NOT on this census -- those tests all PASS. It is TASK-31976, filed separately because its done condition is about fixture shapes blocking extraction, not about failing tests

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the six tests passes on dev, or is rewritten/removed with the reason recorded in this task (no bare skip markers)
- [ ] #2 The root cause of the `#library-media-edit never mounted` group is identified and recorded, whether the fix lands in production code or in the test contract
- [ ] #3 test_library_shell.py, test_library_per_click_recompose_t21116.py, test_library_review_round_t21116.py and test_library_choice_strips.py run green in separate processes on dev
<!-- AC:END -->
