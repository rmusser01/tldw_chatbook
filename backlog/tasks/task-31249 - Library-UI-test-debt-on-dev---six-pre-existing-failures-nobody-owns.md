---
id: TASK-31249
title: Library UI test debt on dev - six pre-existing failures nobody owns
status: Done
assignee:
  - rmusser01
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
- The Media controller's own exclusion debt (89 fixture-shape exclusions and 7 named move candidates) is NOT on this census -- those tests all PASS. It is TASK-32013, filed separately because its done condition is about fixture shapes blocking extraction, not about failing tests

Wave 8 (2026-09-08, the notes series -- the FINAL extraction wave) -- each proven at the wave-8 start commit `889e12b86` in an isolated `git worktree` with its own `uv venv`, interpreter parity (3.14.2 both) and package parity (identical 106-name `uv pip list`) verified before any count was read:
- Tests/UI/test_library_file_notes_workspace.py::test_production_compact_folder_files_disclosure_and_states_are_painted -- a paint settle race (`assert 'Export exact copy' in 'Saved'` on `Button(id='file-notes-save-copy')`), red on BOTH trees with the identical assertion. Rates rather than a verdict, per the recipe's disposition rule: **6/10 on the branch, 3/10 at the isolated parent** in a matched, interleaved batch. New to this census
- Tests/UI/test_library_recompose_ratchet.py::test_library_screen_whole_screen_recompose_count_is_ratcheted -- **still 66 found / 63 allowed**, measured byte-identically by all four wave-8 tasks and at the isolated parent. Already on this census (wave-5 additions) and re-confirmed rather than re-derived; recorded here only so the number's age is visible: it has not moved in three waves
- Tests/Notes/test_notes_sync_cutover.py::test_library_screen_has_no_legacy_timer_worker_or_mutating_handler -- was RED at `889e12b86` and is now GREEN, for a reason the guard did not intend: the field it censuses moved into `LibraryNotesState` and its `ast.Attribute.attr` no longer matches the guard's prefix. It is now permanently vacuous and not naively retargetable (a prefix repoint false-positives on `_library_notes_sync_controller`, the WIRING binding accessor). **Filed as TASK-32088** (filed as TASK-32040; renumbered at the round-2 reconciliation when dev independently minted its own 32040); not test debt on dev, a product decision for whoever owns the notes-sync cutover
- Not a failing test, but found by the same wave and filed rather than left: the `canvas_sync.py` shared dispatchers take two receiver types with no guard, and 8 of the 10 controllers this program created declare no state accessor. One live production defect had already shipped from this shape (media Select-all/Clear silently full-screen recomposing) and was fixed at the program close. **Filed as TASK-32089** (filed as TASK-32041, renumbered to 32047 when dev minted its own 32041, renumbered AGAIN to 32089 at the round-2 reconciliation when dev minted its own 32047 — twice-struck by the same collision within one wave endgame)
- Wave 8 is the SECOND wave to pay TASK-31880's bill: the receiver defect above had to be guarded with a hand-built screen double, because `test_library_honesty_accessibility.py::test_row_toggle_patcher_rebuilds_marker_label_both_directions` -- the only real-row Pilot test that drives the `_apply_library_row_toggle` path end to end -- is still RED. Phase C changes exactly that path, in the widgets, for real

## Implementation Plan (added 2026-10-02, at base 2612fc56b2 = origin/dev tip)

1. Verify the premise at this base: run the eight census names (the four AC#3 files' named failures) and classify each as fixed-upstream / live defect / known class. Result recorded in Implementation Notes.
2. Root-cause the `#library-media-edit never mounted` group (AC#2): trace the media row press -> viewer mount path in the current (post media-series decomposition) code.
3. Enroll only the tests this task needs in the bootstrap profile (per-node `@pytest.mark.bootstrap_profile`, TASK-32873/ADR-179 precedent) to clear the config-participant admission class (`RecoveryRequired("raw_source_selection_changed")`) that masks every scenario at this base.
4. Fix each live defect red -> green with targeted runs only; baseline A/B via `git checkout HEAD -- <paths>`.
5. Verify AC#3: each of the four files green in its own process (separate invocations).
6. Close honestly: tick what is verified, annotate anything unreachable at this base, one commit.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each of the six tests passes on dev, or is rewritten/removed with the reason recorded in this task (no bare skip markers)
- [x] #2 The root cause of the `#library-media-edit never mounted` group is identified and recorded, whether the fix lands in production code or in the test contract
- [ ] #3 test_library_shell.py, test_library_per_click_recompose_t21116.py, test_library_review_round_t21116.py and test_library_choice_strips.py run green in separate processes on dev
  - **task-31249 closeout (2026-10-02, base 2612fc56b2): PARTIAL, and the unmet half is not this task's to take.** The three focused files are green in separate processes (9, 6 and 16 passed, 0 failed, commands in Implementation Notes) after whole-module `bootstrap_profile` enrollment (TASK-32873 precedent: these suites are 100% real-app mounts). `test_library_shell.py` whole-file green is unreachable at this base by this task: at the plain dev tip the whole-file run is 788 failed / 55 passed, and a 150-test sample under temporary module enrollment still shows 30 scenario-level residual failures (~20%) beyond the admission class -- so module-wide enrollment for the big Library file (explicitly reserved to the config-admission class owner per TASK-32873/ADR-179) would not even finish the job; the residual ~160-name set needs its own triage owner. Left unticked per the TASK-14877 precedent rather than laundering a partial pass.
<!-- AC:END -->

## Implementation Notes (2026-10-02, base 2612fc56b2 = origin/dev tip, worktree fix/task-31249-library-test-debt)

### Premise verification

Every named test was masked at this base by the config-participant admission class
(`RecoveryRequired("raw_source_selection_changed")`, TASK-32628): the description's first
failure list names eight tests across the four AC#3 files (the title says six; the census
list is authoritative). After per-node `@pytest.mark.bootstrap_profile` enrollment
(TASK-32873 precedent), the ORIGINAL 2026-09-04 signatures re-emerged on all eight --
none had been fixed upstream. All eight were live defects (class (b)), fixed red -> green.

### Per-test disposition

| Test | Signature at base (enrolled) | Root cause | Disposition |
|---|---|---|---|
| per_click `test_media_viewer_substate_escape_is_viewer_scoped` | `#library-media-edit never mounted within 30.0s` | AC#2 root cause below | Contract fix: disclose "More" first; green |
| per_click `test_open_item_by_id_media_is_canvas_scoped` | 5x `refresh(recompose=True)` storm | task-31797 fired browse+facet requests BEFORE the media canvas existed (each sync missed the canvas and took the whole-screen fallback, racing the mount -- the M3 DuplicateIds shape); the detail worker's callback added a 5th racing refresh | Production fix (mount-first projection + M3 suppression, below); contract updated to the adaptive-architecture reality (structural `Screen.recompose()` rebuilds the rail -- measured: a plain rail conversations->media press rebuilds it too); green |
| per_click `test_export_open_from_media_is_canvas_scoped` | DEADLOCK: press hung >300s (pytest-timeout kill; reproduced at base with `git checkout HEAD -- <prod files>` A/B) | `_open_library_export_canvas` awaited the open-item projection inline from the media canvas's own press handler; the projection's structural branch `await self.recompose()` tears down the very canvas whose dispatch is running the handler -- Textual waits for that dispatch forever (instrumented: `recompose` ENTER, never EXIT) | Production fix: schedule the projection via `call_after_refresh` (screen helper `LibraryScreen._project_library_export_canvas`); counts worker starts after the surface lands; contract updated; green |
| review_round `test_viewer_substate_escape_refreshes_the_footer_shortcut_set` | edit never mounted, then footer set diff | AC#2 root cause + product change: entering metadata edit switches the Reader to its Info tab (`set_mode(..., "info")`, d3c4b44a9b) and Escape mirrors Cancel by dropping only the sub-state, so the post-Escape footer is the Info-tab set (no `ctrl+f find` -- correctly gated on the no-text tab) | Contract fix: disclose More; assert post-Escape set != sub-state set, then Read-tab press re-registers exactly the captured plain set; green |
| choice_strips `test_media_type_strip_works_in_both_layouts` | `compact class never reached True at (100, 30)` | The adaptive media reader (40a1b99576, 2026-09-20) deliberately drops the legacy `library-notes-compact` class off the canvas ("Adaptive readers share only the shell box-model contract") and paints `library-adaptive-compact` on the shell grid instead; `_notes_state.compact` was True all along | Contract fix: the regime probe accepts the mounted evidence from either regime (loop-safe bindings); green |
| shell `test_library_starter_deep_link_opens_hidden_collection_or_note_route` | worker `AttributeError: 'SimpleNamespace' object has no attribute 'active_authority'` | The census's own suspect, confirmed: `_LibraryEvidenceGates.install` faked `collections_capture_scope_service` with only the evidence method; the capture controller's entry load reads `scope_service.active_authority` first | Test-double fix (TASK-21232 class): fake extended with `active_authority=None` (the real contract's "no active authority" -> unavailable state, where the entry load correctly stops); green |
| shell `..._inplace_search_preserves_identity_focus_and_parse_count` | `Markdown(...) is Markdown(...)` | The Find bar composes only while open (task-31237's collapsed bar): OPENING it legitimately recomposes the viewer once to mount `LibraryMediaContentSearchControls`; the tests captured their identity baseline BEFORE the Find press | Contract fix: baseline captured after the bar mounts; submit + Next must not rebuild (still fully pinned); green |
| shell `..._inplace_search_chrome_paints_above_content` | same identity trap under the layout assertions | same | Contract fix: viewer/markdown baseline captured after the submit; layout assertions unchanged and passing; green |

### AC#2 root cause (recorded as required)

`#library-media-edit` has composed ONLY inside the viewer's collapsed "More" actions
disclosure since d3c4b44a9b (2026-08-24, "organize media reader modes and actions";
guarded further by task-31633/a4682f17e9). It no longer mounts merely because the viewer
opened, and no test-side contract was updated. Recorded in the new shared helper
`_disclose_media_viewer_more_actions` (Tests/UI/test_library_shell.py). The fix landed in
the test contract (the product change is intentional UX).

### Production changes (2 files, targeted)

1. `tldw_chatbook/UI/Screens/library_screen.py`
   - `_open_library_item_by_id` media branch: mount the media surface FIRST through the
     sanctioned `_apply_library_open_item_surface` projection, THEN issue task-31797's
     browse+facets requests (they now find the mounted canvas and patch in place).
     Removes the 5-refresh storm and the mount race.
   - `_apply_library_open_item_surface`: hold `_library_canvas_projection_depth` across
     the structural `await self.recompose()` too, so worker callbacks landing inside the
     window suppress instead of racing (extends the documented M3 rule).
   - `_sync_library_media_surfaces_or_recompose`: M3 suppression + resync-pending replay
     while a projection owns the surface (mirrors `_sync_library_canvas`'s rule).
   - New `_project_library_export_canvas` (the out-of-band export projection + counts).
2. `tldw_chatbook/UI/Library_Modules/library_export_controller.py`
   - `_open_library_export_canvas` schedules the projection via `call_after_refresh`
     instead of awaiting it inline (deadlock fix above). Net-zero line delta -- the file
     is at its size-ratchet budget (1453); helper lives on the screen.

### Enrollment decisions

- The three focused suites (per_click/review_round/choice_strips) are 100% real-app
  mounts: whole-module `pytestmark = pytest.mark.bootstrap_profile` (TASK-32873
  precedent). No per-node markers left inside them.
- test_library_shell.py: only the three named tests carry per-node markers. Module-wide
  enrollment for the big Library files belongs to the config-admission class owner; see
  the AC#3 annotation for the measurement that shows enrollment alone would not finish
  that file anyway.

### Evidence (exact commands + results, all `-p no:randomly`)

- Eight named tests: `8 passed` (36.40s).
- Separate processes: per_click `9 passed`, review_round `6 passed`, choice_strips
  `16 passed` (multiple runs).
- Base A/B (prod files at HEAD via checkout swap, same seven temporarily-enrolled
  neighbor suites): base 86 failed / 245 passed vs fixed 82 failed / 249 passed.
  Name-diff: 81 both (pre-existing drift in reader_flow/render_fixes -- the stale
  `#library-media-edit` contract extends there: `test_edit_metadata_from_read_routes_to_info_form_actions`
  et al. -- mapped to the class owner, NOT fixed here); 5 base-only (fixed by this work:
  reader escape/filter-anchor/l/t-key flows, empty-find-bar focus); 1 fixed-only
  (`test_more_reads_as_an_open_disclosure_while_it_is_open`) which passes 3/3 in
  isolation -- load flake, the documented Library-under-load churn class.
- Export deadlock at base: `git checkout HEAD -- <2 prod files>` -> test killed at
  --timeout=90 (93.68s) with the press hung; with the fix: `1 passed` in ~9s.
- Ratchets: `test_library_modules_size_ratchet.py` export_controller + library_screen
  rows pass (net-zero line delta in the controller; verified red before the trim).
  `test_library_media_wiring.py` passes; `test_library_recompose_ratchet.py` source
  pins pass (its 6 app-mounting tests error with the admission class at setup --
  pre-existing, unenrolled, not this task's).
- Lint: `ruff check` on all six changed files -- zero new findings vs the base tree
  (fixed the one B023 and one F401 my first pass introduced).
- test_library_shell.py whole-file baseline at this tip: 788 failed / 55 passed
  (1628s); enrollment probe (first 150 tests, temporary module marker, removed):
  120 passed / 30 failed.

ADR required: no -- test-contract updates plus bug fixes that route existing seams
(the task-21116 projection, the documented M3 suppression rule, `call_after_refresh`)
through their intended paths; no new boundary, storage, or service-contract decision.

Modified files: Tests/UI/test_library_per_click_recompose_t21116.py,
Tests/UI/test_library_review_round_t21116.py, Tests/UI/test_library_choice_strips.py,
Tests/UI/test_library_shell.py, tldw_chatbook/UI/Screens/library_screen.py,
tldw_chatbook/UI/Library_Modules/library_export_controller.py, this task file.
