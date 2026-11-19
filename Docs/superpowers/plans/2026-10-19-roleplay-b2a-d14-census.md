# D14 dry-run census (Roleplay B2a, TASK-33910.3)

Run on a scratch worktree from origin/dev 4de437a5c5 on 2026-10-19 (workflow wf_784ebf80-c3f). Transcribed from the census agent's structured result by the controller. Generated tables: `census/` (counts.md, lvapi.md, per_file.md, refs.tsv, summary.json) by `d14_census.py`; runs: `d14-runs/`; prototype: `d14-prototype.patch`.

## Recommendation: D14 = A

RECOMMEND D14 = A (RoleplayItemList(OptionList) with database windows), as the spec decided. The census does not show that the migration 'cannot ride the slice'.

CRITERION. It was fixed before the final results were read. Fallback B is chosen if ANY of these holds:
- (1) More than 25% of the A-only migrating refs are SEMANTIC.
- (2) Some ref class has no helper path: a census row falls to rule F99 ('other use'), or a pinned behaviour is reachable only through private OptionList API (_lines, _clear_caches).
- (3) The prototype has a head-only failure caused by the design rather than unfinished wiring or not-yet-migrated tests.

MEASURED against each criterion:
- (1) No. 19 of 256 A-only refs are SEMANTIC (7.4%), inside 9 of 121 A-only functions.
- (2) No. There are 0 F99 rows, and every MECHANICAL rule maps to one of the five helpers or a public OptionList attribute. The prototype uses no private API.
- (3) No. Migrated files, final paired run: 2 head-only reds. Both are flakes on dev; each failed on base too when run alone (flake protocol). Unmigrated files: 115 head-only reds.
  - 111 are explained by migration refs, per their junit text (row-id NoMatches, ListView WrongType). One more, hostile_names, times out waiting on an M2 row-id predicate.
  - The other 3 are flakes on dev, each failing on base when run alone.

COST. B is not free: it must also touch the 57 'also under B' refs (pager into the border title; count line and unsaved badge leave). A adds 256 A-only refs, 237 of them MECHANICAL. The prototype migrated 237 of the 256 with helper calls. A buys PERF-22 AC#4 (0 widget mounts per keystroke), which B does not fix.

FINDINGS THE B2a PLAN MUST CARRY. None argues for B; each states its status on dev today and whether B2a fixes it.

F1. A focus border shifts option rows on mouse-down, so a click selects the wrong row.
- Status: design A only; not on dev today.
- Mechanism: Textual 8.2.8 OptionList.DEFAULT_CSS has '&:focus { border: tall $border }' (textual/widgets/_option_list.py:154). The mouse-down focuses the list, the border appears, every row moves down one line, and the click resolves on the neighbouring row or the border.
- Mutation evidence (rule removed with Edit, then restored with Edit; restore verified byte-identical to the patch):
  - mutant: '3 failed, 2 warnings in 30.78s', with 'assert '1' == '2'', 'assert ['a b'] == ['a.b', 'a b']' and 'assert [] == [('character'...tective Sam')]';
  - restored: '3 passed, 2 warnings in 29.94s'.
- Consequence: spec §1.5.3 asks for a 'thick $ds-action-focus' focus edge. B2a must paint it without changing the list's box (a border present in both states that changes colour, or a non-layout paint). Pin it with a test: a real click on row 2 of an unfocused list loads row 2.

F2. At 52x20 the rail is not painted, and dev's row click silently does nothing.
- Status: exists on dev today as a test-validity weakness, not a user bug. B2a surfaces it; B2a does not fix it.
- Evidence (probe-base-52x20.log): the row's region is Region(0,0,0,0), pilot.click returns False, and the widget at (0,0) is nav-overflow-hint-left.
- Effect: TestConversationsPanel._select_first_character[size0] passes on dev only because first-paint auto-select already loaded character 1. The strict click helper refuses unpainted rows, so the prototype moved that helper to via='enter'. Keep the strict click.

F3. Transitive consumers.
- The census counts each ref once, where it lives. Three more files go red only through an imported helper: test_buddy_character_review.py (seen red in the U run), test_persona_policy_rules_editor.py and test_petdex_import_review.py, which all import _open_editor from test_personas_persona_visual_authoring.py. Their list is in d14-runs/transitive.out.
- They hold no extra refs, but they belong in the plan's re-pin list and its gate file set.
- Similarly, test_personas_expression_slots.py has 1 M1 ref (in _select_character) that turns 45 tests red. Refs are not tests, and one helper edit fixes all 45.

F4. Spec figures that do not reproduce.
- '_row_dom_id imported 11x': 0 imports at the spec rev and on dev. It is a private pane helper (1 definition, 4 call sites) that the rewrite deletes.
- 'pagebar 7 in 2': 5 in 2.
- '46 typed ListView refs in 6 files': reproduces only as ALL refs to the list id. The typed form is 40 in 4 at the spec rev and 41 in 5 on dev.
- 'about 70 ListView-API uses': 62 in 5 by AST. Every highlighted_child use in the Roleplay tests is on #personas-conversations-list, which stays a ListView.
- The spec's test_personas_dictionaries.py:875-884 is test_empty_state_prompts_create, now at :835-842 (U4, UNCHANGED).

Other plan inputs:
- Non-test consumers B2a must touch:
  - PS:16497 (typed focus);
  - PS:1271-1276 (DEFAULT_CSS: ListItem.personas-library-row);
  - css/components/_agentic_terminal.tcss:868-901 (the 3 ListItem.personas-library-row broad rules; '#personas-library-rows:focus { outline:none }' stays pinned, R18);
  - css/build_css.py:535,563 (the boot pin);
  - Docs/superpowers/qa/personas-workbench/capture_personas_workbench.py:141,182 (row-id clicks).
- No unscoped query_one(OptionList) exists anywhere in the Roleplay DOM, and every OptionList message handler in Roleplay is selector-scoped. The new widget therefore collides with nothing.
- origin/dev has moved to ce4aa4a231. All diffs and runs are pinned to 4de437a5c5.

Suggested order for the plan: land the helpers, then the mechanical sweep with the prototype's patterns, then the 9 SEMANTIC rewrites, then delete the OBSOLETE refs with paging.

## Counts

Generated by $EV/d14_census.py (git objects only). Run: --repo .worktrees/roleplay-b2a --rev 4de437a5c5 --spec-rev c21d5cb04c --out $EV/census. A re-run reproduced every output byte-identically (cmp).

Unit and scope: lines matching (grep -c semantics), in Tests/**/*.py. Format: spec | at spec rev c21d5cb04c | on dev 4de437a5c5 (lines in files).

Spec ref classes:
- #personas-library-row-* refs: 91 in 12 | 91 in 12 | 95 in 14.
  - Strict per-row ids (no -meta): 87 in 11 at the spec rev, 91 in 13 on dev.
  - Dev files: test_personas_workbench 68, test_personas_library_pane 10, test_personas_dictionaries 4, test_personas_character_world_books_screen 2, test_roleplay_hostile_names 2, and 1 each in character_attach, character_editor_avatar, editor_save_in_place, expression_slots, persona_visual_authoring, persona_visual_identity_pack, subscription_readiness, roleplay_quit_guard, integration/test_pocket_tts_character_roleplay.
- #personas-library-rows refs (spec: 'query_one(..., ListView)'): 46 in 6 | 46 in 6 (as ALL refs) | 52 in 10.
  - Typed form: 40 in 4 at the spec rev, 41 in 5 on dev (dictionaries 26, library_pane 10, lore 2, workbench 2, hostile_names 1).
- ListView-API uses (.children/.index/highlighted_child/action_select_cursor/subscripts/iteration; AST data flow): 'about 70' | – | 62 in 5 (dictionaries 49, library_pane 6, lore 4, hostile_names 2, workbench 1).
  - The spec's '26 refs in dictionaries' are 26 list-id refs, carrying 49 API uses.
  - focus() is excluded (13 uses, generic Widget API).
- personas-library-* refs: 345 in 24 | 345 in 23 | 356 in 28.
- _row_dom_id imported: 11x | 0 | 0. It is a private pane helper (1 definition, 4 call sites).
- #personas-library-count: 14 in 4 | 14 in 4 | 14 in 4 (library_pane 6, workbench 6, destination_visual_parity 1, library_pane_paging 1).
- #personas-library-pagebar: 7 in 2 | 5 in 2 | 5 in 2 (library_pane_paging 4, library_scale 1). The whole pager family is 12 in 3.
- test_personas_library_pane.py: 24 tests (spec 24) | 24 | 24 tests, 49 personas-library- occurrences.
- test_personas_library_pane_paging.py: 8 tests (spec 8) | 8 | 8 tests, 19 occurrences.

Test seams (all U3 UNCHANGED: the names stay importable from PS):
- count_character_page: spec 7x patched | 15 in 4 | 15 in 4.
- get_character_page_for_ui: spec 3x patched | 16 in 4 | 16 in 4.
- PERSONAS_LIBRARY_PAGE_SIZE: no spec count | 9 in 2 | 9 in 2.
- Identity checks 'is personas_screen_module.<seam>': 10 in 1 (workbench).

Classes the spec does not list (dev):
- .personas-library-row class queries: 28 in 2.
- ListItem refs: 31 in 6.
- Paging state (page_offset / PersonaPageChanged / ...): 24 in 3.

Classification (statement-level refs, one rule each; census/per_file.md, census/refs.tsv). Format: MECHANICAL/SEMANTIC/OBSOLETE/UNCHANGED (also-under-B).
- Totals: 525 refs = MECHANICAL 237, SEMANTIC 27, OBSOLETE 49, UNCHANGED 212. Of these, 57 are 'also under B'.
- Per file:
  - test_personas_workbench: 109/6/16/83 (B 20)
  - test_personas_dictionaries: 68/3/0/15
  - test_personas_library_pane: 36/14/4/24 (B 7)
  - test_personas_library_pane_paging: 0/0/15/9 (B 15)
  - test_personas_library_scale: 0/0/14/8 (B 14)
  - test_actor_pack_creation_workflow: 0/0/0/16
  - test_personas_library_toolbar_layout: 0/0/0/13
  - test_personas_lore: 5/0/0/4
  - test_roleplay_hostile_names: 5/0/0/1
  - test_roleplay_header: 1/3/0/1
  - test_destination_visual_parity_correction: 0/1(C1)/0/4
  - roleplay_frame_harness: 2/0/0/2
  - 12 more files at 0-1 MECHANICAL each; the rest UNCHANGED.
- A-only migrating refs (MECHANICAL + SEMANTIC, not also under B): 256 in 121 functions in 18 files. 19 are SEMANTIC (7.4%), in 9 functions:
  - dictionaries: test_meta_row_geometry_not_clipped (S2, S3), test_meta_row_renders_two_lines (S3);
  - library_pane: 5 per-row CSS-class tests (S1);
  - workbench: test_server_start_chat_carries_exact_source_and_active_target (S4: server limit=page_size);
  - roleplay_header: test_seven_characters_show_at_120x36 (S2).
- F99 'other use' rows: 0.

## Prototype

Scratch worktree .worktrees/b2a-census (branch b2a-census-scratch, from 4de437a5c5).

PATCH: $EV/d14-prototype.patch = git diff 4de437a5c5. 8 files, +574/-433. It applies cleanly to 4de437a5c5, it was re-verified byte-identical after the F1 mutation restore, and it contains no origin/dev drift.

BUILT:
- RoleplayItemList(OptionList):
  - id #personas-library-rows, markup=False, public API only;
  - option ids '<kind>:<id>';
  - Content prompts: a 3-cell gutter (loaded marker and mark), a bold loaded name, a dim meta line;
  - one set_options per update, duplicate ids dropped and logged, the cursor restored by id;
  - OptionSelected re-posted as PersonaEntitySelected;
  - marks as an id set, using replace_option_prompt;
  - its own ':focus { border: none }' (F1).
- The pager and count line are kept on purpose, to isolate the A-only refs. The empty/recovery copy stays a Static, with today's ids, beside the list.
- PS:16497 is untyped.
- Five helpers in Tests/UI/roleplay_frame_harness.py: select_roleplay_item (via enter or a real click on the painted line; refuses unpainted rows), highlighted_item_id, visible_item_ids, item_at (index or id), list_state (rows/empty/recovery). first_list_item now returns the painted region.
- Migrated: every ListView use in test_personas_dictionaries.py; all of test_personas_workbench.py (the file with the most row-id refs, 68); test_personas_library_pane.py; test_personas_lore.py; test_roleplay_frame_harness.py.
- Coverage: 237 of 256 A-only refs (92.6%). 8 of 9 SEMANTIC functions were rewritten or needed nothing; the 9th, in roleplay_header, was left to test the prediction.

FINAL PAIRED RUN, migrated files (5 files; bootstrap plugin; -n 4; both arms concurrent; load average 55-100 from other sessions):
- base (4de437a5c5): '5 failed, 569 passed, 20 warnings, 6 errors in 910.03s (0:15:10)'
- head (prototype): '5 failed, 567 passed, 2 skipped, 20 warnings, 6 errors in 896.37s (0:14:56)'
- Both arms share 9 reds (6 TestPersonaHumanIdentityRemoval setup errors plus 3 failures).
- The 2 skips are the OBSOLETE unsaved-badge tests (R24).
- 2 head-only reds, both flakes on dev (flake protocol: alone, alternating):
  - test_independent_buddy_actions_use_shared_coordinator_without_persona_selection[size1]: base failed on run 4; never touches the list (a worker race under load);
  - TestConversationsPanel::test_task_31243_real_pilot_preview_back_and_focus_evidence[size1]: base failed on run 5; also red on base in the earlier base.log.
- 2 reds were base-only.

UNMIGRATED CONSUMERS (22 files, left unmigrated to test the census prediction; prediction written first, at 17:19:41, to d14-runs/predict_u.out):
- base: '56 failed, 380 passed, 20 warnings, 2 errors in 1737.58s (0:28:57)'
- head: '121 failed, 269 passed, 15 warnings, 49 errors in 1454.29s (0:24:14)'
- 115 head-only reds:
  - 111 are explained by migration refs in their junit text (92 row-id NoMatches, 20 ListView WrongType, by error line);
  - 1 more, roleplay_hostile_names, times out waiting on its M2 row-id predicate;
  - 3 are flakes on dev, each failing on base when run alone (actor_pack WorkerCancelled; the settings advanced-config geometry; roleplay_header[80x24-full], which errors in teardown on both arms).
- Every predicted-RED file went red, or was already red on dev (quit_guard 9/9, pocket_tts 1/1). test_roleplay_header::test_seven_characters_show_at_120x36 went red in both the full and mock tiers at its predicted S2/M5 ref.
- Every predicted-GREEN file stayed green, except test_buddy_character_review.py. That file went red transitively, through _open_editor imported from visual_authoring. Two more importers exist (d14-runs/transitive.out).

EARLIER ITERATIONS (logs kept):
- head1: '72 failed, 500 passed, 2 skipped, 24 warnings, 6 errors'. Of these, 31 were WrongType on not-yet-migrated typed lookups, 6 wrong-row selections (F1) and 4 unpainted 52x20 rows (F2).
- head2-wb: '4 failed, 24 passed'. All 4 were 52x20 (F2).
- head3-convpanel: '8 failed, 69 passed'.
- head-final: cut off at 86% when the previous session ended.

F1 MUTATION (the :focus override removed with Edit, then restored with Edit):
- mutant: '3 failed, 2 warnings in 30.78s'
- restored: '3 passed, 2 warnings in 29.94s'

Lint: ruff on the changed files shows 1 F401 (Tests/UI/test_personas_library_pane.py:6), which also exists on base.

CLEANUP: worktrees .worktrees/b2a-census (--force; its work is preserved in the patch) and .worktrees/b2a-census-base removed; branch b2a-census-scratch deleted. The B2a worktree .worktrees/roleplay-b2a is untouched (clean, at 4de437a5c5). Nothing was pushed or committed.

## Evidence

Generated tables and evidence (not committed) are under /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b2a-evidence:
- census tables: census/counts.md, census/per_file.md, census/lvapi.md, census/refs.tsv, census/summary.json;
- coverage and predictions: d14-runs/coverage.out, d14-runs/predict_u.out, d14-runs/compare_u.out, d14-runs/transitive.out;
- final paired runs: d14-runs/final-{M,U}-{base,head}.{log,xml,failed};
- flake protocol: d14-runs/flake-*.txt;
- F1 mutation: d14-runs/f1-{mutant,restored}.log;
- scripts: d14_census.py, d14-runs/final_runs.sh, d14-runs/flake.sh, d14-runs/coverage.py, d14-runs/predict_u.py, d14-runs/compare_u.py, d14-runs/transitive.py.
The patch is at /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b2a-evidence/d14-prototype.patch.

## Rule legend

The rule ids Task 10's per-file table uses (M1, F9, O5, U1-U5 …): each rule's class, whether design B would migrate it too, its count, and the replacement it takes. From the census's `census/per_file.md`.

| Rule | Class | Also under B? | Count | Meaning / replacement |
|---|---|---|---|---|
| F1 | MECHANICAL | no | 24 | list.index = N -> select_roleplay_item(..., index=N, load=False) / OptionList.highlighted |
| F11 | MECHANICAL | no | 5 | row.query(Static) text -> item.lines |
| F12 | MECHANICAL | no | 1 | rows.first() -> item_at(node, 0) |
| F14 | MECHANICAL | no | 1 | row.id -> option id |
| F2 | MECHANICAL | no | 3 | list.index read -> highlighted_item_id(node) / OptionList.highlighted |
| F20 | MECHANICAL | no | 2 | compare -> helper value |
| F3 | MECHANICAL | no | 17 | list.action_select_cursor() -> select_roleplay_item(...) (Enter) |
| F5 | MECHANICAL | no | 2 | list.children -> list_state(node).items |
| F6 | MECHANICAL | no | 1 | children[i] -> item_at(node, i) |
| F7 | MECHANICAL | no | 3 | iterate rows -> list_state(node).items |
| F8 | MECHANICAL | no | 3 | len(rows) -> len(visible_item_ids(node)) |
| F9 | MECHANICAL | no | 18 | _row_text(row) -> item.name |
| M1 | MECHANICAL | no | 64 | click a row by DOM id -> select_roleplay_item(pilot, node, '<kind>:<id>', via='click') |
| M2 | MECHANICAL | no | 14 | row existence / lookup by DOM id -> '<kind>:<id>' in visible_item_ids(node) / item_at |
| M3 | MECHANICAL | no | 9 | row-id literal in an expected value -> option id '<kind>:<id>' |
| M4 | MECHANICAL | no | 2 | meta text gathered by class -> item_at(i).meta / list_state(node).items |
| M5 | MECHANICAL | no | 28 | rows queried as widgets -> visible_item_ids / list_state(node).items / list_state(node).text |
| M6 | MECHANICAL | no | 1 | typed focus -> the type changes (RoleplayItemList), the call stays |
| M7 | MECHANICAL | no | 39 | typed list lookup -> RoleplayItemList / helpers (its uses are classified with it or on their own statements) |
| O1 | OBSOLETE | yes | 18 | pager widgets / page message / page-delta handler: deleted with the pager |
| O2 | OBSOLETE | yes | 17 | paging state, update_rows page kwargs, reset_offset reload kwarg |
| O3 | OBSOLETE | yes | 4 | page-size constant (A: windows of 200 / server 100 replace it) |
| O4 | OBSOLETE | yes | 6 | count line pinned EMPTY (F-033): the line itself leaves |
| O5 | OBSOLETE | yes | 4 | row unsaved badge: spec 1.5.3 drops the list's unsaved marker under either design (R24; the title row carries `unsaved`) |
| C1 | SEMANTIC | yes | 8 | count line text: moves to the border title (N of M) or the marks bar (N marked); re-pinned to the new home |
| S1 | SEMANTIC | no | 11 | per-row CSS class contract (ListItem classes): becomes prompt state (gutter glyph / bold) or the option-list component class |
| S2 | SEMANTIC | no | 4 | row-widget geometry: rows are option lines, not widgets; re-measure via painted lines |
| S3 | SEMANTIC | no | 2 | meta-line widget looked up by class (styling becomes a Content span) |
| S4 | SEMANTIC | no | 2 | server request shape pinned to the page size (A: server windows of 100) |
| F4 | UNCHANGED | no | 11 | list.focus(): generic Widget API |
| U1 | UNCHANGED | no | 5 | untyped list id: focus / scrollable_content_region / CSS rule (id kept) |
| U2 | UNCHANGED | no | 2 | focused.id == 'personas-library-rows' (id kept) |
| U3 | UNCHANGED | no | 52 | seam patch / identity check: names stay importable from PS (re-export convention) |
| U4 | UNCHANGED | no | 5 | empty / recovery Static ids (the prototype keeps them as Statics beside the option list) |
| U5 | UNCHANGED | no | 137 | toolbar / search / pane ids: B2a keeps today's toolbar until B2b |
