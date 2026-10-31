# Roleplay frame B2b (TASK-33910.4): the re-pin census

Counted on dev `38623632e9` (2026-10-26) before any B2b code, then per task while B2b was dry-run
on the branch `b2b-draft-seq` (one commit per task). The plan
(`Docs/superpowers/plans/2026-10-26-roleplay-b2b-toolbar-new-import.md`) re-pins each consumer in the
task whose production change breaks it; this file is the census those re-pins were drawn from.
Line numbers are dev's at `38623632e9` and go stale as dev moves: re-locate by symbol.

## 1. The old toolbar's ids, the stacked class, the label setters and the layout helpers

From the toolbar survey (Survey 1, section 7).

**Production consumers:**
- **`PLP`:** 59 hits. That is the compose site, the `set_mode` gating `:888-934`, `on_mount` `:731`, the handlers `:1258-1291`, the label setters `:1151, 1201`, and the layout helpers `:739-781` plus `_TOOLBAR_BUTTON_CHROME_COLS` `:51`.
- **`PS`:** 24 hits:
  - search: `:1126` focus target, `:4759` and `:14409` clear, `:16605` focus, `:16676` Escape;
  - worker group `"personas-library-search"`: `:3835`, `:4413` (a name, not the widget);
  - `:1128` the New focus target;
  - `:3645-3651` server-mode disable (new / import / duplicate / tag);
  - sort and tag labels: `:4114-4119`, `:4284-4285`.
- `css/widget_defaults_self.tcss:2735-2768` (generated).

**Non-test consumers outside production:**
- **User Guide:** `Docs/User_Guide/roleplay-chat-dictionaries.md:85-100` (the rail table of Search / New / Import / Duplicate / Sort / Tag, plus the empty-state copy). Toolbar terms also appear in `roleplay-chat-dictionaries/characters-and-personas.md`, `chat-dictionaries.md` and `lore-books.md`.
- **QA capture script:** `Docs/superpowers/qa/personas-workbench/capture_personas_workbench.py:274` (`click("#personas-library-new")`) and `:290` (search).
- `character_conversation_navigation.py:33, 90` (`"personas-filter"`).

**Test consumers.** The numbers are hits for the toolbar ids, the stacked class, the label setters, the layout helpers, and `"Sort: "`/`"Tag: "` label strings. False positives in unrelated widgets are excluded.

| File | Hits | Breakdown | PR-gated? |
|---|---|---|---|
| `Tests/UI/test_personas_workbench.py` | 55 | search 14, new 21, import 1, duplicate 4, sort 2, tag 3, set_*_label 4, labels 6 | no (nightly) |
| `Tests/UI/test_roleplay_item_list.py` | 18 | new 1 (`:1016`), sort 4, tag 5, set_*_label 4, labels 4 (`:1227-1260`) | census |
| `Tests/UI/test_personas_library_pane.py` | 17 | search 4, new 3, import 4, sort 1, stacked 2, helpers 3 (`:50-56, 151-170, 196-205, 252-268, 584-634`) | no |
| `Tests/UI/test_actor_pack_creation_workflow.py` | 16 | new-actor-pack 13, import 1, import-actor-pack 2 (`:107-113`) | no |
| `Tests/UI/test_roleplay_item_list_app.py` | 12 | search 12 | workflow |
| `Tests/UI/test_personas_library_toolbar_layout.py` | 11 | whole file (144 lines; `_TOOLBAR_BUTTON_IDS` `:39-45`; 1 of 4 red on dev) | no |
| `Tests/UI/test_personas_dictionaries.py` | 7 | search 1, new 2, import 1, duplicate 3 | no |
| `Tests/UI/test_personas_lore.py` | 4 | new 1, import 1, duplicate 2 | no |
| `Tests/UI/test_destination_visual_parity_correction.py` | 4 | new 3 (`:1004, 2748, 2937`), "Sort: Recent" `:1747` | no |
| `Tests/UI/test_workbench_pane_focus.py` | 3 | search (`:176-186`) | no |
| `Tests/UI/test_personas_pending_save_drafts.py` | 2 | new | census |
| `Tests/UI/test_roleplay_hostile_names.py` | 2 | new `:357`; `"Tag: {name}"` `:196` (hostile tag in the label) | workflow |
| `Tests/UI/test_dictionary_settings_pending_save_drafts.py` | 1 | new | census |
| `test_personas_editor_save_in_place.py`, `test_personas_generation_wiring.py` | 1 each | new | no |
| `Tests/UI/test_buddy_character_review.py` | 1 | import `:320` | no |
| `Tests/ProductionApp/test_reactive_ownership_maturity.py` | 1 | search `:207` | no |

The total is about 156 hits across 18 test files. Two more files pin **row-19 interim geometry** that B2b must re-pin:
- `Tests/UI/test_roleplay_header.py:132-149`: `band = 13/14`, so the first item is at header_bottom + 14 = row 19.
- `Tests/UI/test_roleplay_header.py:152-171`: `len(rows) >= 13`, which becomes ≥19.

About 34 test sites click or press New to start a create flow. If `+ New` opens a chooser, each must pick Blank first or post `("create")` directly.

## 2. Import and New consumers

From the New and Import survey (Survey 2, section 8).

Counts are per file. `Tests/UI/b2b_survey_toolbar_probe.py` is the toolbar surveyor's untracked probe and is excluded.

| Pattern | Production files | Test files |
|---|---|---|
| `[^a-z_]_import_dialog_worker` | PS 3 | test_personas_workbench 4; test_buddy_character_review 1 |
| `_actor_pack_import_dialog_worker` | PS 2 | test_actor_pack_import_workflow 4 |
| `_open_import_dialog` | PS 2 | test_personas_workbench 3 (also, unrelated: Evals' `snippet_editor` and `test_evals_screen`) |
| `_import_character_from_path` | PS 2 | test_personas_workbench 15; test_uat_first_time_character_chat 2; test_personas_library_scale 1; integration/test_pocket_tts_character_roleplay 1 |
| `_import_world_book_from_path` | PS 2 | test_personas_lore 6 |
| `_import_dictionary_from_path` | PS 2 | test_personas_dictionaries 10 |
| `_character_import_filters` | PS 2 | test_personas_workbench 3 |
| `action="import"` | LP 1 | test_personas_workbench 3 (incl. `test_import_requires_characters_mode` :3639-3662, which pins "Import does nothing in personas mode"; the unified picker changes that) |
| `personas-library-import"` | LP 3, PS 1 | library_pane 4, toolbar_layout 2, dictionaries 1, lore 1, workbench 1, actor_pack_creation 1, buddy_character_review 1 |
| `personas-library-import-actor-pack` | LP 3 | test_actor_pack_creation_workflow 2 |
| `personas-library-new"` | LP 2, PS 2 | 13 test files: workbench 21, library_pane 3, destination_visual_parity_correction 3, dictionaries 2, pending_save_drafts 2, toolbar_layout 2, the other six 1 each (generation_wiring, editor_save_in_place, hostile_names, item_list, lore, dictionary_settings_pending_save_drafts) |
| `personas-library-new-actor-pack` | LP 3 | test_actor_pack_creation_workflow 13 |
| the picker contexts (`character_import` etc.) | EFP 11, PS 4, buddy_conversion 1, ccp_handler 1 | test_file_picker_start_dir 10, test_file_picker_action_tooltips 7, test_uat_first_time_character_chat 1, Utils/test_egress_evict_close 1 |

**User Guide pages that name the Import button or the picker titles:**
- `Docs/User_Guide/roleplay-chat-dictionaries/chat-dictionaries.md:36,147`
- `characters-and-personas.md:34,297,407,457`. Line 297 already says "Import Character Card", but the real title is "Import Character Card or Buddy"; that doc is stale on dev.
- `lore-books.md:32,118,167-169`
- `roleplay-chat-dictionaries.md:68`

## 3. What each task re-pinned in the dry run

Quoted from the dry run's task ledger; the task numbers are the plan's.

### Task 1

- Tests re-pinned (#personas-library-duplicate -> #personas-duplicate): test_personas_workbench.py (copy test, conflict test, source-switch control list, server-tooltip list gains #personas-duplicate), test_personas_dictionaries.py (2 clicks + test_duplicate_button_hidden_only_for_personas re-pinned to the open item's kind; its character/persona legs now live in the new tests), test_personas_lore.py (1), test_personas_library_toolbar_layout.py (_TOOLBAR_BUTTON_IDS).
- Census: git grep 'personas-library-duplicate' in Tests/ tldw_chatbook/ scripts/ Docs/User_Guide Docs/superpowers/qa -> 0 (historical plans and the spec keep it; the User Guide names Duplicate in words, T12).

### Task 2

- Re-pins (census: 34 New sites + 13 New Actor Pack sites): start_new() in test_personas_workbench (20 clicks, 4 Ctrl+N), dictionaries (2), editor_save_in_place (1 click, 2 Ctrl+N), generation_wiring (1), lore (1), hostile_names (1), pending_save_drafts (2 press, 1 Ctrl+N; census-gated), dictionary_settings_pending_save_drafts (1 press; census-gated); test_actor_pack_creation_workflow (11 clicks -> start_new(<kind>:portrait); the server-persona display assert -> the chooser lists persona:portrait disabled; the keyboard-focus test -> + New focus, Enter, chooser offers character:portrait); test_personas_library_pane actions ["create","import"] -> ["new","import"].
- Zero refs left to personas-library-new-actor-pack / _new_actor_pack_pressed outside historical QA captures. Not re-pinned here: capture_personas_workbench.py:274 (T12, per the outline), test_destination_visual_parity_correction's action-id lists (the id is kept).

### Task 4

- Tests re-pinned (census: _import_dialog_worker 5 (workbench 4, buddy 1), _actor_pack_import_dialog_worker 4, _open_import_dialog 3, _character_import_filters 3, action="import" 3, #personas-library-import-actor-pack 2, "Import Actor Pack" label 1, picker tooltips 2, the personas empty copy 2, the interim band 1):
- test_personas_workbench.py (reentrancy test -> _roleplay_import_worker; test_import_requires_characters_mode deleted, re-pinned as test_import_is_offered_in_every_kind; server-disabled control list loses #personas-library-import; the fail-closed test drops "import" from the refused actions and checks _start_character_import returns None; the 3 filter tests -> "Roleplay files"/"Markdown"/"All Files" (packs in, Markdown out); test_personas_mode_empty_state_copy_unchanged -> ..._offers_new_and_import),
- test_buddy_character_review.py (_roleplay_import_worker), test_actor_pack_import_workflow.py (4 calls -> _import_actor_pack_from_path, one fewer push each), test_actor_pack_creation_workflow.py (the Import Actor Pack test -> an Actor Pack picked in the one Import says unavailable), test_personas_library_pane.py (Import shown in every mode; personas copy names Import), test_file_picker_action_tooltips.py (generic copy), test_roleplay_header.py (interim band: 11 at >=120, 10 at 80x24; measured y 15/14/15).
- Zero functional refs left to the removed ids/helpers (remaining hits are historical comments and absence assertions; Evals' own _open_import_dialog is unrelated).

### Task 5

- Tests re-pinned: test_personas_dictionaries.py::TestDictionaryImport::test_import_completion_skipped_when_mode_changed (its "navigated away" was a raw state.active_mode write before the call, i.e. no newer intent; now it captures the intent, switches with _apply_mode, passes intent=, and reads "Imported Late Arrival · in Chat dictionaries" for dev's "open Dictionaries to see it").

### Task 7

- Tests re-pinned (census: personas-library-(sort|tag|toolbar|filterbar|stacked-controls), _sync_control_layout/_required_toolbar_row_width/_TOOLBAR_BUTTON_CHROME_COLS, set_sort_label/set_tag_label, "Sort: "/"Tag: " in Tests = 9 files): test_personas_library_toolbar_layout.py deleted (not censused; its F-039 test_preview_pane_anchors_top_of_center_canvas moved verbatim into test_personas_workbench.py::TestPreviewIntegration); test_personas_library_pane.py (test_sync_control_layout_logs_and_keeps_state_on_failure deleted: its subject is gone); test_personas_workbench.py (_shared_pane_publication reads the sort/tag tooltips; _stamp_owner_labels + _OWNER_SORT/_OWNER_TAG replace the stamped label text in 4 ownership tests; the failed-reload labels "Name" / tag tooltip off); test_roleplay_hostile_names.py (the "Tag: {name}" leg: the tag is painted by the border title only, covered by test_the_items_border_title_paints_the_tag_literally); test_actor_pack_creation_workflow.py ("Import…"); test_roleplay_item_list.py (the 3 _ToolbarHost tests kept, the 2 label tests re-pinned to keys).
- Not re-pinned (outline items that need none): test_destination_visual_parity_correction.py:1747 "Sort: Recent" is the Artifacts screen; test_workbench_pane_focus.py and ProductionApp/test_reactive_ownership_maturity.py use #personas-library-search, whose id is kept.
- Zero refs left in Tests/ tldw_chatbook/ scripts/ to personas-library-filterbar, personas-library-stacked-controls, _sync_control_layout, _required_toolbar_row_width, _TOOLBAR_BUTTON_CHROME_COLS, set_tag_label (the fastpath comment names the old rule; T12 owns the guide and capture script).

### Task 7 (a fixup found by the batch gate)

- Ruling: a button-label census ("New", "Import", "Sort: ", "Tag: " as label sets) belongs in the plan's T7 census step besides the ids — cost if wrong: another fixup.

### Task 9

- Tests re-pinned (census: update_rows noun= 32 calls in 3 files + 1 kwargs.get("noun") in workbench; the status copies 4; list_state consumers 0 state asserts; Escape-as-probe 2):
- noun= removed from test_personas_library_pane.py (20), test_roleplay_item_list.py (9), test_personas_dictionaries.py (3); test_personas_workbench.py:9428 the cleanup-writer detector reads the "Characters" title; :9987 the book recovery copies; test_personas_dictionaries.py:850 "No chat dictionaries yet";
- test_roleplay_item_list_app.py::test_a_leave_inside_a_replaces_paint_reloads_on_the_return: the held paint moves from no-match -> rows (which no longer yields: the block updates in place) to recovery -> rows (Retry's path);
- ::test_the_four_key_find_opens_the_first_match_after_a_read_slower_than_2_s[pending|fired] and ::..._after_a_slow_server_search: the "a key is handled at once" probe is space (_record_spaces) instead of Escape — Escape on the list now clears the search (rung 8) and would end the find under test.
- Not re-pinned: test_personas_library_pane.py:59/:267 and test_personas_workbench.py:1324 (the copy is unchanged for characters and personas), test_destination_visual_parity_correction.py:1005 (any-of markers), mcp_rail.py's own update_rows.

### Task 10

- Tests re-pinned: test_roleplay_item_list.py::_status_lines (T9's helper) reads the two line ids (T10's targets share the copy class).

### Task 12

- Census ($EV/draft/t12_census.sh -> draft/logs/t12-census.txt; code, tests, scripts, User Guide, the QA capture script, workflows; historical Docs/QA, plans, specs, dated QA evidence and backlog excluded): 0 functional refs to personas-library-{new-actor-pack,duplicate,filterbar,stacked-controls}, _sync_control_layout, _required_toolbar_row_width, _TOOLBAR_BUTTON_CHROME_COLS, _open_import_dialog, _character_import_filters, _open_{lore,dictionary}_import_dialog, _{lore,dictionary}_import_dialog_worker, PersonasScreen._import_dialog_worker, set_tag_label, _new_actor_pack_pressed, _import_actor_pack_pressed, _LIBRARY_SORT_LABELS, "Import World Book", "Import Dictionary"; personas-library-import-actor-pack 2 (absence asserts); "Import Character Card" 4 (generic picker tests + the CCP screen's own picker, not Roleplay); "Sort: " 14 / "Tag: " 4 (the sort tooltip, Library's own sort, comments recording dev's labels). Bare-label census (batch 3's gap): "New"/"Import" in Roleplay tests are guidance-copy asserts that still hold.
- Capture script :290 (#personas-library-search) needs no change: the id is kept.
