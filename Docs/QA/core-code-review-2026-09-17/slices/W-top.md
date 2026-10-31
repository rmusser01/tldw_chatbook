# W-top — tldw_chatbook/Widgets/*.py (66 top-level files), 24,763 lines

Worktree `/Users/macbook-dev/Documents/GitHub/tldw-review` @ d8fb4053f9. Read-only; no app run; no full-suite run.

## Coverage

**Every one of the 66 files was put through two mechanical passes over its own AST**
(not grep): (a) an `Import`/`ImportFrom` census resolving every module reference in
`tldw_chatbook/**/*.py`, (b) a class-scope mutable-attribute scan and a `run_worker`
call-shape scan. Reading depth on top of that:

| file | lines | read |
|---|---|---|
| enhanced_file_picker.py | 2646 | **in full** |
| emoji_picker.py | 1136 | **in full** (data loader :119-698, screen :786-1136; CSS block skimmed) |
| settings_theme_editor.py | 1105 | **in full** |
| media_details_widget.py | 1009 | **in full** (symbol census + every `query_one`/`except`/timer/worker site + :373-445, :860-1009) |
| model_search_picker.py | 883 | **in full** (:1-400 read, :400-883 by symbol cluster — worker, render, blur paths) |
| project_skills_import_modal.py | 724 | sampled — :247-346 (worker + guard), :273-290 (cancel), :580-720 (run_worker sites) |
| settings_agents_panel.py | 702 | mechanical only (AST passes + `_set_status:701`) |
| base_components.py | 681 | sampled — symbol census + :330, :350, :635-681 (dead-module verdict) |
| splash_screen.py | 626 | sampled — :56-100 (`__init__`), :120-300 (config), :320-560 (timers/animation) |
| file_extraction_dialog.py | 617 | sampled — :125-140 (size format); dead module |
| voice_input_widget.py | 617 | mechanical only (timer/worker/except census); dead module |
| llamacpp_snapshot_manager.py | 585 | sampled — :47-250 (compose, timers, `_paint*`), :440-540 (workers) |
| activity_log.py | 580 | sampled — :80-90, :195-210, :280-290, :403-420, :490-505; dead module |
| workspace_create_modal.py | 578 | mechanical only (+ `_cancel:278`, `_perform_safe_cancel:282`) |
| AppFooterStatus.py | 546 | mechanical only (+ legacy-marker check) |
| audio_troubleshooting_dialog.py | 494 | sampled — :220-290 (the P0), :360-430 (level-meter timer) |
| password_dialog.py | 471 | sampled — every `logger.` call (secret-leak check) |
| template_selector.py | 452 | sampled — :120-130, :190-200, :270-345; dead module |
| settings_splash_screen_viewer.py | 407 | sampled — :20-110 (config loader + defaults) |
| detail_value_row.py | 365 | mechanical only |
| modal_dismissal.py | 357 | sampled — :110-300 (mixin contract, `_consume`, `_perform_safe_cancel`) |
| detailed_progress.py | 354 | sampled — :130-145, :270-355; dead module |
| status_dashboard.py | 353 | mechanical only; dead module |
| settings_image_gen_panel.py | 350 | sampled — :88-125 (`_key_source_line`/`_secret_placeholder`) |
| document_generation_modal.py | 330 | mechanical only; dead module |
| settings_video_gen_panel.py | 329 | sampled — :50-75 (the duplicate pair) |
| loading_states.py | 326 | sampled — :170-185, :240-255; dead module |
| dictation_performance_widget.py | 319 | sampled — :160-185, :310-319; dead module |
| conversation_selection_dialog.py | 307 | mechanical only |
| tool_message_widgets.py | 305 | sampled — :90-200 (the 4 truncate sites) |
| voice_blend_dialog.py | 292 | mechanical only |
| settings_web_search_panel.py | 291 | mechanical only (+ worker site :275) |
| destination_rail.py | 289 | mechanical only (+ :91-140 recompose path) |
| form_components.py | 284 | sampled — symbol census (canonical-home check) |
| compact_model_bar.py | 278 | mechanical only (+ legacy-marker check) |
| file_picker_dialog.py | 276 | sampled — :40-60, :235-276; dead module |
| enhanced_sidebar.py | 273 | mechanical only; dead module |
| lazy_widgets.py | 272 | mechanical only; dead module |
| workspace_persona_default.py | 257 | mechanical only (+ `_cancel:218`) |
| voice_profile_dialog.py | 250 | mechanical only |
| toast_notification.py | 247 | sampled — :70-115 (timers); dead module |
| delete_confirmation_dialog.py | 219 | mechanical only |
| settings_internal_prompts_panel.py | 214 | mechanical only (+ worker site :164) |
| prune_safe_select.py | 202 | sampled — :45-100 (compat guard) |
| confirmation_dialog.py | 190 | sampled — :135-165 (cancel contract) |
| status_widget.py | 182 | mechanical only |
| settings_advanced_config_panel.py | 176 | **in full** (:55-176 read; 4 worker sites) |
| pausable_progress.py | 171 | sampled — :1-100 (the interception contract) |
| settings_internal_prompts_editor_modal.py | 168 | mechanical only |
| tooltip.py | 160 | sampled — :40-60 (timers); dead module |
| recompose_capture_guard.py | 154 | sampled — :1-110 (documented contract) |
| feedback_dialog.py | 151 | sampled — :140-151; dead module |
| diff_widgets.py | 132 | mechanical only |
| workbench_focus.py | 132 | mechanical only |
| chunk_preview.py | 120 | **in full** |
| glyph_fallback.py | 120 | mechanical only |
| empty_state.py | 104 | mechanical only; dead module |
| cancel_confirmation_dialog.py | 103 | **in full** |
| destination_workbench.py | 75 | mechanical only |
| backup_group_selector.py | 65 | mechanical only |
| select_values.py | 53 | mechanical only |
| reader_scroll.py | 26 | mechanical only |
| custom_list_items.py | 8 | **in full**; dead module |
| chat_message_enhanced.py | 7 | **in full** (re-export shim, live) |
| __init__.py | 5 | **in full** |

Honest summary: 8 files read in full (≈6,900 lines), 27 sampled by symbol cluster,
31 covered only by the two AST passes plus their candidate rows. The depth went to the
largest files and to every candidate row; the un-sampled 31 are small, and 12 of them
turned out to be dead modules.

## Findings

Ordered P0→P3, then D1→D4. Retirement write-ups follow the graded findings.

### P0 [D1] — The Audio Troubleshooting dialog passes a SYNC method to `run_worker` without `thread=True`; Textual raises inside the worker and, because `exit_on_error` defaults to True, kills the app
- Where: `tldw_chatbook/Widgets/audio_troubleshooting_dialog.py:246-248`
  ```python
  self.audio_devices = await self.run_worker(self._get_devices_safe).wait()
  ```
  `_get_devices_safe` is a plain `def` (`:275`). Reached from `_initialize_audio` (`:229`), started at `:226` by `self.run_worker(self._initialize_audio())`.
- Evidence (verified by execution):
  ```
  cd $WT && source env.sh && PYTHONPATH=$WT $PY -   # minimal Textual app, same call shape
  ... textual/worker.py:343 in _run_async
      raise WorkerError("Request to run a non-async function as an async worker")
  <full Textual crash panel printed>
  run_worker(sync_callable) without thread=True ->
     {'outcome': "WorkerFailed: Worker raised exception: WorkerError('Request to run a non-async function as an async worker')"}
  ```
  Mechanism in the installed Textual 8.2.8: `worker.py:342-343` rejects a sync callable on the async path; `worker.py:149` `exit_on_error: bool = True`; `worker.py:382-384` `if self.exit_on_error: app._handle_exception(WorkerFailed(self._error))`.
  Reachability: `grep -rn AudioTroubleshootingDialog tldw_chatbook/` → pushed at `UI/Dictation_Window_Improved.py:735` and `:1047` (`_show_troubleshooting`), and that module is imported by `UI/STTS_Window.py:105`.
- Why it matters: the dialog exists to diagnose a broken microphone; opening it takes the whole app down with a traceback panel. The local `except Exception` at `:265` catches the `WorkerFailed` that `.wait()` re-raises and writes "❌ Unexpected error" — but `app._handle_exception` has already fired from inside the worker, so the swallow does not save the app; it only hides the cause from the log.
- Recommended correction: `self.run_worker(self._get_devices_safe, thread=True, exit_on_error=False)` — device enumeration is exactly the blocking C call a thread worker is for. (`exit_on_error=False` because the method already returns `[]` on failure, `:277-281`.)
- Size: S · ADR: no · Confidence: verified (mechanism executed; end-to-end button press is in Left UNVERIFIED)
- Pinning test: none — `rg -n audio_troubleshooting Tests/` finds nothing.
- Already covered: none

---

### P1 [D2] — The splash screen re-reads the config file 9 times to build one 9-key dict, costing ~59 ms of every app launch before first paint
- Where: `tldw_chatbook/Widgets/splash_screen.py:213-224` (`_load_splash_config`, a dict comprehension calling `get_cli_setting` once per key; called from `__init__:83`), plus a 10th read at `:280` (`_get_predefined_cards` → `splash_screen.custom_image_path`). Near-identical copy at `tldw_chatbook/Widgets/settings_splash_screen_viewer.py:61-79`.
- Evidence (verified):
  ```
  cd $WT && source env.sh && PYTHONPATH=$WT $PY -
  current per-key loop (9 keys): 58.9 ms
  one section read + .get():      6.34 ms      (~9x)
  9x get_cli_setting warm:       48.5 ms       (5.4 ms/call warm — matches the sibling's 11 ms cold figure)
  ```
- Why it matters: `SplashScreen.__init__` is on the startup path of every launch, and 9 of the 10 reads are for keys in the SAME `[splash_screen]` table — `get_cli_setting` pays a full settings resolution per call. This is not a cache-miss story: the numbers above are all warm.
- Recommended correction: one section read — `section = get_cli_setting("splash_screen", default={}) or {}`, `effects = section.get("effects", {})`, then `.get(key, default)` per key. `get_cli_setting`'s own docstring (`config.py:8447-8477`) documents the section-only call shape. Put the resulting loader in ONE place (see the D4b below) and have both callers use it.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none

---

### P2 [D1] — A saved theme whose name contains a Rich markup tag can never be re-opened, and the failure is completely silent
- Where: `tldw_chatbook/Widgets/settings_theme_editor.py:297` (`theme_name = str(event.node.label)`), `:635`/`:641` (`_write_theme_file` duplicate-leaf check), `:865`/`:867` (`_delete_user_theme` node removal), and the silent half at `:339-361` (`load_user_theme`: `if theme_path.exists():` with **no** `else`).
- Evidence (runnable, verified):
  ```
  cd $WT && source env.sh && PYTHONPATH=$WT $PY -   # Tree.add_leaf -> str(node.label)
  'Solar [Flare]'   ACCEPTED by validate_filename   -> 'Solar [Flare]'  round-trips=True
  'My [bold]theme'  ACCEPTED by validate_filename   -> 'My theme'       round-trips=False
  'Solar[x]'        ACCEPTED by validate_filename   -> 'Solar'          round-trips=False
  ```
  Cause: `textual/widgets/_tree.py:858-859` `Tree.process_label` runs `Text.from_markup()` on every `str` label, so `add_leaf(theme_name)` stores markup-*parsed* text. `Utils/path_validation.validate_filename` (`:284-330`) rejects only separators / `..` / NUL — `[` is accepted at save time.
- Why it matters: save a theme as `my [x]theme` → the file `my [x]theme.toml` is written, the tree shows `my theme`, clicking that row calls `load_user_theme("my theme")`, `theme_path.exists()` is False and the method **returns with no notify and no log** — the theme silently cannot be reloaded. The same mismatch makes `_write_theme_file`'s `theme_exists` check (`:635-641`) never match, so each save appends a duplicate leaf, and `_delete_user_theme` (`:865-867`) leaves a stale row behind after deleting the file. Same defect class as the confirmed sibling P1 at `Chat/console_display_state.py:93`.
- Recommended correction: two independent one-liners. (1) Stop round-tripping identity through the label: store the name in `node.data` (already a dict-free slot carrying only `"user"`/`"catalog"` — make it `("user", theme_name)`) and read *that*, exactly as `Library/library_rag_state.py:392` documents for the same class. (2) Give `load_user_theme` an `else: self.app.notify(f"Theme file for '{theme_name}' not found", severity="warning")` — a missing file must not be a silent no-op regardless of cause.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none — `rg -n "process_label|node.label" Tests/` finds nothing for this widget.
- Already covered: none

---

### P2 [D2] — Opening any enhanced file picker costs ~9.6 ms of synchronous config reads on the event loop, because `RecentLocations` was left eager when its sibling `BookmarksManager` in the same file was made lazy
- Where: `tldw_chatbook/Widgets/enhanced_file_picker.py:52-57` (`RecentLocations.__init__` → `load_from_config()`), `:1308-1316` (`EnhancedFileDialog._get_last_directory`, called from `__init__` at `:1279`), and a third read at `on_mount` → `_update_bookmarks_list` → `BookmarksManager._ensure_loaded` (`:162-165`). 37 construction sites repo-wide.
- Evidence:
  ```
  cd $WT && source env.sh && PYTHONPATH=$WT $PY -
    get_cli_setting warm x20:        5.39 ms/call
    RecentLocations() x10:           5.55 ms/construct
    BookmarksManager() x10:          0.000 ms/construct   <- task-261 made this one lazy
    EnhancedFileOpen.__init__ x10:   9.58 ms each
  grep -rn "EnhancedFileOpen(|EnhancedFileSave(|EnhancedSelectDirectory(" tldw_chatbook/ | wc -l  -> 37
  ```
- Why it matters: `BookmarksManager`'s docstring (`:117-127`) says in so many words that per-construction config I/O was a stall hazard and was deferred by task-261 — but the fix stopped at one of the two managers. Every picker open still pays two config reads inline on the click handler (a third at mount), so the deferral bought ~0 for the common path.
- Recommended correction: give `RecentLocations` the same `_ensure_loaded()` shape `BookmarksManager` already has (`self._recent: Optional[list] = None`; load on first `get_recent`/`add`), and move `_get_last_directory()` out of `__init__` to the point where `effective_location` is actually needed — or fold both into the single deferred read the file already has machinery for. Canonical home: this file; no new helper.
- Size: S · ADR: no · Confidence: verified (timings above)
- Pinning test: none found asserting eager load. `Tests/UI/test_enhanced_file_dialog_mount.py` exists but does not pin construction cost.
- Already covered: none (task-261 covered `BookmarksManager` only)

---

### P2 [D2] — First open of the emoji picker blocks the event loop for ~180 ms building the emoji index inside `EmojiPickerScreen.__init__`
- Where: `tldw_chatbook/Widgets/emoji_picker.py:889` (`get_emoji_data()` called from `__init__`), building via `_load_emojis()` at `:119-684`; cache at `:688-698`.
- Evidence (verified):
  ```
  cd $WT && source env.sh && PYTHONPATH=$WT $PY -
  get_emoji_data cold=179.7 ms  warm=0.0004 ms  n_emojis=5225 n_categories=9
  _filter_emojis('sm') = 0.8 ms/call over 5225 emojis      <- debounce is fine
  ```
- Why it matters: CLAUDE.md's own rule is "Workers for operations >100ms". This is 1.8x over, on the click that pushes the picker screen, once per process — the picker visibly appears late and keystrokes queue behind it. The module already has the lazy-cache half of the fix (`_EMOJI_DATA_CACHE`); it just runs the expensive half on the loop.
- Recommended correction: keep the cache, move the *first* build off the loop — `__init__` composes with an empty grid, `on_mount` runs `self.app.run_worker(get_emoji_data, thread=True, group="emoji-index")` and populates on completion. The file already uses exactly this shape for its recents write (`_save_recent_emoji_off_loop`, `:1029-1051`).
- Size: M (needs a loading state in compose) · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none

---

### P2 [D3] — 21 of the 66 modules in `Widgets/` (7,512 of 24,763 lines — 30% of the slice) are imported by nothing in production; ~15 test files exist only to keep them alive
- Where (module, lines, test files that reference it):
  ```
  media_details_widget        1009   1 test     base_components              681   6 tests
  file_extraction_dialog       617   2 tests    voice_input_widget           617   4 tests
  activity_log                 580   2 tests    template_selector            452   1 test
  detailed_progress            354   2 tests    status_dashboard             353   1 test
  document_generation_modal    330   1 test     loading_states               326   2 tests
  dictation_performance_widget 319   1 test     file_picker_dialog           276   3 tests
  enhanced_sidebar             273   0          lazy_widgets                 272   0
  file_list_item_enhanced      263   1 test     toast_notification           247   1 test
  tooltip                      160   0(*)       feedback_dialog              151   2 tests
  chunk_preview                120   0          empty_state                  104   0(*)
  custom_list_items              8   0
  ```
- Evidence (two independent passes, not a bare grep):
  1. AST walk of every `Import`/`ImportFrom` in `tldw_chatbook/**/*.py`, resolving the last path component → these 21 have **0** importers.
  2. Per-module `rg` for the module's own public class/function names (`ast` list) excluding its own file → **0 production hits** for every one. The four that first looked live are name collisions, each checked by hand:
     - `MediaDetailsWidget` — the only prod mention is `Local_Ingestion/local_file_ingestion.py:1875`, a comment reading *"mirrors the **dead** ``MediaDetailsWidget`` writer"*.
     - `file_picker_dialog.create_filter` — `UI/Screens/chat_screen.py:20237` defines its own nested `create_filter`.
     - `tooltip.TooltipMixin` / `HelpIcon` — zero prod hits; the many `tooltip` matches are Textual's own `widget.tooltip` attribute.
     - `base_components.NavigationButton` — `UI/Navigation/main_navigation.py:204` **defines its own** `class NavigationButton(Button)`. Two classes, same name; the live screen uses its local one and `Tests/UI/test_focus_token_parity.py:22` imports the dead one.
     (*) `tooltip` and `empty_state` show many grep hits for the *words*; zero for their classes.
- Why it matters: (a) it is 30% of the package's maintenance surface — every CSS-bundle, focus-token and timer-inventory sweep pays for widgets no screen mounts (`Tests/Architecture/test_timer_path_static_update_inventory.py` already indexes three of them); (b) dead code hides live-looking defects — `dictation_performance_widget.py:172,175` carries an exact copy of the P0 `run_worker` crash above, and `base_components.NavigationButton` shadows the live one; (c) the tests give false confidence that these surfaces are covered.
- Recommended correction: delete in batches, one PR per group, with the existing precedent as the shape — `backlog/tasks/task-1280 - Delete-dead-Widgets-voice_input_button.py-zero-callers...md` did exactly this for a sibling widget. Delete the module *and* its tests together. Caveat: several carry `DEFAULT_CSS`, so run `./scripts/preflight.sh` (CSS bundle sync) in the same commit, and `Tests/UI/test_widget_css_consolidation.py` / `test_css_class_coverage_contract.py` / `test_timer_path_static_update_inventory.py` each need their rows dropped.
- Size: L (many PRs; each individually S) · ADR: no · Confidence: verified
- Pinning test: the ~15 test files above — none states the widget as a *product requirement*; they assert CSS/focus/timer hygiene of whatever exists.
- Already covered: none (task-31584 is a different helper; task-1280 is the precedent, already done)

---

---

### P2 [D4a] — `Widgets/base_components.py` (681 lines, 5 widget classes + 2 factories) has zero production importers; its `create_form_field` duplicates the live one in `form_components.py`
- Where: `tldw_chatbook/Widgets/base_components.py` — whole file. Duplicate surface: `base_components.create_form_field:652` vs `form_components.create_form_field:22`; `base_components.create_button_row:635` vs `form_components.create_button_group:131`; `base_components.FormField:41` (a dataclass) vs `form_components.FormField:175` (a Container).
- Evidence (AST, not grep):
  ```
  cd $WT && $PY -  # ast.walk every tldw_chatbook/**/*.py for Import/ImportFrom of base_components
  AST ImportFrom/Import of base_components in tldw_chatbook/: 0
  grep -rn base_components Tests/ -> 3 files, all keeping the dead code alive:
    Tests/UI/test_focus_token_parity.py:22        from ...base_components import NavigationButton
    Tests/UI/test_widget_css_consolidation.py:1120-1124  5 (file, class, "DEFAULT_CSS") rows
    Tests/UI/test_non_obscuring_focus_contract.py:17     BASE_COMPONENTS = ROOT / ".../base_components.py"
  ```
- Why it matters: 681 lines of widget code that nothing composes, plus three CSS/focus-contract tests whose only job is to police it — every future focus-token or CSS-bundle sweep pays for a file no screen renders. It is also the second `create_form_field` in the same package, which is what sends the next author to the wrong one.
- Recommended correction: delete the module and the three test references (the two CSS tests lose 6 rows; `test_focus_token_parity` needs a live widget instead). Canonical home for anything a future caller wants: `Widgets/form_components.py` (4 importers, live).
- Size: S · ADR: no · Confidence: verified
- Pinning test: the three above — none states the behaviour as a *requirement*, they assert CSS-token hygiene of whatever classes exist.
- Already covered: none

---

### P3 [D1] — `_update_breadcrumbs` wraps its whole body in `except Exception: pass`, so any breadcrumb-build failure silently yields an empty breadcrumb bar
- Where: `tldw_chatbook/Widgets/enhanced_file_picker.py:2108` (the `except` closing the try opened at `:2113`-ish; body is `:2114-2160`, ~45 lines including `remove_children()`, index arithmetic and mounts)
- Evidence: read only — the body begins with `breadcrumb_container.remove_children()`, so a failure *after* that leaves the bar empty with no log line. Same shape at `:2015` (`_load_recent_locations`, whole ListView population inside one bare `except Exception: pass`).
- Why it matters: the intent of these guards is "the widget may not be in the DOM" (`QueryError`); they are written broad enough to also eat an `IndexError` in the `visible_indices` arithmetic or a `MountError`, and the user sees a blank breadcrumb bar with nothing in the log.
- Recommended correction: narrow to `except QueryError` around the `query_one` only, or keep the broad catch and `logger.debug`/`logger.warning` it (the sibling `_update_bookmarks_list` at `:2053` already does exactly this — `except Exception as e: logger.error(...)`). Inconsistent inside one file.
- Size: S · ADR: no · Confidence: inferred (no reproduction attempted; `sed -n '2100,2165p'` settles the shape)
- Pinning test: none
- Already covered: none

---

### P3 [D3] — `SettingsThemeEditor.__init__` does filesystem I/O (`mkdir`) on a widget that is re-constructed on every Settings recompose
- Where: `tldw_chatbook/Widgets/settings_theme_editor.py:99-101` (`raw._mkdirs(operation)` inside `__init__`)
- Evidence: read only. The class's own `is_modified` docstring (`:42-46`) states that SettingsScreen recomposes on this editor's messages and "each recompose mounted an editor", so construction is not once-per-session.
- Why it matters: the identical pattern in `enhanced_file_picker.BookmarksManager` was removed by task-261 with an explicit "constructor is I/O-free" contract; this one is the same shape in the same slice.
- Recommended correction: move `raw._mkdirs` to the first write (`_write_theme_file` already opens a writing scope) or to `on_mount`.
- Size: S · ADR: no · Confidence: inferred (cost not measured; `mkdir(exist_ok=True)` is one syscall — this is a consistency finding, not a measured cost)
- Pinning test: none

---

---

### P3 [D3] — `config._get_effective_config_path` is a private helper used at 53 sites across 20+ packages
- Where: first use in this slice at `tldw_chatbook/Widgets/emoji_picker.py:71`. `grep -rn _get_effective_config_path tldw_chatbook/ | grep -v ^tldw_chatbook/config.py | wc -l` → **53**. One site already re-exports it deliberately with a noqa (`UI/Screens/chat_screen.py:485`), and `Utils/sensitive_paths.py:149,167,374` documents it in prose.
- Why it matters: the underscore is now a lie — it is the de-facto public config-path API. New code copies the private import, and nothing stops `config.py` from renaming it.
- Recommended correction: one line in `config.py` — `get_effective_config_path = _get_effective_config_path` — and migrate opportunistically. Not worth a 53-site rename PR.
- Size: S · ADR: no · Confidence: verified (count above)
- Already covered: none

---

### P3 [D3] — `Widgets/dictation_performance_widget.py` (319 lines) has zero users and carries the same `run_worker` defect twice
- Where: `tldw_chatbook/Widgets/dictation_performance_widget.py:172`, `:175` — `await self.run_worker(monitor.get_session_summary).wait()` / `monitor.get_provider_comparison`; both are plain `def` in `Audio/dictation_metrics.py:202,257`.
- Evidence: `grep -rn "DictationPerformanceWidget|dictation_performance_widget" --include=*.py tldw_chatbook/` → **no hits outside the file itself**. So the same defect proven P0 above is latent here, not shipped.
- Why it matters: it is a delete candidate that would otherwise look like a second P0 to the next reviewer.
- Recommended correction: delete the module. If it is kept, fix both call sites with `thread=True`.
- Size: S · ADR: no · Confidence: verified

---

### P3 [D4b] — Two copies of the splash config loader + defaults table, no shared home
- Where: `tldw_chatbook/Widgets/splash_screen.py:120-224` (private literal `default_config` + `_load_splash_config`) and `tldw_chatbook/Widgets/settings_splash_screen_viewer.py:23-32,61-79` (module constant `DEFAULT_SPLASH_CONFIG` + `_load_config`). The comprehension bodies — including the `_EFFECTS_KEYS = {"fade_in_duration","fade_out_duration","animation_speed"}` split — are line-for-line identical.
- Evidence: `sed -n '20,45p'` on the viewer vs `sed -n '122,131p'` on the screen — the 8 scalar defaults are byte-identical today (`enabled/duration/skip_on_keypress/card_selection/show_progress/fade_in_duration/fade_out_duration/animation_speed`); the screen's table adds `active_cards`. **No drift today**, so this is P3 and not a D1.
- Why it matters: the Settings splash *editor* and the splash *screen* must agree on what "unset" means; today they do only by hand-copy, and the viewer already imports from `splash_screen` (`:20 from ..Widgets.splash_screen import SplashScreen`), so there is no import-cycle excuse.
- Recommended correction: move `DEFAULT_SPLASH_CONFIG` and one `load_splash_config()` into `splash_screen.py` (the owner) and have the viewer import both. Fold in the P1 single-section-read fix at the same time — one PR, one function.
- Size: S · ADR: no · Confidence: verified
- Already covered: none

---

### P3 [D4a] — 4 hand-rolled `[:N] + "..."` truncations beside `Utils/Utils.truncate_content`, which is a drop-in
- Where: `tldw_chatbook/Widgets/tool_message_widgets.py:101` (`[:97]`), `:177` (`[:97]`), `:185` (`[:77]`), `:195` (`[:197]`); plus `enhanced_file_picker.py:2085` (`name[:12] + "..."` for a 15-char budget) and `detailed_progress.py:241` (dead module).
- Evidence: `Utils/Utils.py:253-261` — `truncate_content(content, max_length=200)` returns `content[:max_length - 3] + "..."`, i.e. **exactly** the `[:97]+"..."` / `[:197]+"..."` semantics at limits 100 and 200. `grep -rn "def .*truncate|def shorten|def elide" tldw_chatbook/{Utils,Chat}/` shows the wider cluster (8 more private truncators in `Chat/`, a sibling slice's problem).
- Why it matters: only consistency — the shape is identical, so there is no drift to find.
- Recommended correction: `from ...Utils.Utils import truncate_content` in `tool_message_widgets.py`; leave `Chat/`'s copies to whoever owns that slice.
- Size: S · ADR: no · Confidence: verified
- Already covered: none

---

### P3 [D4b] — Byte-identical credential-provenance strings in two settings panels, with a third spelling in the settings screen
- Where: `settings_image_gen_panel.py:92-116` and `settings_video_gen_panel.py:54-69` — `_key_source_line` and `_secret_placeholder` are line-for-line identical (the video panel's docstring even says "(image-panel strings)"). A third, differently-worded copy: `UI/Screens/settings_screen.py:12696` `"API key source: local config key saved"`.
- Evidence: `grep -rn "local config key saved" --include=*.py tldw_chatbook/` → 3 production sites; `sed` on both panels shows identical bodies. **No drift today.**
- Why it matters: these strings tell the user where their API key is stored. If one drifts, a user is told the wrong thing about a credential — which promotes this cluster to D1 the moment it happens. `backlog/decisions/012-provider-credential-settings-boundary.md` is the ADR that owns the boundary.
- Recommended correction: one `key_source_line()` / `secret_placeholder()` pair in a shared module (`Chat/provider_readiness.py` already owns key-source resolution), imported by both panels; settings_screen's variant then reads `f"API key source: {key_source_line(src)}"`.
- Size: S · ADR: no (012 already covers the boundary) · Confidence: verified

---

### P3 [D4b] — Three byte-size formatters in this slice alone
- Where: `enhanced_file_picker.py:431-447` (`_human_readable_size`, live), `file_extraction_dialog.py:129-135` (inline, dead module), `file_list_item_enhanced.py:236-242` (`_format_file_size`, dead module).
- Evidence: read all three; the live one is the most complete (trailing-zero trim, full unit ladder to YB). The brief records `Widgets/Console/console_transcript.py:639` as a *documented* copy of `Chat/attachment_core._format_size` — so the repo already has a named canonical home.
- Recommended correction: nothing to do if the two dead modules are deleted (see the P2 above); otherwise point `enhanced_file_picker` at `Chat/attachment_core._format_size` and delete `_human_readable_size`.
- Size: S · Confidence: verified

---

### Retired — all 23 `except Exception: pass` in `enhanced_file_picker.py` are DOM-presence guards, not data-path swallows
- Evidence: read the file in full. Every one of `:1314 :1514 :1524 :1536 :1552 :1563 :1591 :1754 :1856 :1938 :1965 :1971 :2015 :2025 :2053 :2108 :2228 :2275 :2280 :2304 :2320 :2367 :2579` wraps a `query_one`/`styles.display` toggle or an optional-widget lookup (`#filename-input` is genuinely absent in the multi-select and directory dialogs — `_input_bar` at `:2412`/`:2494`/`:2559` proves it). The single data-path write in this file, `_persist_recent_and_last_directory`'s `persist()` (`:1345-1352`), *does* log its exception.
- Two exceptions worth the P3 above (`:2015`, `:2108`) because the guarded block is 20-45 lines wide, not one lookup.

### Retired — `_confirm_single`'s unguarded `self.query_one("#filename-input", Input)` (`:1802`) is not reachable without that input
- Evidence: `_confirm_single` returns immediately when `self.multi_select` (`:1797`), which is the only `EnhancedFileOpen` shape whose `_input_bar` omits the field (`:2415`); `EnhancedSelectDirectory` (no filename input) adds `EnhancedFileDialog._on_select_button` to `_SUPPRESSED_BASE_HANDLERS` (`:2536`) and routes Select through `_select_viewed_directory` instead. Every other caller (`_on_select_file` `:1751`, `_on_open_file` `:1784`, `_on_path_input_submit` `:1690`) guards the same lookup first.

### Retired — `enhanced_file_picker.py` bypassing `Utils/path_validation.py` is deliberate and documented, and the one place validation belongs it is applied
- Evidence: `grep -n validate_path_simple` → imported `:41`, applied at `:2178` (`_jump_to_bookmark`, the one path that comes from *stored config* rather than the user's own keyboard). The two typed-path handlers (`:1636-1646`, `:2605-2616`) carry a written rationale for not applying it (it rejects `~/`, `../`, and shell metacharacters, all legal in a local picker) and both explicitly reject the one input that is never a path (NUL) before any `Path()` call. This is a local-filesystem picker, not a server-side trust boundary.

---

### Retired — `chunk_preview.py:98 len(chunk_text) // 4` is NOT a token estimate
- Evidence: read `:91-102`. It is `overlap_chars = min(self.overlap_size * 5, len(chunk_text) // 4)` under the comment "Simple approximation - show first N **characters** as overlap" — a character cap that stops the overlap preview exceeding a quarter of the chunk. `Utils/token_counter.estimate_tokens` is the wrong helper for it; swapping it in would change the rendering. The mechanical `token_est_len_div4` row mis-classified this site.

### Retired — `media_details_widget._format_text_for_reading` is not a hot-path cost, despite `import re` inside its per-paragraph loop
- Evidence (verified): `cd $WT && … $PY -` calling `MediaDetailsWidget._format_text_for_reading(None, text)`:
  `164 KB / 200 paragraphs -> 3.9 ms`, `41 KB / 50 paragraphs -> 1.0 ms`.
  The 24 abbreviation `.replace()` passes per paragraph and `re.sub` (which `re`'s own pattern cache memoizes) cost single-digit ms on a full transcript. The `import re` at `:429` sitting inside `for paragraph in paragraphs:` is a style nit (P3, move to module scope), not a measured cost.

---

### Retired — the 5 `_perform_safe_cancel` "copies" are the mixin's template hook working as designed; no behavioural drift
- Evidence: `modal_dismissal.py:256-259` defines `_perform_safe_cancel(self, *, source)` → `del source; self.dismiss_safe_once(None)` as the *default* terminal cancellation, invoked by `request_safe_cancel` (`:239-254`) which owns the re-entrancy guard, the generation token and the `finally` cleanup. The overrides differ only in the value each dialog must return on cancel: `cancel_confirmation_dialog.py:101-103` → `dismiss_safe_once(False)`; `confirmation_dialog.py:148-154` → runs `cancel_callback` once through `run_cancel_effect_once` then `dismiss_safe_once(False)`. That value IS the hook's reason to exist. Collapsing them would be a regression, not a cleanup. Same reading for the `_cancel`/`_close` 2-line adapters in this slice (`workspace_persona_default.py:218`, `workspace_create_modal.py:278`, `project_skills_import_modal.py:288`, `enhanced_file_picker.py:1721`): each is `event.stop()` + one `request_safe_cancel(source=...)` with a *different* source string, which the mixin records. P3 at most, and only if someone wants a `@on(Button.Pressed, "#cancel")` helper in the mixin itself.

---

### Verified-fine — no mutable class attributes anywhere in this slice, and no `run_worker(exclusive=True)` without `group=`
- Evidence: AST walk over all 66 files for `List`/`Dict`/`Set` literals assigned at class scope (non-CONSTANT names) → **0 hits**. AST walk for every `.run_worker(` call → 21 sites; every one that passes `exclusive` also passes `group`, and the three `exclusive=` values that appear without a group are `False`. Matches the brief's expectation (the one repo-wide offender is in a sibling slice).

---

## Candidate dispositions

| candidate | disposition |
|---|---|
| `query_one_in_timer_no_try` emoji_picker.py:967, :1027 (`_perform_search`, timer :957) | **retired** — `#search-results-grid` (`:923`) and `#search-input` (`:899`) are yielded unconditionally in `compose`; the widget never recomposes; Textual stops a pump's timers on close (`message_pump.py:533-535` → `Timer._stop_all`). No miss is reachable. |
| `query_one_in_timer_no_try` llamacpp_snapshot_manager.py:214 (`_paint_elapsed`, timer :141) | **retired** — `#snapshot-operation-status` yielded unconditionally (`:88`); callback guards `if not self.is_mounted or self._attachment is None: return` (`:209-211`). |
| `query_one_in_timer_no_try` model_search_picker.py:773 (timer :761) | **retired** — `#model-search-picker-input` yielded unconditionally (`:185`); callback guards `not self.is_mounted` (`:770`); no `recompose` anywhere in the file. |
| `except_exception_pass` × 23 in enhanced_file_picker.py | **retired** (all DOM-presence guards) except `:2015` and `:2108`, **confirmed** as the P3 above (20-45-line bodies inside one bare catch). |
| `except_exception_pass` activity_log:415, detailed_progress:282/336/353, loading_states:178, feedback_dialog:150, file_picker_dialog:242/275, media_details_widget:1008, project_skills_import_modal:326 | **retired** — every one wraps a single `query_one`/style toggle; `project_skills_import_modal:326` carries an explicit `# noqa: BLE001 - purely cosmetic, never fatal`. 6 of the 10 are in modules with zero importers anyway. |
| `except_exception_pass` file_list_item_enhanced.py:223 | **retired** — a failed `stat()` under-counts a summary byte total; module is dead (0 importers). |
| `except_exception_pass` emoji_picker.py:115 (`save_recent_emoji`) | **retired** — documented (`:116` "Refusal before IO is harmless"), and the recents file is explicitly cosmetic. |
| `except_exception_pass` voice_input_widget × 6, template_selector × 4, splash_screen:437/541 | **retired** — dead modules (voice_input_widget, template_selector) / single `query_one` guards (splash_screen). |
| `except_exception_return` model_search_picker.py:285 | **retired** — not silent: it sets `self._load_errors[cache_key] = True`, which `_render_catalog_status`/`_render_matches` (`:650-658`) read to change the user-facing status line. |
| `plain_readback` settings_theme_editor.py:297/635/641/865/867 | **confirmed** — see the P2. `Tree.process_label` markup-parses every `str` label (verified probe). |
| `dotted_section_setting` splash_screen.py:217/:280, settings_splash_screen_viewer.py:65 | **confirmed**, but not as a *dotted-lookup* bug (those resolve since TASK-1771) — as the P1 cost (9 reads for one table) and the P3 duplication. |
| `token_est_len_div4` chunk_preview.py:98 | **retired** — it is a character cap, not a token estimate. Full reasoning in Retired. |
| `run_worker_coroutine_per_file` audio_troubleshooting_dialog.py | **confirmed — P0** (`:246`, sync callable, no `thread=True`). |
| `run_worker_coroutine_per_file` dictation_performance_widget.py | **confirmed** (`:172`, `:175`, same defect) but the module has zero importers → P3. |
| `run_worker_coroutine_per_file` llamacpp_snapshot_manager (5), model_search_picker, project_skills_import_modal, settings_advanced_config_panel (4), settings_internal_prompts_panel, settings_web_search_panel | **retired** — all pass a coroutine or an `async def`/`partial(async_fn,…)`; `llamacpp_snapshot_manager._refresh` even wraps its blocking load in `asyncio.to_thread` (`:171`). No sync work on the loop. |
| `try_import_guard` detailed_progress.py:328 (`import psutil` per tick) | **retired** — measured `psutil.Process()+memory_info()` = **0.015 ms/call**; the timer is 1 Hz; and the module is dead. |
| `try_import_guard` emoji_picker.py:24 (module-scope `emoji` import) | **retired** — the `except ImportError` falls back to a different symbol of the *same* package; `emoji` is a hard dependency of this module either way. |
| `try_import_guard` media_details_widget:344/522/579, model_search_picker:285, prune_safe_select:86, settings_theme_editor:856, template_selector:274/396, enhanced_file_picker:1852 | **retired** — none is an optional-dependency guard; each is a broad `except Exception` around a local call. `prune_safe_select._textual_version` (`:84-91`) is a documented diagnostics-only helper with `# pragma: no cover`. |
| `function_body_import_per_file` emoji_picker (8), enhanced_file_picker (5), media_details_widget (3), settings_theme_editor (2), audio_troubleshooting_dialog, document_generation_modal, settings_image_gen_panel, workspace_create_modal | **retired as a cost** — every target resolves (`ls`'d via import in the probes above); after first import each is a `sys.modules` lookup. One is a real D3: `emoji_picker.py:71` imports the **private** `config._get_effective_config_path` — reported as P3 with its 53-site count. `media_details_widget`'s three are `from ..DB…`/`from ..Event_Handlers…` cycle breakers in a dead module. |
| `inline_truncate` tool_message_widgets:101/177/185/195, enhanced_file_picker:2085, detailed_progress:241 | **confirmed** — P3 D4a above; `Utils/Utils.truncate_content` is a drop-in at limits 100/200. |
| `raw_1024x1024` file_extraction_dialog.py:132/135 | **confirmed** as one copy of the byte-size-formatter cluster (P3 D4b); module is dead. |
| `strftime` × 7 (activity_log:81/283/499, file_list_item_enhanced:63, llamacpp_snapshot_manager:47/364, status_widget:56) | **retired** — all are Python `datetime.strftime` display formats, not the SQLite `%f` case the brief excludes; 4 of the 7 are in dead modules. No UTC/local mix-up: `llamacpp_snapshot_manager:47,364` both format a local timestamp for display only. |
| `legacy_markers_per_file` AppFooterStatus, compact_model_bar, enhanced_file_picker (3) | **retired** — `grep -n "DEPRECATED|LEGACY|TODO|FIXME|XXX|HACK"` on all three returns only the word "legacy" inside explanatory docstrings (`enhanced_file_picker:459,1248,1292` — "legacy list/tuple filters"). No deprecation marker, no TODO. |
| `dup_shape`/`dup_verbatim` `_cancel`/`_close`/`_perform_safe_cancel`/`_not_now` clusters (5 rows touching this slice) | **retired** — template-hook pattern, evidence in Retired below. |
| `dup_verbatim` `_consume@modal_dismissal.py:137` vs `Console/console_terminal_workspace.py:397` | **retired** — a 2-line `event.stop(); event.prevent_default()` static method on two unrelated overlay widgets. Extracting it would cost more than it saves. |
| `dup_verbatim` `filter_func@enhanced_file_picker.py:424` vs `file_picker_dialog.py:46` | **retired** — `file_picker_dialog` has zero importers (dead module). The live copy is `_make_glob_filter`'s closure. |
| `dup_verbatim` `_key_source_line`/`_secret_placeholder` (video vs image gen panel) | **confirmed** — P3 D4b above. |
| `dup_shape` `update_chunks@chunk_preview.py:83`, `get_section@enhanced_sidebar.py:246`, `set_status@base_components.py:330`, `_set_status@model_search_picker.py:595`, `update_path@enhanced_file_picker.py:322`, `_apply_search_filter_after_debounce@enhanced_file_picker.py:748`, `_default_bookmarks@enhanced_file_picker.py:142` | **retired** — same *shape* (a 1-3 line setter/refresh), different types and different state. 3 of the 7 live in dead modules. No shared helper is extractable without inventing an interface. |
| `seed_name__set_status` settings_agents_panel:701, model_search_picker:595, audio_troubleshooting_dialog:432 | **retired** — three unrelated one-line `Static.update()` setters on three different widgets. |

## Verified-fine
- **No mutable class attributes** anywhere in the 66 files (AST scan of class-scope `List`/`Dict`/`Set` literals → 0), and **no `id()`-keyed caches**.
- **No `run_worker(exclusive=True)` without `group=`** in the slice (AST scan of all 21 `run_worker` calls).
- **No loguru + stdlib `logging` in the same file** anywhere in the slice.
- `password_dialog.py` never logs a secret — its single `logger.error` (`:319`) logs the exception only.
- `splash_screen.py`'s 20-100 Hz animation timer is already hardened: `_update_animation` (`:450-478`) skips while the screen is inactive, passes `layout=False` with a written justification (TASK-21595), and is pinned by `Tests/UI/test_timer_path_layout_cost.py` as a geometry A/B.
- `audio_troubleshooting_dialog._update_level_meter` (`:397-430`, a 10 Hz timer) guards `is_attached`/`screen.is_active` **and** wraps its body — the right shape; the P0 in the same file is a different method.
- `pausable_progress.py` intercepting `set_interval` is a deliberate, documented framework workaround (`:1-100`), not mixin-as-implementation.
- `recompose_capture_guard.py` and `prune_safe_select.py` are both narrow, documented Textual-internals guards with the root-cause task ids in the docstrings.
- `enhanced_file_picker`'s `_SUPPRESSED_BASE_HANDLERS` / `_get_dispatch_methods` is the correct fix for Textual's whole-MRO handler dispatch, and it is explained at all four sites.
- `Widgets/__init__.py` does not re-export anything — it only installs the Textual compat shims — so the dead-module census cannot be defeated by a package-level re-export.
- `Backup_Recovery.raw_participants._scope/_file/_mkdirs/_replace/_unlink` used from `settings_theme_editor.py` and `emoji_picker.py` looks like "private helpers across packages" but is the ADR-126 storage-admission API; **out of scope by ADR** (`backlog/decisions/126-*`).

## Retired
(see the Candidate-dispositions table; the four substantive retirements are written up as `### Retired — …` blocks above: the 23 `except Exception: pass`, `_confirm_single`'s unguarded lookup, `path_validation` in the picker, `chunk_preview`'s `// 4`, `_format_text_for_reading`'s cost, and the `_perform_safe_cancel` cluster.)

## Left UNVERIFIED
| claim | why not verified | literal command to run |
|---|---|---|
| Pressing **Troubleshoot** in the Dictation/STTS surface takes the app down (the P0's end-to-end effect). The mechanism is proven by execution and both push sites are grepped, but the button was not pressed. | The brief forbids running the app (it regenerates `css/tldw_cli_modular.tcss`). | Per `.claude/skills/verify/SKILL.md`: `tmux -L verify new-session -d -s w -x 200 -y 50 "cd $WT && $PY -m tldw_chatbook.app"` → navigate to STTS ▸ Dictation → click **Troubleshoot** → `tmux -L verify capture-pane -p -t w` and look for the Textual crash panel / `WorkerError: Request to run a non-async function as an async worker`. |
| A theme saved as `my [x]theme` cannot be re-opened from the Settings theme tree (the P2's end-to-end effect). Round-trip mangling is proven by probe; the click was not made. | Same. | `tmux -L verify …` → Settings ▸ Theme → type `my [x]theme` in the name field → Save → click the new row under "Your themes" → confirm nothing loads and no notification appears. |
| The ~59 ms splash config cost is visible as a startup delay. | Startup timing needs the app. | `cd $WT && source env.sh && $PY -X importtime -m tldw_chatbook.app` (or instrument `SplashScreen.__init__`); the per-call numbers above are already measured in isolation. |
| Whether deleting the 21 dead modules changes `css/tldw_cli_modular.tcss`. | Building the bundle means running the app / the build step, which the brief excludes. | `cd $WT && ./scripts/preflight.sh` after a trial deletion on a scratch branch. |
