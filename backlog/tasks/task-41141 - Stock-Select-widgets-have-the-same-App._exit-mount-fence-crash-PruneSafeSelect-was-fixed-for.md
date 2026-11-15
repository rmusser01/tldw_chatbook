---
id: TASK-41141
title: "Stock Select widgets have the same App._exit mount-fence crash PruneSafeSelect was fixed for"
status: Done
created_date: 2026-11-15 02:20
updated_date: 2026-11-15 19:29
assignee:
- '@claude'
labels:
- follow-up
- ui
- textual
- crash
priority: low
references:
- tldw_chatbook/Widgets/prune_safe_select.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** TASK-40323's review found a crash: once `App._exit` is set (`App.exit`, and first thing in `TldwCli._shutdown`), `Widget.mount_all` mounts nothing, so a `Select` composed then has no `SelectOverlay`, but `Select._on_mount` and `set_options` still rebuild it: `NoMatches: 'SelectOverlay'`. It fixed `PruneSafeSelect` only (`prune_safe_select.py:130-170`, `_retired`).

**Evidence.** About 70-110 modules under `tldw_chatbook/` import or construct a stock Textual `Select` (72 files both import `Select` and call `Select(`). Any of them that composes or updates a Select during shutdown can raise the same error.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A test composes a stock Select after App._exit is set and records whether it raises
- [x] #2 If it does, the app's Selects get the same exit gate (for example by using PruneSafeSelect or one shared subclass), with the test red before the change
- [x] #3 The notes list which Selects can compose or update during shutdown
<!-- AC:END -->

## Implementation Plan

1. Write a mounted test that composes a stock `Select` after `App._exit` is set (the in-flight mount shape TASK-40323 used) and record whether it raises.
2. If it raises, gate the two methods that reach the missing children (`_setup_options_renderables`, `_watch_value`) on stock `Select` itself, once, from the one place every app Select routes through, keeping PruneSafeSelect's prune/close half where it is.
3. Keep the import-weight and size ratchets honest: install at app construction (no new module at `import tldw_chatbook.app`), raise the app.py row by the lines added.
4. List in the notes which Selects can compose or update during shutdown.

## Implementation Notes

- **AC#1, recorded: it raises.** `Tests/UI/test_console_archive_recovery_cards.py::test_a_stock_select_composed_after_app_exit_began_does_not_crash` mounts a stock `Select` whose `compose` runs as `App._exit` is set (past the fence `mount` refuses with `MountError`, so an in-flight mount is the only way a Select composes then). Without the gate: `NoMatches: No nodes match 'SelectOverlay' on Select(id='stock')` from `_on_mount`, as `app._exception`. The test then calls `set_options` and sets `value`, the two later update paths.
- **AC#2, one gate for every Select.** `install_select_exit_gate()` (`tldw_chatbook/Widgets/prune_safe_select.py`) wraps `Select._setup_options_renderables` and `Select._watch_value` on the stock class: past `App._exit` the first does nothing and the second keeps only `self._value = value`, exactly PruneSafeSelect's exit half. Idempotent (a marker on the wrapper). PruneSafeSelect keeps its `_pruning`/`_closing` half (its `test_stock_select_still_raises_*` controls pin that stock `Select` still raises under a prune), and its `_retired` now reuses the shared `_past_exit_fence`. `TldwCli.__init__` installs the gate (2 lines), so every Select in the app, stock or subclass, routes through it. Installing at `import tldw_chatbook.app` instead was tried first: it adds `Widgets.prune_safe_select` to the import (687 -> 688 modules against `test_app_import_weight.py`'s `< 688` ratchet), while the module is already resident at UI-ready (Console's control bar), so construction time costs nothing new. Red/green: the test above fails with the app.py change reverted (`assert NoMatches(...) is None`) and passes with it (it builds the app through `Tests/UI/app_factory._build_test_app`).
- **AC#3, which Selects can compose or update during shutdown.** `TldwCli._shutdown` sets `_exit` first, then awaits the app-owned drains (Console runtime, Buddy, notes sync, actor packs...) for seconds while message pumps still run. In that window `mount` refuses and `recompose`/`mount_all` mount nothing, so the Selects at risk are those whose mount was already in flight: a screen or modal opened just before quit, and any subtree a worker, timer or message handler was rebuilding. By that rule (a census over the 98 modules that construct a Select, grep):
  - rebuilt by a recompose at runtime (15): `UI/Evals/bench_editor.py`, `UI/Evals/results_grid.py`, `UI/MCP_Modules/mcp_rail.py`, `UI/Screens/model_remote_view.py`, `UI/Screens/settings_hooks.py`, `UI/Screens/settings_screen.py`, `UI/Settings_Modules/providers_models_card.py`, `Widgets/Library/library_ingest_canvas.py`, `Widgets/Library/library_prompts_canvas.py`, `Widgets/Settings_Widgets/personal_context_panel.py`, `Widgets/Settings_Widgets/personal_context_review_modal.py`, `Widgets/Settings_Widgets/speech_tts_settings_panel.py`, `Widgets/TTS/chapter_editor_widget.py`, `Widgets/TTS/character_voice_widget.py`, `Widgets/workspace_persona_default.py`;
  - mounted dynamically (28): `Third_Party/textual_fspicker/base_dialog.py`, `UI/Dictation_Window_Improved.py`, `UI/LLM_Management_Window.py`, `UI/MCP_Modules/mcp_inspector.py`, `UI/MCP_Modules/mcp_permissions_mode.py`, `UI/Screens/backup_restore_screen.py`, `UI/Screens/change_review_screen.py`, `UI/Screens/meetings_screen.py`, `UI/STTS_Window.py`, `UI/Study_Window.py`, `Widgets/audio_troubleshooting_dialog.py`, `Widgets/backup_group_selector.py`, `Widgets/Console/console_prompt_mixins_view.py`, `Widgets/Console/console_prompts_browse.py`, `Widgets/Console/console_scope_picker_modal.py`, `Widgets/enhanced_file_picker.py`, `Widgets/form_components.py`, `Widgets/NewIngest/UnifiedProcessor.py`, `Widgets/Persona_Widgets/buddy_character_review.py`, `persona_profile_editor_widget.py`, `personas_persona_visual_pack_widget.py`, `personas_preview_pane.py`, `petdex_import_review.py`, `portrait_performance_editor.py`, `Widgets/Prompts/prompt_block_editor.py`, `Widgets/settings_splash_screen_viewer.py`, `Widgets/settings_web_search_panel.py`, `Widgets/voice_blend_dialog.py`;
  - the other 55 compose their Selects only in a screen's or modal's own `compose`, at risk only when quit lands while that screen opens;
  - already covered by PruneSafeSelect before this task (13 modules): the Watchlists panes and modals, `watchlists_collections_screen.py`, the Speech settings pane and axis row, `compact_model_bar.py`, `archive_recovery_cards.py`.
  An overlay-bearing Select (composed before `_exit`) updates safely; only a Select that composed past the fence lacks its children, and the gate covers both of its update paths.
- **Runs** (run.sh, load ~100-115 on 18 CPUs): `Tests/Widgets/test_prune_safe_select.py`, `Tests/UI/test_console_archive_recovery_cards.py`, `Tests/Performance/test_app_import_weight.py` at `-n 4`: 66 passed, 1 failed: `test_app_import_stays_well_under_pre_fix_baseline`, the 8.0 s wall-clock tripwire (`import tldw_chatbook.app took 10.42s`, load ~103; this change adds no import at module import). `Tests/Architecture/test_module_size_ratchet.py`: 207 passed (app.py row 5705 -> 5707 with a dated comment).
- Files: `tldw_chatbook/Widgets/prune_safe_select.py`, `tldw_chatbook/app.py`, `Tests/UI/test_console_archive_recovery_cards.py`, `Tests/Architecture/test_module_size_ratchet.py`.
