---
id: TASK-15479
title: 'Character voice widget: reactive self-assignment can never fire'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-11 12:05'
labels:
  - bug
  - tts
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the latency audit: `Widgets/TTS/character_voice_widget.py:455/:465` uses `self.characters = self.characters` to trigger a refresh after add/remove — but assigning the same object compares equal and the reactive has no `always_update`, so the watcher never runs and the table never refreshes.

Fix direction: `mutate_reactive`, assign a new list, or `always_update=True`; add the missing refresh test; grep for other instances of the dead pattern. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Adding and removing a character updates the table (test)
- [x] #2 No other self-assignment reactive triggers remain (grep evidence in notes)
<!-- AC:END -->

## Implementation Plan

1. Read `Widgets/TTS/character_voice_widget.py` in full to confirm both cited sites (`_add_character_manually`, `_remove_selected_character`) and check for a third instance nearby (`_reset_all_voices`).
2. TDD: write tests mounting `CharacterVoiceWidget` in a minimal `App`, calling the add/remove handlers, asserting the `#character-table` `DataTable`'s live row count/content — confirm red on unmodified code.
3. Fix the triggering bug using the codebase's established `mutate_reactive` idiom (grepped precedent in `settings_screen.py` / `speech_catalog_mixin.py`), applied at all matching call sites in this file.
4. Repo-wide grep for the exact `self\.(\w+) = self\.\1` shape (refined to bare self-assignment only, excluding `self.x = self.x.method()`/slicing/swap-idiom false positives) and classify every hit.
5. Re-run tests; investigate any surprising failures rather than papering over them (this surfaced two extra pre-existing issues, handled below).
6. Update backlog task and write the report.

## Implementation Notes

**Fixed** (`tldw_chatbook/Widgets/TTS/character_voice_widget.py`): both cited self-assignment sites (`_add_character_manually` line ~469, `_remove_selected_character` line ~486) now call `self.mutate_reactive(CharacterVoiceWidget.characters)` instead of `self.characters = self.characters`. This is the codebase's existing idiom for forcing a mutable reactive's watcher to run (precedent: `UI/Screens/settings_screen.py:_refresh_settings_workspaces_pane`, `UI/Speech/speech_catalog_mixin.py`), and matches Textual's own documented use of `mutate_reactive`.

**TDD evidence**: `Tests/Widgets/test_character_voice_widget.py` (new) mounts the widget in a minimal `App`, calls the private add/remove handlers, and asserts on the live, freshly-queried `#character-table` `DataTable`. All 3 tests were confirmed red against the unmodified widget (`assert table.row_count == 1` failed with `0`, etc.) before the fix, and green after.

**Second bug found via TDD, fixed in the same commit (in scope — same file, directly entangled with the trigger fix)**: `characters = reactive([], recompose=True)`. `compose()` never reads `self.characters` — the `DataTable` is built empty (columns only) and populated *imperatively* by `_refresh_character_table()`, called from `watch_characters`. `recompose=True` was a silent no-op only because the self-assignment bug meant the reactive never actually fired. Once fixed to fire (via `mutate_reactive`), `recompose=True` tore down and rebuilt the whole widget subtree on every add/remove: the freshly-added row was there for one instant, then discarded when the recompose swapped in a brand-new (columns-only, zero-row) `DataTable`, with nothing to repopulate it afterward. Confirmed live with a throwaway repro script: after settling, a **fresh** query of `#character-table` reported `row_count == 0` even though `widget.characters` correctly held the added item. Removed `recompose=True` (the other 3 reactives on this widget — `selected_character_index`, `voice_assignments`, `provider` — already have no such flag, and none of them are read from `compose()` either, so this makes `characters` consistent with its siblings). This was necessary for AC #1 to be genuinely true (not just true for a stale/orphaned test reference) — see comment block above the reactive declaration for the full reasoning, and the docstring of the new test file.

**AC #2 grep sweep**: exact-shape regex for a bare self-assignment reactive trigger, `^\s*self\.([A-Za-z_]\w*)\s*=\s*self\.\1\s*(#.*)?$` (anchored at both ends so `self.x = self.x.method()`, slicing `self.x = self.x[:-10]`, and the `a, self._b = self._b, None` swap idiom are correctly excluded as **not** the dead-trigger shape):
```
grep -rnP '^\s*self\.([A-Za-z_][A-Za-z0-9_]*)\s*=\s*self\.\1\s*(#.*)?$' --include='*.py' tldw_chatbook/
```
Before the fix: exactly 3 hits, all 3 in this file (`characters` ×2, `voice_assignments` ×1). After the fix: **0 hits repo-wide**. A broader, unanchored version of the same regex (allowing anything after the repeated name, e.g. `self.x = self.x.toggle(...)`) turned up ~50 more matches across the tree (`lab_frame.py`, `video_player_screen.py`, `dictation.py`, `code_audit_tool.py`, `mindmap_model.py`, `Embeddings_Lib.py`, `reply_sentence_sequencer.py`, `console_voice_input.py`, `swarmui_client.py`, `evaluation_state.py`, `streaming_sink.py`, `transcription_service.py`, `dictation_service_lazy.py`, `directory_navigation.py`, `world_info_processor.py`, `chatterbox_isolated.py`, `TTS/utils/performance.py`, `status_widget.py`, `file_list_item_enhanced.py`, `enhanced_file_picker.py`, `detailed_progress.py`, `console_prompts_modal.py`) — every one of these was manually inspected and classified as **not** the dead-trigger shape: each RHS calls a method, slices, concatenates, or is a swap-tuple-unpack (`old, self._x = self._x, None`), all of which produce a genuinely new object rather than the identical one, so they legitimately change the reactive/attribute's value (or are plain, non-`reactive()` attributes with no watcher to fail to trigger in the first place). None were touched.

**Also fixed the identical one-line pattern found in the same file** (`_reset_all_voices`, `self.voice_assignments = self.voice_assignments`) → `self.mutate_reactive(CharacterVoiceWidget.voice_assignments)`. Note for the record: `voice_assignments` has no `watch_voice_assignments` method anywhere in the class, so this line was already a pure no-op before *and* after the fix in terms of observable behavior (the explicit `_refresh_character_table()` / `_update_assignment_summary()` calls immediately below are what actually keep `_reset_all_voices`'s UI in sync). Fixed anyway for consistency with the identical pattern and to keep the reactive honest if a watcher is ever added later — this did not require a new test since there is currently no watcher-driven behavior it could regress.

**Found but explicitly out of scope, flagging for a follow-up task**: `characters = reactive([])`'s default value is a bare list literal. Textual's `Reactive._initialize_reactive` uses `default_or_callable() if callable(...) else default_or_callable` — since `[]` is not callable, the *exact same list object* is installed as `_reactive_characters` on every `CharacterVoiceWidget` instance that hasn't explicitly reassigned it yet (the classic mutable-default-argument trap, applied to a Textual reactive default). This is independent of the self-assignment trigger bug and was **not** introduced by this fix (the aliasing existed before, under the old `recompose=True` declaration too) — it was only discovered because it made the new tests order-dependent within one pytest session (an earlier test's `.append()` leaked into a later test's row count via the shared default). Production impact: any two `CharacterVoiceWidget` instances that both start from the un-set default and never reassign `characters` to a fresh list would share and cross-mutate the same underlying list. Not fixed here (outside this task's AC boundary); the new tests work around it defensively by assigning `widget.characters = []` (a fresh list) before use, which is documented in the test file's module docstring.

**Files modified**: `tldw_chatbook/Widgets/TTS/character_voice_widget.py` (reactive declaration + 3 call sites).
**Files added**: `Tests/Widgets/test_character_voice_widget.py` (3 tests: add refreshes table, two adds each refresh, remove refreshes table).

**Tests run**: `Tests/Widgets/test_character_voice_widget.py` (3/3 passed, confirmed red pre-fix), `Tests/UI/test_stts_profile_library.py` (73/73 passed — the only other test file referencing `character_voice_widget`, via an import-isolation stub, unaffected), `Tests/Widgets/` full directory (376/377 passed; the 1 failure, `test_library_collections_panel.py::test_library_collections_panel_empty_state_renders_message_once`, is unrelated to this widget/change — reproduces in isolation on unmodified files, pre-existing on `dev`), `Tests/TTS/test_tts_profile_capabilities.py` (40/43 passed; the 3 failures are the standing pre-existing Protocol-`isinstance` failures called out in the environment brief, untouched by this change).
