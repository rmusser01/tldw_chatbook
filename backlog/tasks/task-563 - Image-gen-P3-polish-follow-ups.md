---
id: TASK-563
title: Image-gen P3 polish follow-ups
status: Done
assignee: []
created_date: '2026-07-24 23:34'
updated_date: '2026-07-25 08:48'
labels:
  - image-generation
  - personas
  - followup
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred polish from the image-gen P3 whole-branch review (PR #859: ✨Generate for character avatar + expression slots; spec `Docs/superpowers/specs/2026-07-24-image-gen-p3-expression-generation-design.md`). None are defects in shipped behavior — the High/Medium findings were fixed pre-merge. Distinct from [[task-497]]/[[task-558]]/[[task-559]] (P1/P2a/P2b polish) and the pre-existing test failures ([[task-564]]).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Expressions action row remains usable at narrow terminal widths (at 120 cols Import/Export currently overflow the row; at 80 cols Generate-all does too — pre-existing since the row shipped). Wrap, scroll, or fold the set-actions into a menu; pin with a width-parameterized test replacing the current 200-col-only claim.
- [x] #2 A generation in progress shows an in-slot "Generating…" affordance (spec §1 promised it; today the only feedback is the completion/failure notify or the "already generating" refusal on a second click).
- [x] #3 "✨ Generate all" asks for confirmation when it would overwrite existing images (staged avatar or populated expression slots) — the sweep's blast radius exceeds the per-slot regenerate-by-click contract.
- [x] #4 The Generate-all summary counts only genuinely persisted slots (today `_apply_expression_upload` swallows its own DB-write failure and the sweep counts that slot as a success — user sees both the per-slot error AND an inflated "k/4").
- [x] #5 Cosmetics: a one-line comment at `_after_character_save`'s record-reread-failure fallback noting the style-reset invariant holds via the closed editor gates; the generate-all narrow race (per-slot key freed mid-loop allowing a duplicate regeneration) either guarded or documented as accepted last-write-wins.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. AC1 (narrow-width row): add overflow-x: auto; scrollbar-size-horizontal: 0; to .personas-char-editor-expr-set-row (mirrors MainNavigationBar's .main-nav idiom, tldw_chatbook/UI/Navigation/main_navigation.py:94) - the row becomes horizontally scrollable instead of hard-clipping past its right edge, so every button stays reachable (scroll/keyboard-tab) at any width. Replace test_expressions_header_does_not_push_siblings_off_row's 200-col-only claim with a width-parameterized test (80/120/200) that scroll_visible()s + pilot.click()s each action button and asserts the correct message posts - proving reachable AND functional, not just positioned. This is DEFAULT_CSS inside the widget's Python file, not the generated tldw_cli_modular.tcss bundle, so no build_css.sh run is needed.
2. AC2 (in-slot Generating... affordance): add PersonasCharacterEditorWidget.set_expression_generating(state, busy) (overwrites the per-state -hint Static) and set_avatar_generating(busy) (overwrites avatar-status Static), both self-healing on clear (recompute the real default rather than remembering one). Screen sets busy=True at the same three sites that already add an in-flight key (_handle_character_expression_generate_requested, _handle_character_avatar_generate_requested, and inside _generate_all_expression_images_worker's loop). Clear busy=False from one place - _generate_one_slot's existing finally - via a new _clear_slot_generating_indicator(character_id, state) helper that re-checks _character_editor_is_active() + editor.expression_character_id() == character_id first (mirrors the sweep's own per-iteration identity guard) so a mid-generation character switch never paints a stale "Generating..." onto whatever now-different character is loaded - the switch's own _sync_expression_slots_enabled/_set_avatar_status_from_record already reset the widget for real.
3. AC3 (Generate-all overwrite confirmation): add editor.has_avatar_image() (factored out of _set_avatar_status_from_record) and a screen helper that also checks the 3 DB-backed expression states via db.get_character_expression_image (mirrors _render_character_expression_slot's read). When any exist, _generate_all_expression_images_worker awaits a new _confirm_generate_all_overwrite() (same ConfirmationDialog/push_screen_wait idiom as _confirm_delete/_confirm_dictionary_revert) before the loop; Cancel aborts silently (no notify, no writes, in-flight key still released via the existing finally). No dialog when everything is empty.
4. AC4 (honest k/N summary): _apply_expression_upload returns bool (True on persisted write, False on the already-notified DB-write failure); _generate_one_slot's expression branch returns that value instead of a hardcoded True, so _generate_all_expression_images_worker's succeeded counter (already `if await self._generate_one_slot(...): succeeded += 1`) only counts genuinely persisted slots.
5. AC5 (cosmetics): one-line comment at _after_character_save's saved_record is None fallback explaining why leaving _expression_generate_style un-reset there is still safe (editor closes to view mode; the next genuine session-open resets it before any button is reachable). For the generate-all narrow race, GUARD it: _handle_character_expression_generate_requested and _handle_character_avatar_generate_requested additionally refuse when (character_id, "all") is in-flight, closing the window where a per-slot key freed mid-sweep let an independent single-slot click regenerate a slot the sweep already (or is about to) touch - documented as the chosen approach (not last-write-wins) in Implementation Notes.
6. TDD per AC: write/adjust failing tests first in Tests/UI/test_personas_expression_generate.py (screen/worker level) and Tests/UI/test_personas_expression_slots.py or a new widget test as needed, confirm red, implement, confirm green. Run ruff + import-clean check. Update task file ACs/Notes, commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
All 5 ACs implemented with TDD (failing test confirmed red, then implementation, then green) in `tldw_chatbook/UI/Screens/personas_screen.py` and `tldw_chatbook/Widgets/Persona_Widgets/personas_character_editor_widget.py`.

**AC1 (narrow-width row).** Added `overflow-x: auto; scrollbar-size-horizontal: 0;` to `.personas-char-editor-expr-set-row`'s `DEFAULT_CSS` (widget-scoped Python CSS, not the generated `tldw_cli_modular.tcss` bundle — no `build_css.sh` needed), mirroring `MainNavigationBar`'s `.main-nav` idiom. The row becomes horizontally scrollable instead of hard-clipping. Replaced the 200-col-only `test_expressions_header_does_not_push_siblings_off_row` with `test_expr_set_row_buttons_reachable_and_functional_at_width` parametrized over 80/120/200 cols. Discovered mid-implementation that `Pilot.click()` dispatches successfully even at an off-screen region (headless-mode leniency a real terminal can't replicate), so the test instead asserts the button's region falls entirely within the viewport after `scroll_visible()` (the assertion that actually discriminates the fix — verified red at width=80 without the CSS change, green with it) plus a separate `Button.press()` functional check.

**AC2 (in-slot "Generating…" affordance).** Added `PersonasCharacterEditorWidget.set_expression_generating(state, busy)` and `set_avatar_generating(busy)`, both self-healing on clear (recompute the real default text rather than remembering one). Screen sets busy=True at the three dispatch sites that already add an in-flight key; clears via one path — `_generate_one_slot`'s existing `finally` — through a new `_clear_slot_generating_indicator(character_id, state)` that re-checks `_character_editor_is_active()` + `editor.expression_character_id() == character_id` before touching the widget, so a mid-generation character switch never paints a stale indicator onto a different now-loaded character. Pinned with a dedicated leak test simulating a switch to a character with its own independent "thinking" generation in flight, proving the stale clear does not clobber it.

**AC3 (Generate-all overwrite confirmation).** Added `editor.has_avatar_image()` (factored out of `_set_avatar_status_from_record`) and `_generate_all_would_overwrite()` (avatar + the 3 DB-backed expression states via `db.get_character_expression_image`, mirroring `_render_character_expression_slot`'s read). `_generate_all_expression_images_worker` now confirms via `_confirm_generate_all_overwrite()` (same `ConfirmationDialog`/`push_screen_wait` idiom as `_confirm_delete`/`_confirm_dictionary_revert`) before the loop, only when something would actually be overwritten; a decline aborts with no writes and no summary notify (a "0/4 generated" notify on an explicit decline would misleadingly read as an under-performing attempt rather than a withdrawn request).

**AC4 (honest k/N summary).** `_apply_expression_upload` now returns `bool` (`True` on a persisted write, `False` on the already-notified DB-write failure); `_generate_one_slot`'s expression branch returns that value instead of a hardcoded `True`. `_generate_all_expression_images_worker`'s existing `if await self._generate_one_slot(...): succeeded += 1` now only counts genuinely persisted slots. Pinned with a test that exercises the real (unmocked) `_apply_expression_upload` with a flaky DB write for one state, proving the summary reads "3/4" (not "4/4") while the per-slot error notify still fires.

**AC5 (cosmetics + narrow race).** Added a one-line comment at `_after_character_save`'s `saved_record is None` fallback explaining the style-reset invariant still holds (editor closes to view mode; the next genuine session-open resets the style before any button is reachable). Chose to **guard** (not document as accepted) the generate-all narrow race: `_handle_character_expression_generate_requested` and `_handle_character_avatar_generate_requested` now also refuse when `(character_id, "all")` is in-flight, closing the window where a per-slot key freed mid-sweep let an independent single-slot click race a redundant regeneration against the still-running sweep. This is simpler and stronger than reasoning about per-slot timing windows, since the "all" key's lifetime spans the sweep's entire duration.

**Testing:** `Tests/UI/test_personas_expression_generate.py` (net +~430 lines; the old 200-col-only test was replaced by a 3x-parametrized one) and `Tests/UI/test_personas_expression_slots.py` (+2 tests). Full targeted suite: 69 passed (was 52 baseline). Broader regression sweep `Tests/UI/ -k personas` (all 26 persona-related files, 620+ tests): 3 pre-existing failures reproduced identically on the base commit (`test_import_offpage_name_conflict_message`, `test_character_book_errors_render_in_editor_footer`, `test_export_json_rejects_hidden_directory_destination` — task-564's scope, not touched here), everything else green. `ruff check` on touched files shows only pre-existing findings at unchanged line positions (verified against `git show HEAD`). `python -c "import tldw_chatbook.app"` clean.

**Files modified:** `tldw_chatbook/UI/Screens/personas_screen.py`, `tldw_chatbook/Widgets/Persona_Widgets/personas_character_editor_widget.py`, `Tests/UI/test_personas_expression_generate.py`, `Tests/UI/test_personas_expression_slots.py`.
<!-- SECTION:NOTES:END -->
