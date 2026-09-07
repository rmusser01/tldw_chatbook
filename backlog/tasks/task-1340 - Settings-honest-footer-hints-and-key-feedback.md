---
id: TASK-1340
title: Settings honest footer hints and key feedback
status: Done
assignee: []
created_date: '2026-08-04 23:47'
updated_date: '2026-08-05 01:13'
labels:
  - settings
  - ux
  - keybindings
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Footer advertises s/r/t screen-wide but t works on only 5 of 19 categories (ADR-031 advertised-but-dead-key violation); s/r/t silently no-op whenever a text input has focus (settings_screen.py:9144, 9626, 9715) with zero feedback; footer hints collapse to ellipsis at <=100 cols while keys remain bound. Also: revert (r) discards all staged edits with no confirmation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Footer hint set reflects the active category and only advertises working keys
- [x] #2 Pressing s/r/t while a text field is focused shows a one-line notice instead of silence
- [x] #3 Revert of a dirty category asks for confirmation per ADR-031
- [x] #4 Critical bindings stay visible or discoverable at 80-100 col widths
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no — direct implementation of existing ADR-031 conventions (truthful footer hints, guarded destructive single-letter actions). ADR path: backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md (linked).
1. Per-category honest hints: add SETTINGS_TEST_ACTION_CATEGORIES frozenset (providers-models, appearance, storage, privacy-security, diagnostics — the 5 categories whose test action is implemented); derive the advertised hint set per active category (s/r only for GUIDED_SETTINGS_MUTATION_CATEGORIES, t only for test categories); re-register footer shortcuts via a watch_active_category watcher so category switches update hints (registration API already persists across recompose).
2. Swallowed-key feedback: in action_settings_save/revert/test_category, replace the silent 'return' when a text entry has focus with a one-line app.notify notice.
3. Revert confirmation: extract the revert body into _revert_category(category); when the category has unsaved changes, push the existing Widgets/confirmation_dialog.py ConfirmationDialog (chat_screen.py:12901 pattern) and revert only on confirm; clean category keeps the 'No Settings changes to revert.' path.
4. Narrow-width discoverability: add action_show_workbench_help override rendering the ACTIVE category's advertised shortcut set in WorkbenchHelpPanel (app.py delegates F1 to the screen when present), so bindings stay discoverable when the footer collapses to ellipsis at <=100 cols.
5. Tests (new file Tests/UI/test_settings_footer_hints.py): per-category hint honesty across all 19 categories, swallowed-key notice, revert confirm/cancel, F1 help content, narrow-width footer; update the two existing tests that asserted screen-wide s/r/t hints and instant revert.
6. Run new tests + test_settings_configuration_hub.py + test_screen_footer_hints.py + test_settings_category_sweep.py + test_destination_shells.py.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented all four ACs per ADR-031 (no new ADR required — direct implementation of decision 031's truthful-hint and destructive-key-guard rules; linked in the plan).

- Per-category honest hints: new `SETTINGS_TEST_ACTION_CATEGORIES` frozenset (providers-models, appearance, storage, privacy-security, diagnostics) and `SettingsScreen._category_footer_shortcuts(category)` deriving the advertised set — s/r only for `GUIDED_SETTINGS_MUTATION_CATEGORIES`, t only where a test action is implemented, nothing elsewhere. A `watch_active_category` watcher re-registers through the persisting `register_footer_shortcuts` API on every category switch; the dead screen-wide `SETTINGS_SHORTCUTS` constant was removed. Keys stay bound screen-wide (they still respond with guidance), but the footer teaches only working keys.
- Swallowed-key feedback: `_settings_shortcut_swallowed_by_text_entry(key)` shows a one-line `app.notify` ("Finish editing the focused field, then press s/r/t again.") instead of the silent `return` in the save/revert/test actions; the explicit button paths (`allow_text_entry_focus=True`) never emit it.
- Revert confirmation: revert body extracted to `_revert_category(category)`; a dirty category now pushes the existing `ConfirmationDialog` ("Discard all unsaved changes to {title}?", confirm="Discard changes", cancel="Keep editing") — same pattern as chat_screen's close-tab guard. Clean categories keep the "No Settings changes to revert." path; THEME/SPLASH_SCREEN keep the "use the editor's own buttons" path.
- Narrow-width discoverability: new `action_show_workbench_help` override (app.py delegates F1 to the screen when present, mirroring chat_screen) renders the ACTIVE category's truthful shortcut set in `WorkbenchHelpPanel`, so the bindings stay discoverable when the footer collapses to "…" at ≤100 cols. No footer-widget changes — the existing ellipsis degradation is the house precedent.

Files changed:
- `tldw_chatbook/UI/Screens/settings_screen.py` (frozenset, per-category hint helper, watcher, F1 handler, swallowed-key helper, revert confirm gate + `_revert_category` split, ConfirmationDialog import)
- `Tests/UI/test_settings_footer_hints.py` (new: pure 19-category mapping test + 5 Pilot tests)
- `Tests/UI/test_screen_footer_hints.py` (settings registration test updated for per-category hints)
- `Tests/UI/test_settings_configuration_hub.py` (6 revert tests now confirm the dialog; two also switched from transient status-copy assertions to state assertions — the status text is overwritten by a queued Input.Changed on the next pause, a pre-existing race, so the tests assert reverted values/draft state instead)

Verification: `Tests/UI/test_settings_footer_hints.py` (9), `test_screen_footer_hints.py` (9), `test_app_footer_shortcut_context.py`, appearance/library-rag/theme/model-catalog/privacy/splash settings suites (71 total), `test_settings_configuration_hub.py` + `test_settings_category_sweep.py` (244) all pass; `test_destination_shells.py` 101 passed with only the known pre-existing failure `test_destination_action_buttons_explain_their_outcome[library]`. Ruff check/format clean on all touched code (remaining findings in settings_screen.py are pre-existing at HEAD).

Refinement (review follow-up): the honesty invariant is "advertised ⊆ bound, advertised == working in the active context" — keys stay bound screen-wide to provide guidance, but only working keys are advertised per category (recorded in ADR-031). Review hardening added: a capability-probe drift guard (probes the real s/t branches per category in clean state and asserts stub-vs-real matches the frozensets; mutation-verified red when a frozenset entry is removed), toast dedup for repeated swallowed-key presses (time-windowed, catches synchronous hammering), a stacked-dialog guard in `_confirmed_revert` (re-checks dirty state so double confirms can't double the revert toast), and a lazy-import rationale comment on the F1 help import.

Follow-up (task-1367): the swallowed-key notice machinery was unreachable by real keyboards (Textual `Input` consumes printable keys, so s/r/t never reach the screen bindings while a field is focused — typing into the field IS the feedback); `_settings_shortcut_swallowed_by_text_entry`, its dedup state and the `allow_text_entry_focus` escape hatch were removed, and the regression tests now drive real `pilot.press` sequences documenting actual field behavior.
<!-- SECTION:NOTES:END -->
<!-- SECTION:NOTES:END -->

<!-- SECTION:NOTES:END -->
