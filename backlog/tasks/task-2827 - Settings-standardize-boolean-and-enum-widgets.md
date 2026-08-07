---
id: TASK-2827
title: Settings standardize boolean and enum widgets
status: Done
assignee: []
created_date: '2026-08-04 23:47'
updated_date: '2026-08-05 06:38'
labels:
  - settings
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Booleans appear as text Inputs ('true or false' placeholder, settings_screen.py:6618-6627), Enabled/Disabled Buttons (:7145-7158), and Checkboxes (:6269-6303) in different panes; closed enums (reasoning effort, verbosity) accept typed strings validated only at save. Wrong widget idioms invite invalid input.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All booleans use one toggle idiom (Checkbox)
- [x] #2 Closed enums use Select widgets
- [x] #3 Free-text validation remains only for genuinely open values
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: UI-widget idiom change only (Checkbox/Select replacing Inputs/Buttons); no storage, schema, sync, or contract change. Staged-save commit model (task-1341) is preserved.

1. Pick one toggle idiom: Checkbox (majority across UI/ 76 vs 38 Switch; already used in this screen's model-catalog group).
2. Update tests first: booleans asserted as Checkbox.value, closed enums as Select values, invalid-typing rejection tests removed/replaced with by-construction assertions.
3. Booleans -> Checkbox: console-default streaming, paste-collapse toggle, background-effect enabled, library-rag include-citations, appearance animations/smooth-scrolling.
4. Closed enums -> Select (allow_blank, blank=inherit): console reasoning_effort/reasoning_summary/verbosity/thinking_effort + the four model-profile enum fields; tri-state model-profile streaming -> Select(Inherit/On/Off).
5. Rewire handlers (Checkbox.Changed/Select.Changed), staging (unchanged staged semantics), provider form-values save path, provider/console/appearance/library sync + revert paths; clamp legacy invalid config strings to Select.NULL.
6. Update inspector Validation-row copy for changed fields; remove Enabled/Disabled label helpers; run suites.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Standardized Settings boolean and closed-enum widgets in `tldw_chatbook/UI/Screens/settings_screen.py`.

- Toggle idiom decision: **Checkbox** — majority precedent across `UI/` (76 vs 38 `Switch(` uses) and already the idiom inside this screen (ADR-020 model-catalog group, left untouched).
- Booleans converted to Checkbox: console-default streaming (was "true or false" Input), paste-collapse toggle, console background-effect enabled, library-rag include-citations, appearance animations/smooth-scrolling (latter five were Enabled/Disabled Buttons). Handlers moved to `Checkbox.Changed` and keep staged-save semantics (task-1341); new `_syncing_console_paste_toggle` guard mirrors the other sync guards.
- Closed enums converted to `Select` (`allow_blank`, blank = inherit/provider default): console-default reasoning_effort/reasoning_summary/verbosity/thinking_effort and the four gated model-profile enum fields. Ordered option lists (`REASONING_EFFORT_SELECT_OPTIONS` etc., unified in `CLOSED_ENUM_SELECT_OPTIONS`) are shared by both panes; invalid input is now impossible by construction.
- Tri-state `model_profile_streaming` (inherit/on/off) is a closed 3-value enum, so it renders as `Select(Inherit default/On/Off)` instead of a Checkbox, which cannot represent inherit.
- Save/staging paths updated: `_provider_form_values_from_widgets` reads Selects; provider/console/appearance/library-rag `_sync_*` and provider revert map `""` <-> `Select.NULL` and clamp legacy invalid config strings to blank (avoids `InvalidSelectValueError`). Save-time normalisers retained as boundary defense.
- Inspector Validation-row copy updated for the changed fields; removed now-unused Enabled/Disabled label helpers (`_appearance_bool_label`, `_console_background_effect_enabled_label`, `_library_rag_include_citations_label`, `_collapse_large_pastes_button_label`); removed the paste-toggle Button from the `on_key` Enter special-case (Checkbox toggles natively).
- Free-text Inputs remain only for genuinely open values (numbers, seeds, endpoints, model names, stale-hours, storage paths).

Files changed: `tldw_chatbook/UI/Screens/settings_screen.py`, `Tests/UI/test_settings_configuration_hub.py`, `Tests/UI/test_destination_shells.py`.

Follow-up (spec review): AC1 now holds literally — the four remaining Switch booleans inside the Settings screen were converted to the same Checkbox idiom: `settings-splash-enabled`, `settings-splash-show-progress`, `settings-splash-skip-on-keypress` (`Widgets/settings_splash_screen_viewer.py`) and `settings-theme-dark-mode` (`Widgets/settings_theme_editor.py`). Their save semantics are unchanged (splash defaults still persist on change per the instant-apply exception, task-1341; the theme editor toggle still only flags `is_modified`); only the widget idiom and the `Switch.Changed` -> `Checkbox.Changed` handler wiring changed. No CSS targeted these widget ids, so no stylesheet changes were needed. Additional files changed: the two widget modules and `Tests/UI/test_settings_splash_screen_viewer.py`.

Tests: updated hub/destination tests to the new widgets first; removed two invalid-typing rejection tests (impossible by construction) and added `test_settings_provider_streaming_and_enums_prevent_invalid_input` asserting the constrained option sets. Results: `test_settings_configuration_hub.py` 245 passed; `test_settings_save_commit_models.py` 7 passed; sweep/footer-hints/narrow 20 passed; destination suites 171 passed with 14 failures reproduced against HEAD copies of these files (pre-existing: the known `...[library]` failure plus watchlists/schedules failures from other in-flight task work — unrelated to this change). Ruff clean except two pre-existing F401/F811 on HEAD.
<!-- SECTION:NOTES:END -->
