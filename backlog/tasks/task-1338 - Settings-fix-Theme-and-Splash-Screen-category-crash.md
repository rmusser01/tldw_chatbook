---
id: TASK-1338
title: Settings fix Theme and Splash Screen category crash
status: Done
assignee: []
created_date: '2026-08-04 23:47'
updated_date: '2026-08-05 00:10'
labels:
  - settings
  - ux
  - crash
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UAT (headless Pilot, 120x35, repro 2/2 each) found selecting the Theme or Splash Screen category raises KeyError at settings_screen.py:5938 (_inspector_guidance covers only 9 of 19 categories); the whole Settings screen renders blank and never recovers. Appearance category copy directs users into the crash. Source: .impeccable/critique/2026-08-04T23-45-33Z__tldw-chatbook-ui-screens-settings-screen-py.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Selecting every one of the 19 categories at 120x35 and 80x24 never crashes or blanks the screen,Focus is restored after each category recompose,Pilot regression test covers all categories at both sizes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce with a Pilot regression test that visits all 19 SettingsCategoryId categories at 120x35 and 80x24, asserting the screen still renders content and focus is restored after each switch (fails today on theme/splash_screen via KeyError in _inspector_guidance).
2. Add THEME and SPLASH_SCREEN entries to the _inspector_guidance dict and switch the final lookup to guidance.get() with a generic fallback so no category can ever raise.
3. Restore focus after every category recompose (focus the selected category button, not conditional on click focus).
4. Run new test plus Tests/UI/test_settings_configuration_hub.py and related settings suites.
ADR required: no — routine bug fix, direct implementation of existing behavior contracts
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed the Settings Theme/Splash Screen category crash plus two latent defects the crash had been masking.

**Root causes & fixes:**
1. `KeyError` crash: `_inspector_guidance` covered only 9 of 19 `SettingsCategoryId` values and ended in `return guidance[category]`. Added entries for `THEME` and `SPLASH_SCREEN`, and switched the final lookup to `guidance.get()` with a generic fallback so no category can ever raise inside a recompose (an exception there blanks the whole screen until restart).
2. Focus loss (AC2): `_select_category`'s `call_after_refresh` restore raced the recompose it scheduled — the restore landed on the doomed widget and `app.focused` ended up `<none>`; the click handler also gated restore on `event.button.has_focus`. Now the handler always restores, and a category switch records a one-shot intent (`_pending_category_focus_value`) consumed by a `recompose()` override after the fresh children mount, focusing the selected category button.
3. Latent infinite recompose storm on the Theme category (masked by the KeyError — the category had never rendered successfully): mounting `SettingsThemeEditor` re-flagged `is_modified` True/False on every mount because programmatic loads (`_update_color_inputs`, `_update_dark_mode_switch`) deliver `Input.Changed`/`Switch.Changed` asynchronously; each toggle recomposed the screen (`theme_editor_modified` is `recompose=True`), remounting the editor and looping forever. `settings_theme_editor.py` now only marks modified when a value actually differs from the loaded theme data.

**Files changed:**
- `tldw_chatbook/UI/Screens/settings_screen.py` — guidance entries + `.get()` fallback; focus-intent restore via `recompose()` override; click handler always restores focus.
- `tldw_chatbook/Widgets/settings_theme_editor.py` — programmatic loads no longer count as modifications.
- `Tests/UI/test_settings_category_sweep.py` (new) — Pilot regression test visiting all 19 categories at 120x35 and 80x24; asserts the screen still renders content and focus lands on the selected category button after each switch. Failed pre-fix with the reported `KeyError: SettingsCategoryId.THEME`, passes post-fix.

**Verification:** new sweep test 2 passed; `test_settings_configuration_hub.py` 242 passed; `test_settings_theme_editor.py` + `test_settings_privacy_security.py` 13 passed; `test_screen_navigation.py` + usability smoke 59 passed; `test_destination_shells.py` + footer hints 110 passed (1 pre-existing `[library]` tooltip failure, verified failing at HEAD); remaining phase-1/phase-6 UI suites 15 passed with 2 failures also verified pre-existing at HEAD. Ruff findings on touched files are all pre-existing lines; files compile clean.

ADR: not required — routine bug fix implementing existing behavior contracts (noted in plan).
<!-- SECTION:NOTES:END -->
