---
id: TASK-80
title: Functionalize Settings Appearance defaults
status: Done
labels:
- settings
- appearance
- configuration
- ux
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Turn Settings > Appearance from a routed/read-only summary into a guided configuration category for persisted visual defaults while preserving Customize as the deeper theme editor.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Settings > Appearance loads current persisted theme, palette limit, font size, density, animation, and scrolling defaults from config.
- [x] #2 Users can edit, validate, preview/apply where safe, save, and revert Appearance defaults from the Settings category.
- [x] #3 Invalid values block save, keep the category dirty, and show visible recovery copy in the detail pane and inspector.
- [x] #4 Settings copy clearly distinguishes global appearance defaults from the dedicated Customize theme editor and runtime-only preview behavior.
- [x] #5 Focused automated tests cover load, edit, validation failure, save, revert, ownership copy, focus readability, and route-to-Customize behavior.
- [x] #6 Actual Textual-web/CDP screenshots verify baseline, dropdown visibility, focused input readability, and validation recovery before PR creation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Uses existing persisted UI/general/web_server/appearance config keys and preserves the Settings/Customize boundary; no new schema, runtime boundary, service contract, security policy, dependency, or long-lived application structure is introduced.

Implementation plan:
1. Add pure Appearance defaults tests and helper module for load, validation, and save payloads.
2. Render Settings > Appearance as guided controls while keeping Open Customize as the full editor route.
3. Wire Appearance draft, validation, save, revert, and preview/apply using the existing Settings draft and worker patterns.
4. Add focused mounted tests for dirty state, validation recovery, save/revert, focus readability, and route-to-Customize behavior.
5. Run focused pytest verification plus git diff checks.
6. Capture actual Textual-web/CDP screenshots for baseline, dropdown visibility, focused input readability, and invalid recovery before PR creation.

Plan document: Docs/superpowers/plans/2026-06-07-settings-appearance-defaults.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added a focused Appearance defaults helper for loading, normalizing, validating, and building save payloads for `general.default_theme`, `general.palette_theme_limit`, `web_server.font_size`, and `appearance.*` defaults.
- Converted Settings > Appearance from a read-only route summary into a guided settings category with editable controls, dirty/invalid/saved/reverted state copy, runtime-safe preview, Save/Revert handling, and a preserved `Open Customize` action for full theme editing.
- Implemented the Appearance save path with an exclusive Textual background worker so config writes stay off the UI thread and UI mutations return through the app thread.
- Added automated coverage for pure helper behavior, mounted Settings behavior, validation recovery, save/revert, focus readability, route-to-Customize compatibility, and CDP screenshot-gate copy.
- Captured and received approval for actual Textual-web/CDP screenshots: `settings-appearance-baseline.png`, `settings-appearance-theme-dropdown.png`, `settings-appearance-focused-input.png`, and `settings-appearance-invalid-palette-limit.png`.
- Verification: `python -m pytest -q Tests/UI/test_settings_appearance_defaults.py Tests/UI/test_settings_configuration_hub.py --tb=short` passed with `193 passed, 1 warning`; `python -m pytest -q Tests/UI/test_destination_shells.py::test_settings_appearance_action_routes_to_customize_surface --tb=short` passed with `1 passed, 1 warning`; `git diff --check` returned clean.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Settings > Appearance is now a functional configuration hub category for persisted global visual defaults, while Customize remains the deeper theme editing surface.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
