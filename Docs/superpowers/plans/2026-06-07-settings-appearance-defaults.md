# Settings Appearance Defaults Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `Settings > Appearance` into a guided editor for persisted visual defaults while preserving Customize as the deeper theme editor.

**Architecture:** Add a focused Appearance defaults helper that loads, validates, and builds save payloads from existing config keys. Settings renders compact terminal-native controls, owns only persisted global defaults, and keeps Customize as the full visual editor and preview surface.

**Tech Stack:** Python 3.11+, Textual, existing Settings screen contracts, `SettingsConfigAdapter`, pytest, Textual-web/CDP screenshot QA.

---

## Scope

This plan implements `TASK-80`. It is intentionally limited to the Settings Appearance category and existing visual-default config keys.

In scope:

- `general.default_theme`
- `general.palette_theme_limit`
- `web_server.font_size`
- `appearance.density`
- `appearance.animations_enabled`
- `appearance.smooth_scrolling`
- visible validation, save, revert, and runtime-safe preview/apply
- direct route to Customize for full theme editing

Out of scope:

- raw color token editing
- theme creation
- full Customize rewrite
- sync, server handoff, workspace policy, or Settings-wide restructuring

## ADR Check

ADR required: no

ADR path: N/A

Reason: This slice uses existing persisted config sections and preserves the established Settings/Customize boundary. It does not introduce a storage schema, runtime boundary, service contract, security policy, dependency, or long-lived application structure.

## Files And Responsibilities

- `tldw_chatbook/UI/Screens/settings_appearance_defaults.py`
  - Pure load, normalize, validate, and save-payload helpers for Appearance defaults.
- `tldw_chatbook/UI/Screens/settings_screen.py`
  - Appearance category rendering, draft state, validation copy, save/revert/preview actions, and inspector guidance.
- `Tests/UI/test_settings_appearance_defaults.py`
  - Pure helper coverage for load, validation, and save payload behavior.
- `Tests/UI/test_settings_configuration_hub.py`
  - Mounted Settings coverage for Appearance controls, dirty state, validation, save, revert, focus readability, and route-to-Customize.
- `Docs/superpowers/qa/product-maturity/screen-qa/settings/`
  - Actual Textual-web/CDP screenshot evidence for the visible Appearance slice.
- `backlog/tasks/task-80 - Functionalize-Settings-Appearance-defaults.md`
  - Task status, ADR check, implementation notes, and verification evidence.

## Task 1: Add Pure Appearance Defaults Contract

- [x] **Step 1: Write failing helper tests**

Create `Tests/UI/test_settings_appearance_defaults.py` covering:

- empty config loads safe defaults
- nested config values load from `general`, `web_server`, and `appearance`
- invalid numeric values fall back for load but fail strict validation when staged
- save payload deep-merges without dropping unrelated config values
- public helper functions have Google-style docstrings

Run:

```bash
python -m pytest -q Tests/UI/test_settings_appearance_defaults.py --tb=short
```

Expected: fail because the helper module does not exist yet.

- [x] **Step 2: Implement minimal helper module**

Create `settings_appearance_defaults.py` with:

- `SettingsAppearanceDefaults`
- `load_appearance_defaults(app_config)`
- `validate_appearance_defaults(values)`
- `build_appearance_save_sections(app_config, values)`
- small normalizers for theme, density, booleans, and integer ranges

- [x] **Step 3: Verify helper tests pass**

Run:

```bash
python -m pytest -q Tests/UI/test_settings_appearance_defaults.py --tb=short
```

Expected: pass.

## Task 2: Render Guided Appearance Controls

- [x] **Step 1: Write failing mounted tests**

Extend `Tests/UI/test_settings_configuration_hub.py` to assert:

- Appearance shows editable controls, not just an `Open Appearance` route
- inspector explains each Appearance setting
- save/revert buttons enable only when Appearance has a valid dirty draft
- invalid values block save and show recovery copy
- focused Appearance inputs stay readable

Run:

```bash
python -m pytest -q Tests/UI/test_settings_configuration_hub.py --tb=short
```

Expected: fail because Appearance is still routed/read-only.

- [x] **Step 2: Wire Appearance draft state**

Modify `SettingsScreen` to:

- include `SettingsCategoryId.APPEARANCE` in guided mutation categories
- load and merge Appearance defaults through the helper
- stage changes through `SettingsDraft`
- show dirty, invalid, saved, and reverted copy
- keep `Open Customize` available as the deeper editor route

- [x] **Step 3: Verify mounted tests pass**

Run:

```bash
python -m pytest -q Tests/UI/test_settings_configuration_hub.py --tb=short
```

Expected: pass.

## Task 3: Save, Revert, And Preview Safely

- [x] **Step 1: Write failing save/revert tests**

Add mounted tests that use a fake `SettingsConfigAdapter` and assert:

- valid Appearance changes save to expected sections
- `app_instance.app_config` updates after save
- revert restores last loaded values in widgets
- preview/apply updates only safe runtime state and never writes config

- [x] **Step 2: Implement save worker**

Add a worker-backed Appearance save path using `@work(exclusive=True, thread=True)` and `SettingsConfigAdapter.save_sections(...)`, matching the Library/RAG and Console patterns.

- [x] **Step 3: Implement preview/apply action**

Use the `t` shortcut and an explicit button for safe runtime preview:

- apply theme to the running app when possible
- report when a runtime preview target is unavailable
- keep the category dirty until Save or Revert

- [x] **Step 4: Verify focused tests**

Run:

```bash
python -m pytest -q Tests/UI/test_settings_appearance_defaults.py Tests/UI/test_settings_configuration_hub.py --tb=short
```

Expected: pass.

## Task 4: QA Evidence And Task Closure

- [x] **Step 1: Run static and focused verification**

Run:

```bash
python -m pytest -q Tests/UI/test_settings_appearance_defaults.py Tests/UI/test_settings_configuration_hub.py --tb=short
git diff --check
```

- [x] **Step 2: Capture actual rendered screenshots**

Use Textual-web/CDP and save:

- Appearance baseline controls
- theme dropdown open
- focused input with visible typed text
- invalid value recovery state

Save under:

```text
Docs/superpowers/qa/product-maturity/screen-qa/settings/
```

- [x] **Step 3: Request user approval**

Show actual screenshots. Do not open a PR until the user approves the rendered UI.

- [x] **Step 4: Update task notes**

Check all task acceptance criteria, add implementation notes, and include verification/screenshot paths.

## QA Evidence

Automated verification:

```bash
python -m pytest -q Tests/UI/test_settings_appearance_defaults.py Tests/UI/test_settings_configuration_hub.py --tb=short
python -m pytest -q Tests/UI/test_destination_shells.py::test_settings_appearance_action_routes_to_customize_surface --tb=short
git diff --check
```

Rendered Textual-web/CDP screenshots approved by the user:

- `Docs/superpowers/qa/product-maturity/screen-qa/settings/settings-appearance-baseline.png`
- `Docs/superpowers/qa/product-maturity/screen-qa/settings/settings-appearance-theme-dropdown.png`
- `Docs/superpowers/qa/product-maturity/screen-qa/settings/settings-appearance-focused-input.png`
- `Docs/superpowers/qa/product-maturity/screen-qa/settings/settings-appearance-invalid-palette-limit.png`
