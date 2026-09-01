---
id: TASK-26039
title: 'Config: unknown-key and deprecated-key validation'
status: Done
assignee: []
created_date: '2026-08-31 15:47'
updated_date: '2026-09-01 23:09'
labels:
  - ops
  - config
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A typo in config.toml is silently ignored. Verified on origin/dev: validation is a TOML parse plus a table-shape check surfaced in Settings Diagnostics (UI/Screens/settings_screen.py:9030,9044-9060); nothing detects a key that no code reads. A user who writes api_setting instead of api_settings gets defaults and no warning, and a key renamed by a past refactor sits in the file forever doing nothing. Hermes validates structure, collects deprecated keys and suggests corrections for near-misses.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Keys present in the config file that no code reads are reported to the user
- [x] #2 A near-miss key suggests the intended key rather than only reporting it as unknown
- [x] #3 Keys known to be renamed or removed are reported as deprecated with their replacement named
- [x] #4 Reporting is advisory: an unknown key never prevents startup or discards the rest of the file
- [x] #5 The check covers nested tables, not only top-level keys
- [x] #6 Keys under sections documented as free-form or user-extensible are exempt and do not produce noise
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pure validate_config_keys(user, reference=DEFAULT_CONFIG_FROM_TOML) walking nested tables\n2. difflib near-miss suggestions against sibling reference keys\n3. _DEPRECATED_CONFIG_KEYS map + _FREEFORM_CONFIG_PREFIXES exemptions\n4. format_config_key_report advisory string\n5. Wire into SettingsConfigAdapter.validate_config_file (advisory, stays valid)\n6. TDD + mutation checks
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Advisory config-key validation surfaced in Settings ▸ Diagnostics.

Approach:
- Pure validate_config_keys(user_config, reference=DEFAULT_CONFIG_FROM_TOML) in config.py walks the config recursively (AC#5). A key absent from the programmatic-defaults shape is 'unknown' (AC#1); difflib.get_close_matches(cutoff=0.8) against sibling reference keys attaches a near-miss suggestion (AC#2). Keys equal to or nested under a _DEPRECATED_CONFIG_KEYS entry are reported 'deprecated' with the replacement path instead of unknown (AC#3, seeded with the legacy [API] -> api_settings rename). _FREEFORM_CONFIG_PREFIXES exempts genuinely user-extensible sections (api_settings, providers, model_capabilities.models/patterns, SearchEngines, Prompts/prompts) so real typos elsewhere still surface without noise (AC#6).
- format_config_key_report renders a one-line advisory. Wired into SettingsConfigAdapter.validate_config_file: findings are appended to the validation message while the result stays valid=True, so an unknown key never blocks startup or discards the file (AC#4).

Deliberately advisory + conservative: the reference is the authoritative default shape the app already ships, and free-form sections are exempted rather than deeply policed, trading some typo coverage in dynamic sections for zero false positives.

Tests: Tests/test_config_key_validation.py (8, incl. adapter integration); near-miss + freeform assertions mutation-verified.

Files: tldw_chatbook/config.py, tldw_chatbook/UI/Screens/settings_config_adapter.py, Tests/test_config_key_validation.py.
<!-- SECTION:NOTES:END -->
