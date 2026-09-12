---
id: TASK-26036
title: 'Config: serve last-known-good on parse failure'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-31 15:47'
updated_date: '2026-09-01 22:38'
labels:
  - ops
  - config
  - reliability
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A single bad character in config.toml silently reverts security-relevant settings to defaults. Verified on origin/dev: on a TOMLDecodeError the loader falls back to DEFAULT_CONFIG_FROM_TOML and records a ConfigLoadFailure (config.py:5052,5079,5165-5178) - so encryption settings, custom database paths and provider configuration all revert while the app keeps running. Hermes copies the corrupt file aside and continues serving the last successfully loaded config, explicitly so security-critical settings survive a mid-edit break. Roughly five lines, and the failure it prevents is the kind that is discovered late.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 When a config file fails to parse, the last successfully loaded configuration is served instead of built-in defaults
- [x] #2 The unparseable file is preserved on disk under a distinct name so the user's edits are not lost
- [x] #3 The user is told plainly that config failed to load, which file is at fault, and which configuration is in effect
- [x] #4 With no previously loaded configuration available (first run), the existing default fallback applies unchanged
- [x] #5 A subsequent successful load replaces the retained copy and clears the warning
- [x] #6 Tests cover: parse failure after a good load, parse failure on first run, and recovery after the file is fixed
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: parse-fail-after-good serves last-known-good + preserves aside; first-run uses defaults; recovery clears\n2. Capture retained_good before the cache is cleared on force_reload\n3. TOMLDecodeError branch: _preserve_corrupt_config_aside + serve retained_good instead of defaults; failure message names which config is in effect\n4. Config-loader regression
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Built on existing TASK-13157 infra (_CONFIG_CACHE last-successful-load + _LAST_CONFIG_LOAD_FAILURE signal app.py surfaces). Key fix: the force_reload path cleared _CONFIG_CACHE BEFORE re-reading, so a parse failure lost the last-known-good; now retained_good is captured (for the same path) before the clear. On TOMLDecodeError: _preserve_corrupt_config_aside copies the unparseable file to <name>.corrupt-<UTC-timestamp> (best-effort, never raises — AC#2, edits never lost) and, when a retained good exists, loaded_config = that instead of DEFAULT_CONFIG_FROM_TOML (AC#1 — encryption/db/provider settings survive a mid-edit break). The ConfigLoadFailure message now names which config is in effect + the aside filename (AC#3, app.py already renders it). First run / no prior good load keeps the default fallback unchanged (AC#4, pinned); a later success replaces the cache + clears the failure (AC#5, pre-existing, pinned). 3 new tests (after-good / first-run / recovery). Two test_config_read_fastpath_task21124 failures are PRE-EXISTING (verified: fail identically with config.py reverted). Lesson candidate: a resilience fallback that reads from a cache the SAME function clears must snapshot the cache before clearing — an easy self-inflicted loss.
<!-- SECTION:NOTES:END -->
