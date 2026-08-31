---
id: TASK-26036
title: 'Config: serve last-known-good on parse failure'
status: To Do
assignee: []
created_date: '2026-08-31 15:47'
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
- [ ] #1 When a config file fails to parse, the last successfully loaded configuration is served instead of built-in defaults
- [ ] #2 The unparseable file is preserved on disk under a distinct name so the user's edits are not lost
- [ ] #3 The user is told plainly that config failed to load, which file is at fault, and which configuration is in effect
- [ ] #4 With no previously loaded configuration available (first run), the existing default fallback applies unchanged
- [ ] #5 A subsequent successful load replaces the retained copy and clears the warning
- [ ] #6 Tests cover: parse failure after a good load, parse failure on first run, and recovery after the file is fixed
<!-- AC:END -->
