---
id: TASK-26039
title: 'Config: unknown-key and deprecated-key validation'
status: To Do
assignee: []
created_date: '2026-08-31 15:47'
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
- [ ] #1 Keys present in the config file that no code reads are reported to the user
- [ ] #2 A near-miss key suggests the intended key rather than only reporting it as unknown
- [ ] #3 Keys known to be renamed or removed are reported as deprecated with their replacement named
- [ ] #4 Reporting is advisory: an unknown key never prevents startup or discards the rest of the file
- [ ] #5 The check covers nested tables, not only top-level keys
- [ ] #6 Keys under sections documented as free-form or user-extensible are exempt and do not produce noise
<!-- AC:END -->
