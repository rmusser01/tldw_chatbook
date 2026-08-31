---
id: TASK-26040
title: 'Config: schema version and forward migrations'
status: To Do
assignee: []
created_date: '2026-08-31 15:48'
labels:
  - ops
  - config
dependencies:
  - TASK-26036
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The config file has no version, so renames leave orphan keys forever. Verified on origin/dev: a grep for config_version, CONFIG_VERSION and migrat across config.py returns only database and feature-flag comments - the loader deep-merges user TOML over defaults (config.py:5113,5128) and a key that moved in a past refactor simply stops being read while remaining in the file. Every rename accumulates dead configuration. Databases in this repo already have versioned stepwise migrations (DB/ChaChaNotes_DB.py:582,605); config has none. Hermes carries numbered config migrations behind a version check.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The config file carries a schema version
- [ ] #2 Stepwise migrations transform an older config forward to the current version on load
- [ ] #3 Migrations run atomically through the existing locked write path, and a failed migration leaves the original file untouched
- [ ] #4 The original file is backed up before the first migration writes
- [ ] #5 A config from a newer version than the running code is detected and reported rather than silently mangled
- [ ] #6 An unversioned (pre-existing) config is treated as the baseline version and migrated, not rejected
- [ ] #7 Migrations are covered by tests that assert a realistic old config reaches the current shape with values preserved
<!-- AC:END -->
