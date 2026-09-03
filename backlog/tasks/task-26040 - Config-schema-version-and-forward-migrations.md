---
id: TASK-26040
title: 'Config: schema version and forward migrations'
status: Done
assignee: []
created_date: '2026-08-31 15:48'
updated_date: '2026-09-01 23:05'
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
- [x] #1 The config file carries a schema version
- [x] #2 Stepwise migrations transform an older config forward to the current version on load
- [x] #3 Migrations run atomically through the existing locked write path, and a failed migration leaves the original file untouched
- [x] #4 The original file is backed up before the first migration writes
- [x] #5 A config from a newer version than the running code is detected and reported rather than silently mangled
- [x] #6 An unversioned (pre-existing) config is treated as the baseline version and migrated, not rejected
- [x] #7 Migrations are covered by tests that assert a realistic old config reaches the current shape with values preserved
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pure migrate_config_forward runner (version detection, stepwise apply, newer-version conflict)\n2. RED tests for runner + load integration + persist/backup + failure-leaves-original\n3. Wire in-memory migration into load path on the RAW file before default merge\n4. Stamp new configs (CONFIG_TOML_CONTENT) + conflict signal for AC#5\n5. migrate_config_file_if_needed persist-with-backup via locked write path; boot call
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Config now carries config_schema_version (=1); the loader runs stepwise forward migrations on load.

Approach:
- Pure migrate_config_forward(config) -> (migrated, changed, conflict) in config.py: reads the version (missing/junk = baseline 0, AC#6), returns a NEWER-than-supported config untouched with a human-readable conflict (AC#5), else applies each numbered _CONFIG_MIGRATIONS[v] in order and stamps the current version (AC#2).
- Load path: the RAW user file is migrated BEFORE the default deep-merge (so an unversioned file is seen as baseline 0, not the default's current version) and any conflict is recorded in _CONFIG_SCHEMA_CONFLICT / get_config_schema_conflict() for app.py to surface (mirrors ConfigLoadFailure).
- AC#1: new configs are created carrying the version (CONFIG_TOML_CONTENT); existing files get the stamp in-memory on load and on their next natural save (via _config_data_for_persistence), avoiding a boot rewrite that would strip user comments.
- AC#3/#4: migrate_config_file_if_needed() runs under the write lock, backs up the original (_advanced_backup_path) then atomically rewrites the migrated result through _write_raw_cli_config_unlocked; a raising migration leaves the original untouched. Guarded to a free no-op until a real migration function is registered (empty registry today) and only persists an actual content transform, never a bare stamp. Wired once at boot in main_cli_runner.

Registry is empty at v1 (first versioned config); the runner + persist path are proven end-to-end with an injected synthetic v1->v2->v3 rename migration (AC#7), the same way DB migrations are tested. The first real key rename registers a function and bumps the version.

Tests: Tests/test_config_migrations.py (9 tests) - runner unversioned/current/newer/stepwise; load-path stamp+conflict; persist backup+rewrite; failure-leaves-original; bare-stamp-no-write.

Files: tldw_chatbook/config.py, tldw_chatbook/app.py, Tests/test_config_migrations.py.

NOTE: Tests/test_config_read_fastpath_task21124.py has 2 PRE-EXISTING failures (write path does 2 tomllib.loads vs the asserted 1) that are unrelated to this task - proven by stashing these changes. The branch is ~359 commits behind origin/dev, which carries byte-identical code + test; this is stale-branch divergence that reconciles on rebase, not a regression from 26040.
<!-- SECTION:NOTES:END -->
