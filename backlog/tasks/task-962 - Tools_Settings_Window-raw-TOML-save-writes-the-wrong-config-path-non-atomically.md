---
id: TASK-962
title: >-
  Tools_Settings_Window raw-TOML save writes the wrong config path
  non-atomically
status: To Do
assignee: []
created_date: '2026-07-27 14:36'
labels:
  - security
  - config
  - ui
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
_save_raw_toml_config in UI/Tools_Settings_Window.py (the Settings screen's raw-TOML editor Save action) writes the user's edited config to the hardcoded DEFAULT_CONFIG_PATH via a plain open(path,'w')+toml.dump, instead of the effective config path the rest of the app reads and writes through (config._get_effective_config_path(), which honors a TLDW_CONFIG_PATH profile override) and instead of the atomic_write_text helper TASK-851 standardized on for config.py's three encryption entry points. This is the same wrong-path-plus-non-atomic-write shape TASK-851 fixed, on a fourth, previously unnamed write path into the live config file. A user running with a profile active (TLDW_CONFIG_PATH set) who edits the raw TOML in Settings and clicks Save would have their edits silently land in DEFAULT_CONFIG_PATH instead of their active profile file -- the exact same class of silent-data-loss-to-the-wrong-file bug TASK-851 fixed for encryption -- and an interrupted write (crash, kill -9) partway through toml.dump could truncate whichever file it did hit, corrupting the on-disk config.

Filed from TASK-853's audit sweep: the agent fixing TASK-851 (config.py's enable/disable/change_encryption_password) found this sibling defect but left it out of scope since it wasn't one of 851's three named entry points. Same shape, fourth call site.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 _save_raw_toml_config reads and writes through config._get_effective_config_path() instead of the DEFAULT_CONFIG_PATH literal
- [ ] #2 _save_raw_toml_config's file write uses the same atomic_write_text helper TASK-851's three entry points use, not a plain open(path,'w')+toml.dump
- [ ] #3 A regression test sets TLDW_CONFIG_PATH to a profile file, invokes the Save action, and asserts the change landed in the file config._get_effective_config_path() returns (derived via that accessor, not a re-spelled literal path) while a decoy DEFAULT_CONFIG_PATH is left untouched
- [ ] #4 A regression test simulates a mid-write failure (e.g. toml.dumps raising) and asserts the on-disk config file is byte-for-byte unchanged, proving the write is atomic
- [ ] #5 Saving raw TOML config with no profile override active still round-trips correctly (existing default-path behavior is not regressed)
<!-- AC:END -->
