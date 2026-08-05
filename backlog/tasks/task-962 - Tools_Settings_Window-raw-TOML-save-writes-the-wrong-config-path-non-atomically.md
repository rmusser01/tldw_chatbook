---
id: TASK-962
title: >-
  Tools_Settings_Window raw-TOML save writes the wrong config path
  non-atomically
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 14:36'
updated_date: '2026-07-27 18:08'
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
- [x] #1 _save_raw_toml_config reads and writes through config._get_effective_config_path() instead of the DEFAULT_CONFIG_PATH literal
- [x] #2 _save_raw_toml_config's file write uses the same atomic_write_text helper TASK-851's three entry points use, not a plain open(path,'w')+toml.dump
- [x] #3 A regression test sets TLDW_CONFIG_PATH to a profile file, invokes the Save action, and asserts the change landed in the file config._get_effective_config_path() returns (derived via that accessor, not a re-spelled literal path) while a decoy DEFAULT_CONFIG_PATH is left untouched
- [x] #4 A regression test simulates a mid-write failure (e.g. toml.dumps raising) and asserts the on-disk config file is byte-for-byte unchanged, proving the write is atomic
- [x] #5 Saving raw TOML config with no profile override active still round-trips correctly (existing default-path behavior is not regressed)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read _save_raw_toml_config and confirm whether it still uses the described plain open(path,'w')+toml.dump against DEFAULT_CONFIG_PATH, or whether prior dev work already changed it.
2. If already fixed, verify precisely how (which helper, which path resolver) and confirm it matches TASK-851's pattern; if not, route through config._get_effective_config_path() and atomic_write_text as TASK-851's three sites do.
3. Add regression tests: profile-path-vs-decoy-DEFAULT_CONFIG_PATH (AC3), mid-write atomicity failure (AC4), and no-override round trip (AC5).
4. Revert-check each test against a reintroduced version of the bug shape it targets, confirm it fails for the right reason, then restore.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
No source change needed: _save_raw_toml_config already reads/writes through the effective config path and writes atomically. It calls config.replace_cli_config(config_data), which resolves the target via get_cli_config_path() (== _get_effective_config_path(), honoring TLDW_CONFIG_PATH) and persists via _write_raw_cli_config_unlocked() -> atomic_private_write_text() -- the same underlying atomic-write mechanism TASK-851's three encryption entry points now use themselves (they were refactored, after 851 landed, to also go through _write_raw_cli_config_unlocked rather than the literal atomic_write_text call 851's own notes described; atomic_private_write_text is a hardened superset with the same crash-safety guarantee). Confirmed via git history: the old open(DEFAULT_CONFIG_PATH,'w')+toml.dump pattern this task describes was removed from Tools_Settings_Window.py in a prior dev-reconciliation commit (1df0c4cb4), before this task was ever picked up.

Added three regression tests (Tests/UI/test_tools_settings_window.py) covering the three ACs that had no live coverage (the AppTest-based settings_window fixture that used to exercise this path is skipped under this Textual version):
- test_save_raw_toml_config_writes_effective_path_not_default_decoy (AC3): TLDW_CONFIG_PATH set to a profile path, DEFAULT_CONFIG_PATH set to a distinct decoy; asserts the save lands in _get_effective_config_path()'s file and the decoy is never created.
- test_save_raw_toml_config_is_atomic_on_serialization_failure (AC4): patches toml.dumps/dump to raise mid-write; asserts the on-disk file is byte-for-byte unchanged.
- test_save_raw_toml_config_roundtrips_with_no_profile_override (AC5): TLDW_CONFIG_PATH deleted, DEFAULT_CONFIG_PATH pointed at the target; asserts the no-override case still round-trips.

Revert-checked each: AC3/AC5 were verified against the described historical bug shape (DEFAULT_CONFIG_PATH imported into Tools_Settings_Window.py's own namespace via `from ..config import DEFAULT_CONFIG_PATH`, then open(path,'w')+toml.dump) -- both failed, and in a way that itself demonstrates why the "frozen import-time constant" shape is the real bug: monkeypatching config.DEFAULT_CONFIG_PATH does not affect an already-bound `from ... import DEFAULT_CONFIG_PATH` name in a different module, so the wrong-file write landed somewhere else entirely, not even the intended decoy. AC4 needed a variant revert isolating just the atomicity property (correct effective-path resolution via get_cli_config_path(), but a plain open(path,'w')+toml.dump write) -- confirmed it truncates the file to empty on a mid-write failure. All three passed again after restoring.

No product code was changed. Modified: Tests/UI/test_tools_settings_window.py only.
<!-- SECTION:NOTES:END -->
