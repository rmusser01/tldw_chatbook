---
id: TASK-42131
title: "A Settings write drops config migrations from the in-memory config until the next full load"
status: Done
assignee: ['@claude']
created_date: 2026-11-20 20:06
updated_date: 2026-11-21 20:08
labels:
- follow-up
- config
- settings
priority: medium
references:
- tldw_chatbook/config.py
- tldw_chatbook/Utils/reasoning_config.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the providers stream (TASK-42002, round 18). A full config load runs the forward migrations on the raw file before it merges the defaults: `_load_cli_config_bootstrap_unlocked` calls `migrate_config_forward` and `migrate_legacy_reasoning_history` (`tldw_chatbook/config.py:4225-4232`). The two functions that rebuild the config from a raw mapping already read under the lock do not; each only merges the defaults and decrypts:
- `_install_bootstrap_cache_from_raw` (`config.py:4909`), which `_publish_runtime_config_unlocked` uses after every settings write;
- `_atomic_config_values_from_raw` (`config.py:5155`), which `get_atomic_config_snapshot` and the literal settings transaction (`config.py:6230`) use.

So a settings write undoes any migration that lives only in memory. Probe at d85e90bc9e (a scratch profile, through the test wrapper): a config holding only `[console] replay_thinking = false` loads with `reasoning_history = "off"`. After `save_setting_to_cli_config("general", ...)` the bootstrap cache and `load_settings()` both read `"auto"`, so reasoning replay turns back on until the next full load. `migrate_legacy_reasoning_history` is never written to the file. TASK-42002's numbered migration (on the round-18 providers branch, not yet on dev) is written at boot by `migrate_config_file_if_needed`, so it is lost after a write only when that boot step did not run or failed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After a settings write, the in-memory config (the bootstrap cache, `load_settings()` and the atomic snapshot) holds the same migrated values that a full load of the same file gives
- [x] #2 This holds for `migrate_legacy_reasoning_history` and for every numbered migration in `_CONFIG_MIGRATIONS`, including one whose boot write did not run
- [x] #3 A test writes a setting over a real scratch config that needs a migration and asserts the migrated value survives the write; it is red before the fix
<!-- AC:END -->

## Implementation Plan

1. Red first: a test in `Tests/test_config_hot_reload.py` (lane-run) writes a scratch v1 config that needs every registered migration (`_CONFIG_MIGRATIONS` 2 and 3) and the legacy reasoning opt-out, loads it, writes a setting without the boot persist, and asserts the bootstrap cache, `load_settings()` and the atomic snapshot hold what a full load of the same file gives. A guard in the test fails when a newly registered migration leaves its fixture unchanged.
2. Fix once in `config.py`: one helper runs every load-time migration (`migrate_config_forward`, then `migrate_legacy_reasoning_history`) over a copy of a raw mapping; the full load, `_install_bootstrap_cache_from_raw` and `_atomic_config_values_from_raw` all use it. The raw mapping a write persists is never changed, so nothing new is written to the file.
3. Covering tests, the module-size ratchet, the layer direction test.

## Implementation Notes

The fix is in one place. `config.py` gains `_migrated_raw_config(raw)`, which runs every load-time migration over a copy of one raw config: `migrate_config_forward` (every numbered migration in `_CONFIG_MIGRATIONS`), then `migrate_legacy_reasoning_history`. It returns the migrated config, whether nothing beyond the version stamp changed (the boot step's "settled" record), and any newer-version conflict. Three callers use it:
- the full load (`_load_cli_config_bootstrap_unlocked`), whose inline sequence it replaces, with the same result;
- `_install_bootstrap_cache_from_raw`, which every settings write publishes through;
- `_atomic_config_values_from_raw`, which `get_atomic_config_snapshot` and the literal settings transaction's builder snapshot use.

So the bootstrap cache, `load_settings()` (rebuilt from that cache), the atomic snapshot and the literal builder's effective values hold what a full load of the same file gives.
- `raw` is never changed. A write persists the raw mapping as read, so a write never stores an in-memory migration: the numbered migrations are still saved once, by the boot step, and the reasoning opt-out never is. When `migrate_config_forward` returns its input (the config is already current), the helper deep-copies it before the reasoning migration, which mutates in place. Without that copy, the literal transaction would write `reasoning_history` into the file. The copy costs 0.52 ms a call on the full shipped template (merge over defaults: 1.46 ms; load average 33).
- No new import: the lazy `Utils.reasoning_config` import moves from the full load into the helper, still function-level (ADR-097).
- A migration whose numbered log line runs on a write of an unpersisted file logs again on that write, as each full load already did.

**Test:** `Tests/test_config_hot_reload.py::test_a_settings_write_keeps_every_load_time_migration_in_memory` (lane-run, architecture-review step). A real scratch v1 config needs migration 2 (Groq `https://api.groq.com/v1`), migration 3 (Cohere `https://api.cohere.ai/v1`, TASK-42153) and the reasoning opt-out (`[console] replay_thinking = false`). The test loads it, writes `general.users_name` with `save_setting_to_cli_config`, and never runs the boot persist. It then asserts that the bootstrap cache, `load_settings()` and the atomic snapshot equal a forced full load (`["off", "https://api.groq.com/openai/v1", "https://api.cohere.ai"]`). A literal settings transaction's builder sees the same values, and the file still holds the raw values. A guard at the top fails when a registered migration leaves the fixture unchanged, so a new migration must add a row (AC#2).
- Red at the previous commit (a5e4957532): all three views read `['auto', 'https://api.groq.com/v1', 'https://api.cohere.ai/v1']`. Negative controls, each reverted: restoring the raw merge in `_atomic_config_values_from_raw` alone failed the atomic snapshot; in `_install_bootstrap_cache_from_raw` alone, the bootstrap cache and `load_settings()`; registering a no-op migration 4 failed the guard. Green after: the file's 7 passed, 1 xfailed.

**Covering run** (`-n 4`, load 30-48): the 52 test files that name the helper's callers or the migration functions (`_install_bootstrap_cache_from_raw`, `_atomic_config_values_from_raw`, `get_atomic_config_snapshot`, `apply_literal_settings_transaction*`, `migrate_*`, `replay_thinking`, `_publish_runtime_config_unlocked`, `_merge_user_config_over_shipped_defaults`, `_CONFIG_MIGRATION_SETTLED`), plus every `Tests/test_config*.py`: 4 failed, 3243 passed, 1 xfailed, 26 errors.
- The 26 errors are `Tests/UI/test_settings_raw_draft.py`'s local `RecoveryRequired: raw_source_selection_changed`. They are the same with dev's sources (dc371f0842).
- `Tests/UI/test_mcp_root_settings.py::test_local_config_shutdown_drains_and_fences_both_config_owners[workspace_root|local_tools_enabled]` (`DID NOT RAISE ConfigSaveUnavailable`) also fail with dev's sources.
- `test_settings_configuration_hub.py::test_settings_provider_category_rejects_out_of_range_model_profile[...top-p...]` and `test_settings_image_gen_panel.py::test_emptying_a_saved_field_deletes_not_blanks` failed once under load. Re-run at the head, they passed 4 times out of 4 (3 runs alongside `test_settings_raw_draft.py`).

`config.py` 8913 -> 8928 (+15), with a dated row comment in `Tests/Architecture/test_module_size_ratchet.py`.

Files: `tldw_chatbook/config.py`, `Tests/test_config_hot_reload.py`, `Tests/Architecture/test_module_size_ratchet.py`.
