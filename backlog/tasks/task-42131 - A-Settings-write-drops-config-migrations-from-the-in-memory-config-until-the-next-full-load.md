---
id: TASK-42131
title: "A Settings write drops config migrations from the in-memory config until the next full load"
status: To Do
assignee: []
created_date: 2026-11-20 20:06
updated_date: 2026-11-20 20:06
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
- [ ] #1 After a settings write, the in-memory config (the bootstrap cache, `load_settings()` and the atomic snapshot) holds the same migrated values that a full load of the same file gives
- [ ] #2 This holds for `migrate_legacy_reasoning_history` and for every numbered migration in `_CONFIG_MIGRATIONS`, including one whose boot write did not run
- [ ] #3 A test writes a setting over a real scratch config that needs a migration and asserts the migrated value survives the write; it is red before the fix
<!-- AC:END -->
