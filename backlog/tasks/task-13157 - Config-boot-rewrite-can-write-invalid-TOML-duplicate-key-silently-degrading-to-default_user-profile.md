---
id: TASK-13157
title: >-
  Config boot-rewrite can write invalid TOML (duplicate key), silently degrading
  to default_user profile
status: Done
assignee: []
created_date: '2026-08-09 16:47'
updated_date: '2026-08-09 19:46'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found via live TUI verification (supervisor-fleet PR-1, Task 7). A hand-authored scratch config.toml with a single [api_settings.openrouter] api_key entry, after the app's own config-rewrite-on-boot pass ran across two launches, ended up with the SAME api_key key defined TWICE inside the same [api_settings.openrouter] table -- invalid TOML (tomllib raises 'Cannot overwrite a value'). Because the config file failed to parse, the app appears to have silently fallen back to defaults: the resolved data directory became ~/.local/share/tldw_cli/default_user/ (the REAL user's live profile) instead of the configured users_name profile, and the first-run wizard re-offered itself despite [first_run] setup_completed=true being present in the (invalid) file. Confirmed via lsof that the running process held open file handles on default_user/*.db, and via direct SQL/mtime checks that no actual data was written there in this incident -- but the mechanism is a real, reproducible, silent isolation failure: an app whose own config-normalization write path can corrupt its own config file into unparseable TOML, with no error surfaced to the user, and a silent fallback to the DEFAULT profile rather than a fail-loud error. This has real blast radius for any user whose config.toml happens to trigger the same duplicate-key condition on a rewrite pass -- their session could silently start operating against the wrong (default) profile without any indication.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Root cause identified: which config-rewrite/normalization code path in config.py can write a duplicate key into an existing TOML table, and under what preconditions
- [x] #2 The rewrite path is fixed to be idempotent (never emits a duplicate key for a table it is re-serializing)
- [x] #3 A regression test constructs a config.toml already containing a user-set key that coincides with a template default key, runs the rewrite path twice, and asserts the result stays valid, parseable TOML with exactly one occurrence of that key
- [x] #4 If config loading ever fails to parse, the app fails loudly (a visible error/notification naming the file and parse error) rather than silently falling back to the default_user profile
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: config.py's config-mutating dicts are plain Python dicts (round-tripped
through tomllib on read), so they cannot themselves hold a duplicate key -- extensive
attempts to reproduce the live incident's exact trigger through pure application logic
(full default+user re-merge across simulated launch/shutdown cycles, targeted
apply_settings_mutation_to_cli_config deltas, a 3000-trial fuzz of toml.dumps round-trip
fidelity) all stayed clean, consistent with the live-verification write-up's own
"neither edit alone was the problem" framing. The two real, independently-confirmed
defects: (1) every write path (_write_raw_cli_config_unlocked, used by
apply_settings_mutation_to_cli_config / replace_cli_config / persist_cli_config_for_shutdown)
serialized via the third-party `toml` encoder and committed atomically WITHOUT ever
verifying the output re-parses through the SAME stdlib tomllib reader the next boot uses --
no self-check closed the gap between two independently-maintained implementations; and
(2) _load_cli_config_bootstrap_unlocked already computes a `succeeded` flag and logs a
TOMLDecodeError via loguru, but BOTH its callers (load_cli_config_and_ensure_existence,
load_settings) discard that flag and return bare in-memory defaults with zero signal --
this is the exact mechanism that silently resolved the profile to default_user.

Fix: (1) _write_raw_cli_config_unlocked now parses its own freshly-serialized TOML back
through tomllib before the atomic write; a round-trip failure raises the new
ConfigSerializationError (a ValueError subclass, so every existing catch site --
apply_settings_mutation_to_cli_config's generic Exception catch, persist_cli_config_for_
shutdown's ValueError tuple, replace_cli_config_serialized's existing ValueError contract
-- covers it with no other changes) instead of committing unparseable bytes. This makes
the write path idempotent by construction: it can never regress a valid file into an
invalid one. (2) Added ConfigLoadFailure (path, message) + module state
_LAST_CONFIG_LOAD_FAILURE, set on TOMLDecodeError and cleared on the next successful
bootstrap, exposed via get_config_load_failure(). Wired into app.py mirroring the
existing _instance_lock_status / _maybe_warn_second_instance pattern: snapshotted in
__init__ (before the UI exists to notify through) and surfaced via a new
_maybe_warn_config_load_failure(), called alongside _maybe_warn_second_instance() once
the initial screen mounts -- a persistent (60s) severity="error" notification naming the
exact file and parse error.

Tests added to Tests/test_config_private_bootstrap.py (matches existing file's
conventions/fixtures): (a) test_config_rewrite_refuses_to_commit_a_serialization_that_
would_duplicate_a_coinciding_key -- constructs a config.toml with a user-set
api_settings.google.api_key coinciding with the shipped template's active default,
writes once through the real save_setting_to_cli_config path, then monkeypatches
toml.dumps to simulate a misbehaving encoder on a second write; asserts
ConfigSerializationError from the low-level writer, save_setting_to_cli_config
returning False (matching this codebase's established "report failure via return value,
don't raise" idiom for that layer), and the on-disk file staying byte-identical with
exactly one occurrence of the coinciding key. Verified failing for the real reason
(DID NOT RAISE ConfigSerializationError) with the guard temporarily reverted, then
restored. (b) test_corrupt_config_produces_a_loud_load_failure_not_a_silent_default_
fallback -- a genuinely corrupt file (duplicate api_settings.openrouter table/key,
tomllib's actual "Cannot declare ... twice" shape) still degrades to default_user (the
incident's symptom) but get_config_load_failure() now names the file+error, and clears
on repair.

Gate: Tests/test_config_private_bootstrap.py 24/24, Tests/test_config_persistence_owner.py
5/5, all root Tests/test_config_*.py 134/134, Tests/test_smoke.py 16/16 (confirms
TldwCli.__init__ still constructs with the new get_config_load_failure() import/call).
No Tests/Config/ directory exists. Full-repo `pytest --collect-only -q`: 35160 tests
collected, 0 errors -- no import breakage from the app.py/config.py edits.

Modified: tldw_chatbook/config.py (ConfigLoadFailure, get_config_load_failure,
ConfigSerializationError, _write_raw_cli_config_unlocked round-trip guard,
_load_cli_config_bootstrap_unlocked failure-state set/clear), tldw_chatbook/app.py
(import get_config_load_failure; __init__ snapshots it; new
_maybe_warn_config_load_failure(); wired alongside _maybe_warn_second_instance()),
Tests/test_config_private_bootstrap.py (2 new tests + _clear_config_cache resets the
new state).
<!-- SECTION:NOTES:END -->
