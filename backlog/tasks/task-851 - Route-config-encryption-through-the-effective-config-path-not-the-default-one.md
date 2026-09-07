---
id: TASK-851
title: 'Route config encryption through the effective config path, not the default one'
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 04:34'
updated_date: '2026-07-27 14:08'
labels:
  - security
  - config
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
tldw_chatbook/config.py's three encryption entry points -- enable_config_encryption (:4309-4330), disable_config_encryption (:4362-4393), and change_encryption_password (:4425-4458) -- open DEFAULT_CONFIG_PATH (config.py:46) directly instead of _get_effective_config_path() (config.py:49), the accessor every other read/write in the app uses (config.py:681, :3597, :3893, :3994) and that honors TLDW_CONFIG_PATH. TLDW_CONFIG_PATH is a supported, user-facing mode (the "Override config" control at UI/Screens/settings_screen.py:5035-5058, documented at runtime_policy/bootstrap.py:30, and exercised by this project's own test suite), so any user running with a profile active hits this gap.

A sandboxed reproduction (real enable_config_encryption call, HOME redirected, a profile config active via TLDW_CONFIG_PATH) showed: enable_config_encryption('hunter2hunter2') returns True, the ACTIVE profile's plaintext secret is still present afterward, and a completely different file -- DEFAULT_CONFIG_PATH -- was the one actually rewritten. disable_config_encryption and change_encryption_password then read encryption.password_verifier from that same wrong file and fail with "No password verifier found", so a user who did get encrypted (on the default path, no profile active) cannot rotate or remove the password once a profile becomes active. Live UI callers are UI/Tools_Settings_Window.py:3857, :3897, :6787, :6838, :6936, so this is reachable from the Settings screen, not just internal API misuse.

Secondary defect in the same code: config.py:4330 writes with plain open(path, "w") while the sibling entry points (:4393, :4458) use atomic_write_text, so an interrupted "enable encryption" can truncate the config it does manage to hit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All three encryption entry points (enable_config_encryption, disable_config_encryption, change_encryption_password) read and write through _get_effective_config_path() instead of DEFAULT_CONFIG_PATH
- [x] #2 All three entry points use the same atomic-write helper (atomic_write_text) as their siblings
- [x] #3 A regression test sets TLDW_CONFIG_PATH to a profile file, calls each entry point, and asserts the change landed in the file _get_effective_config_path() returns, not in a hardcoded literal
- [x] #4 Enabling, disabling, and rotating encryption with a profile active round-trips correctly (password verifier readable, secrets encrypted/decrypted in the active file)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Repro the wrong-path bug live: set TLDW_CONFIG_PATH to a profile config, call enable_config_encryption, show DEFAULT_CONFIG_PATH gets rewritten instead.\n2. Route enable/disable/change_encryption_password through _get_effective_config_path() instead of DEFAULT_CONFIG_PATH.\n3. Replace enable_config_encryption's plain open(path,'w')+toml.dump with atomic_write_text(path, toml.dumps(...)), matching its two siblings.\n4. Add regression tests (Tests/test_config_encryption_effective_path.py) that set TLDW_CONFIG_PATH to a profile file + a decoy DEFAULT_CONFIG_PATH, and assert each entry point reads/writes only the active file; add a round-trip test (enable->disable, enable->change_password) and an atomicity test that simulates a mid-serialization crash and asserts the file is untouched.\n5. Run targeted pytest, confirm tests fail pre-fix and pass post-fix.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Routed enable_config_encryption, disable_config_encryption, and change_encryption_password (config.py) through _get_effective_config_path() instead of the DEFAULT_CONFIG_PATH literal, so a TLDW_CONFIG_PATH override (active profile) is the file actually read and rewritten. Replaced enable_config_encryption's plain open(path,'w')+toml.dump with atomic_write_text(path, toml.dumps(...)), matching its two siblings (disable/change already used atomic_write_text, just against the wrong path).

Reproduced live before fixing: with TLDW_CONFIG_PATH pointed at a profile config holding a plaintext sk-proj-... key, enable_config_encryption returned True, left the profile's key in plaintext, and instead wrote an [encryption] section into the unrelated DEFAULT_CONFIG_PATH file. Confirmed fixed after the change (same repro now encrypts the active file in place; DEFAULT_CONFIG_PATH untouched).

Tests added: Tests/test_config_encryption_effective_path.py -- 6 tests covering all three entry points writing to the effective path (with a decoy DEFAULT_CONFIG_PATH asserted untouched), an enable->disable round trip, an enable->change_password round trip (old password rejected, new password works), and an atomicity test that patches toml.dump/dumps to raise mid-serialization and asserts the on-disk file is byte-for-byte unchanged (a plain open(path,'w') would have already truncated it). All 6 fail against the pre-fix code (via `git stash`) and pass after.

Files changed: tldw_chatbook/config.py (three entry points); Tests/test_config_encryption_effective_path.py (new).
<!-- SECTION:NOTES:END -->
