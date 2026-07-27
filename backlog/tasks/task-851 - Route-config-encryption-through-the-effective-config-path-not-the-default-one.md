---
id: TASK-851
title: 'Route config encryption through the effective config path, not the default one'
status: To Do
assignee: []
created_date: '2026-07-27 04:34'
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
- [ ] #1 All three encryption entry points (enable_config_encryption, disable_config_encryption, change_encryption_password) read and write through _get_effective_config_path() instead of DEFAULT_CONFIG_PATH
- [ ] #2 All three entry points use the same atomic-write helper (atomic_write_text) as their siblings
- [ ] #3 A regression test sets TLDW_CONFIG_PATH to a profile file, calls each entry point, and asserts the change landed in the file _get_effective_config_path() returns, not in a hardcoded literal
- [ ] #4 Enabling, disabling, and rotating encryption with a profile active round-trips correctly (password verifier readable, secrets encrypted/decrypted in the active file)
<!-- AC:END -->
