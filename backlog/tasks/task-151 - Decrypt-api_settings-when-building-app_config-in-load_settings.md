---
id: TASK-151
title: Decrypt api_settings when building app_config in load_settings
status: Done
assignee: []
created_date: '2026-07-06 15:43'
labels:
  - bug
  - security
  - config
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When config encryption is enabled, load_settings() returns the raw [api_settings] TOML table (it never calls decrypt_config_section), so at startup app.app_config['api_settings'][<provider>]['api_key'] holds enc: ciphertext. The Chat send path resolves keys from app.app_config via get_provider_readiness and passes that value verbatim, so provider auth fails for users who store keys with encryption enabled. load_cli_config_and_ensure_existence() already decrypts (config.py:2803); load_settings() (config.py:613) does not, and app.py:1382/7021 populate app_config from it. Discovered during PR #583 (inline Chat-Defaults API key), which fixed only the save-and-refresh path (_refresh_live_api_settings).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With config encryption enabled and a session password set, app.app_config['api_settings'][provider]['api_key'] is decrypted plaintext at startup rather than enc: ciphertext
- [x] #2 The Chat send path authenticates successfully against a keyed provider whose api_key is stored encrypted (no restart-into-plaintext workaround needed)
- [x] #3 The non-encrypted path and startup performance are unchanged
- [x] #4 A regression test covers the encrypted-startup case
<!-- AC:END -->

## Implementation Notes

Two changes in `tldw_chatbook/config.py`:

1. **`load_settings()`** now calls `decrypt_config_section(toml_config_data)` right after merging the config layers (before section extraction). This mirrors the decrypt step already in `load_cli_config_and_ensure_existence`, so `app.app_config["api_settings"]` (and the legacy `[API]` keys) surface as plaintext. `decrypt_config_section` is a no-op when encryption is disabled or no session password is set, so the non-encrypted path (AC #3) is unaffected — decryption only runs its cost when encryption is actually on.

2. **`set_encryption_password()`** now nulls `_SETTINGS_CACHE` and `_CONFIG_CACHE`. This closes a caching gap: `APP_CONFIG = load_settings()` runs at module import (app.py:409), *before* the startup unlock prompt, priming the settings cache with ciphertext; without cache invalidation, the later `self.app_config = load_settings()` (app.py:1382) would return that stale ciphertext even with change #1 in place. The startup ordering is: unlock prompt → `set_encryption_password` (app.py:7176, clears caches) → `TldwCli()` → `load_settings()` reloads and decrypts.

**Tests:** `Tests/test_config_app_config_encryption.py` — (a) `load_settings(force_reload=True)` returns the decrypted key for an encrypted on-disk config; (b) after priming the cache with ciphertext (no password), `set_encryption_password` invalidates it so a subsequent non-forced load decrypts (the startup path). Broader sweep (config + settings + encryption + feature suites): 276 passed, 16 skipped, 0 failed.

**Files:** `tldw_chatbook/config.py`, `Tests/test_config_app_config_encryption.py`.
