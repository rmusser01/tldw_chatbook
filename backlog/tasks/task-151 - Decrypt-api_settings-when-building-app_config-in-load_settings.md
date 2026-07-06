---
id: TASK-151
title: Decrypt api_settings when building app_config in load_settings
status: To Do
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
- [ ] #1 With config encryption enabled and a session password set, app.app_config['api_settings'][provider]['api_key'] is decrypted plaintext at startup rather than enc: ciphertext
- [ ] #2 The Chat send path authenticates successfully against a keyed provider whose api_key is stored encrypted (no restart-into-plaintext workaround needed)
- [ ] #3 The non-encrypted path and startup performance are unchanged
- [ ] #4 A regression test covers the encrypted-startup case
<!-- AC:END -->
