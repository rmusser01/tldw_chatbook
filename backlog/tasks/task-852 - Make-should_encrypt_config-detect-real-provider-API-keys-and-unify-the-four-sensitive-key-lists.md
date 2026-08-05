---
id: TASK-852
title: >-
  Make should_encrypt_config() detect real provider API keys, and unify the four
  sensitive-key lists
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
Utils/config_encryption.py:245-256 detect_api_keys flags a key only when it exactly equals one of six literals ("api_key", "apikey", "api-key", "secret", "token", "password"). Every real secret-bearing key in this app is prefixed or suffixed instead: openai_api_key, anthropic_api_key, cohere_api_key, bing_search_api_key, tavily_search_api_key, api_token (github/confluence), auth_token ([tldw_api]), OPENAI_API_KEY_fallback, ELEVENLABS_API_KEY_fallback (config.py:66,70, written by UI/Tools_Settings_Window.py:5521,5541) -- none of them equals any of the six literals, so none of them is ever detected. config.py:4285 should_encrypt_config() returns detect_api_keys(config) directly, so a live config full of plaintext provider keys is reported as having nothing worth encrypting and the user is never prompted to turn encryption on. A direct check against a config populated with all of the above real keys returned False; the same function against a synthetic {'x': {'api_key': 'sk-1'}} returned True.

This exact-match/substring mismatch is one of four independent, disagreeing copies of "what counts as a sensitive config key": config.py:3691 _is_sensitive_setting_key (substring match), UI/Screens/settings_privacy_security.py:190 _is_sensitive_config_key (endswith match, guards _env_var), config_encryption.py:250 (exact match, this finding), and a fourth block at config.py:515-546 that duplicates :3693-3718 verbatim. A side-by-side check found keys the two main copies disagree on in both directions: config._is_sensitive_setting_key treats api_key_env_var (an env-var name, not a secret, appearing 20+ times in the default TOML) and max_tokens (30+ occurrences) as sensitive -- so encrypt_sensitive_fields (config.py:554) encrypts an env-var name into an enc: blob and _setting_value_for_log (:3721) logs max_tokens as <redacted> -- while settings_privacy_security's counts (:153,160) miss the real [app_tts] *_fallback keys and search_engine_api_key_* entirely, undercounting what the Privacy & Security panel reports as protected.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 should_encrypt_config()/detect_api_keys() recognizes all real provider-key config names (the *_api_key, *_token, and *_fallback families actually written by the app), not just the six literal names
- [x] #2 The four independent sensitive-key-matching implementations (config.py:3691, config.py:515-546, config_encryption.py:250, settings_privacy_security.py:190) are collapsed into a single shared predicate with one semantics
- [x] #3 The unified predicate excludes non-secret keys such as api_key_env_var and max_tokens that the substring-matching copy currently over-flags
- [x] #4 A test builds its key list by reading the real key names out of CONFIG_TOML_CONTENT (or the equivalent live default config), not by re-asserting a hand-picked literal list, and confirms every one of them is detected
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Repro the detector bug live: build a config with real prefixed/suffixed provider key names (openai_api_key-style, bing_search_api_key, api_token, OPENAI_API_KEY_fallback) and show detect_api_keys()/check_encryption_needed() return False.\n2. Create Utils/sensitive_config_keys.py as the single shared predicate (exact-name set + api_key-family containment + suffix rules + _env_var guard), documenting why substring vs endswith vs containment differ per family.\n3. Rewire config.py (encrypt_sensitive_fields closure, _is_sensitive_setting_key removed in favor of the shared predicate), Utils/config_encryption.py detect_api_keys, and UI/Screens/settings_privacy_security.py _is_sensitive_config_key to all import and delegate to the shared predicate.\n4. Add tests that enumerate real key names out of CONFIG_TOML_CONTENT/DEFAULT_APP_TTS_CONFIG (not hand-typed literals) and confirm detection, plus explicit exclusions (api_key_env_var, max_tokens).\n5. Run targeted pytest across all four former call sites, confirm tests fail pre-fix and pass post-fix.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created tldw_chatbook/Utils/sensitive_config_keys.py as the single shared predicate (is_sensitive_config_key): an exact-name set for bare secrets (api_key, token, auth_token, ...), a containment rule for the api_key/apikey/api-key family (catches provider-prefixed names like openai_api_key and suffix-of-a-suffix names like OPENAI_API_KEY_fallback and search_engine_api_key_baidu, which endswith-only matching missed), an endswith rule for _key/_token/_secret/_password suffixes (deliberately not containment, so max_tokens is not caught by a bare "_token" substring), and an _env_var suffix guard (an env-var *name* is never a secret).

Rewired all four former call sites to delegate to it: config.py's encrypt_api_keys_in_config (encrypt_sensitive_fields closure) and the removed _is_sensitive_setting_key (now callers use the shared predicate directly), Utils/config_encryption.py's detect_api_keys (previously an exact 6-literal match that matched none of this app's real key names), and UI/Screens/settings_privacy_security.py's _is_sensitive_config_key (previously endswith-only, undercounting the *_fallback and mid-name api_key families).

Reproduced live before fixing: detect_api_keys({"SearchEngines": {"bing_search_api_key": "..."}, ...}) returned False against a config full of real plaintext provider keys; check_encryption_needed() likewise never fired for any of this app's actual key names. Confirmed True after the fix, with max_tokens/api_key_env_var-only configs still correctly returning False.

Tests: Tests/Utils/test_sensitive_config_keys.py (new, 15 tests) builds its assertions from real key names parsed out of CONFIG_TOML_CONTENT (tomllib.loads) and DEFAULT_APP_TTS_CONFIG rather than hand-typed literals -- per the audit's core lesson, a hand-re-spelled literal goes vacuous in lockstep with drift. Tests/Utils/test_config_encryption.py gained a TestDetectApiKeysRealProviderNames class (3 tests) doing the same at the detect_api_keys level. Tests/test_config_app_config_encryption.py gained an end-to-end check_encryption_needed() test. All new/updated tests fail against the pre-fix code and pass after.

Files changed: tldw_chatbook/Utils/sensitive_config_keys.py (new); tldw_chatbook/config.py; tldw_chatbook/Utils/config_encryption.py; tldw_chatbook/UI/Screens/settings_privacy_security.py; Tests/Utils/test_sensitive_config_keys.py (new); Tests/Utils/test_config_encryption.py; Tests/test_config_app_config_encryption.py.
<!-- SECTION:NOTES:END -->
