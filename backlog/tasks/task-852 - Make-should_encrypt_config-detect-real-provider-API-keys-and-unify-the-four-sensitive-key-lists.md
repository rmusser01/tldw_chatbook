---
id: TASK-852
title: >-
  Make should_encrypt_config() detect real provider API keys, and unify the four
  sensitive-key lists
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
Utils/config_encryption.py:245-256 detect_api_keys flags a key only when it exactly equals one of six literals ("api_key", "apikey", "api-key", "secret", "token", "password"). Every real secret-bearing key in this app is prefixed or suffixed instead: openai_api_key, anthropic_api_key, cohere_api_key, bing_search_api_key, tavily_search_api_key, api_token (github/confluence), auth_token ([tldw_api]), OPENAI_API_KEY_fallback, ELEVENLABS_API_KEY_fallback (config.py:66,70, written by UI/Tools_Settings_Window.py:5521,5541) -- none of them equals any of the six literals, so none of them is ever detected. config.py:4285 should_encrypt_config() returns detect_api_keys(config) directly, so a live config full of plaintext provider keys is reported as having nothing worth encrypting and the user is never prompted to turn encryption on. A direct check against a config populated with all of the above real keys returned False; the same function against a synthetic {'x': {'api_key': 'sk-1'}} returned True.

This exact-match/substring mismatch is one of four independent, disagreeing copies of "what counts as a sensitive config key": config.py:3691 _is_sensitive_setting_key (substring match), UI/Screens/settings_privacy_security.py:190 _is_sensitive_config_key (endswith match, guards _env_var), config_encryption.py:250 (exact match, this finding), and a fourth block at config.py:515-546 that duplicates :3693-3718 verbatim. A side-by-side check found keys the two main copies disagree on in both directions: config._is_sensitive_setting_key treats api_key_env_var (an env-var name, not a secret, appearing 20+ times in the default TOML) and max_tokens (30+ occurrences) as sensitive -- so encrypt_sensitive_fields (config.py:554) encrypts an env-var name into an enc: blob and _setting_value_for_log (:3721) logs max_tokens as <redacted> -- while settings_privacy_security's counts (:153,160) miss the real [app_tts] *_fallback keys and search_engine_api_key_* entirely, undercounting what the Privacy & Security panel reports as protected.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 should_encrypt_config()/detect_api_keys() recognizes all real provider-key config names (the *_api_key, *_token, and *_fallback families actually written by the app), not just the six literal names
- [ ] #2 The four independent sensitive-key-matching implementations (config.py:3691, config.py:515-546, config_encryption.py:250, settings_privacy_security.py:190) are collapsed into a single shared predicate with one semantics
- [ ] #3 The unified predicate excludes non-secret keys such as api_key_env_var and max_tokens that the substring-matching copy currently over-flags
- [ ] #4 A test builds its key list by reading the real key names out of CONFIG_TOML_CONTENT (or the equivalent live default config), not by re-asserting a hand-picked literal list, and confirms every one of them is detected
<!-- AC:END -->
