---
id: TASK-625
title: Fix local-llm provider reading the wrong settings key
status: Done
assignee:
  - '@claude'
created_date: '2026-07-25 12:40'
updated_date: '2026-07-25 20:04'
labels:
  - llm
  - config
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The `local-llm` provider cannot be configured at all. `chat_with_local_llm` in `tldw_chatbook/LLM_Calls/LLM_API_Calls_Local.py` resolves its config with:

```python
cfg = settings.get("local-llm", {})
api_base_url = cfg.get("api_ip")
```

That reads a **top-level** `local-llm` key. The provider's configuration actually lives at `settings["api_settings"]["local-llm"]` — which is exactly where the app's own documented example puts it (`config.py`, the commented `[api_settings.local-llm]` block). Every sibling local-provider function in the same file resolves correctly via `settings.get("api_settings", {})`, so this one is inconsistent with its own neighbours.

The result is that any user who configures `local-llm` as documented gets:

```
ChatConfigurationError: Llamafile/Local LLM API URL (api_url) is required and
could not be determined from arguments or configuration.
```

surfaced in the Console as *"Agent run failed: provider returned HTTP 502 … configuration error"*.

There is no config workaround: adding a top-level `[local-llm]` section does not help, because `load_settings()` does not preserve arbitrary top-level sections — verified, the key is still absent afterwards. The provider is therefore reachable only by passing `api_url` programmatically.

Note the mismatch also spans key *names*: the documented block uses `api_url`, while the code reads `api_ip`.

Found during live UAT of skill script execution (task-579), where it blocked driving any agent turn until the run was switched to the `llama_cpp` provider (which resolves its config correctly).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A `local-llm` provider configured under `[api_settings.local-llm]` in config.toml is usable for a chat completion
- [ ] #2 The key the code reads and the key the documented example writes agree (`api_url` vs `api_ip` reconciled, with the legacy name still accepted if it was ever functional)
- [ ] #3 The failure mode when the URL genuinely is missing remains a clear configuration error, not an HTTP 502
- [ ] #4 Other local providers in the same module are checked for the same top-level-vs-api_settings mistake and fixed if present
- [ ] #5 A test pins that the provider resolves its base URL from `[api_settings.local-llm]`, failing against the current code
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
One-line root cause, verified end to end.

chat_with_local_llm resolved its config with settings.get('local-llm', {}) — a TOP-LEVEL key. The provider's config lives at api_settings.local-llm, which is where the app's own documented example puts it and where every sibling local provider in the same module looks. load_settings() does not preserve arbitrary top-level sections either, so no config.toml could ever populate that key: the provider was unusable from configuration, always failing with 'API URL is required' and surfacing in the Console as an opaque HTTP 502.

Fix: read cli_api_settings = settings.get('api_settings', {}) then cfg = cli_api_settings.get('local-llm', {}), matching the siblings exactly.

Key-name reconciliation (AC#2): the documented block uses api_url while the code read api_ip, so resolution is now cfg.get('api_url') or cfg.get('api_ip') — the documented key wins, the historical one still works so nobody with a functioning setup breaks.

Sweep (AC#4): grepped every settings.get(' call across LLM_Calls/. Line 503 was the ONLY offender; koboldcpp, ooba_api, tabby_api, aphrodite_api, llama_cpp, vllm and the rest all correctly go through settings['api_settings'].

Verified beyond unit tests: a config.toml with the documented [api_settings.local-llm] block now completes a real chat_api_call against a live llama.cpp server — the exact case that failed during the task-579 UAT.

Tests: Tests/LLM/test_local_llm_provider_config.py (4) written RED-first — 3 failed against the old code, and the missing-URL case passed throughout, proving the error path was untouched. Tests/LLM + Tests/Chat 2189 passed / 69 skipped, ruff clean.

Files: tldw_chatbook/LLM_Calls/LLM_API_Calls_Local.py, Tests/LLM/test_local_llm_provider_config.py
<!-- SECTION:NOTES:END -->
