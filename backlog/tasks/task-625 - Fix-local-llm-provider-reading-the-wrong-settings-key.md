---
id: TASK-625
title: Fix local-llm provider reading the wrong settings key
status: To Do
assignee: []
created_date: '2026-07-25 12:40'
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
- [ ] A `local-llm` provider configured under `[api_settings.local-llm]` in config.toml is usable for a chat completion
- [ ] The key the code reads and the key the documented example writes agree (`api_url` vs `api_ip` reconciled, with the legacy name still accepted if it was ever functional)
- [ ] The failure mode when the URL genuinely is missing remains a clear configuration error, not an HTTP 502
- [ ] Other local providers in the same module are checked for the same top-level-vs-api_settings mistake and fixed if present
- [ ] A test pins that the provider resolves its base URL from `[api_settings.local-llm]`, failing against the current code
<!-- AC:END -->
