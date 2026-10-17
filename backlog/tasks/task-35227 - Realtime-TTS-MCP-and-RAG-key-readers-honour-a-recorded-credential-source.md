---
id: TASK-35227
title: Realtime, TTS, MCP and RAG key readers honour a recorded credential source
status: To Do
created_date: 2026-10-17 15:27
dependencies:
- TASK-33641.11
labels:
- model-config-redesign
- follow-up
- credentials
priority: medium
references:
- tldw_chatbook/config.py
- tldw_chatbook/RAG_Search/simplified/active_config.py
- tldw_chatbook/TTS/TTS_Backends.py
- backlog/decisions/012-provider-credential-settings-boundary.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the round-1 review of Phase 9 Task 17 (2026-10-17). TASK-33641.11 made the loader's legacy `[API]` bridge honour `credential_source`, and TASK-33001.14 made summaries take the key readiness resolves. Readers that go to the raw `[API]` section themselves were outside that task and still ignore the source. `config.get_api_key` never reads `credential_source`: it reads the table's `api_key_env_var` even under "none", then falls back to `get_cli_setting("API", "<provider>_api_key")` and the conventional variable. Its callers are Console realtime (`UI/Console_Modules/realtime.py`), the OpenAI and ElevenLabs TTS backends, and the MCP server's chat tools (`MCP/server.py`, `MCP/tools.py`). `RAG_Search/simplified/active_config.py` reads `OPENAI_API_KEY` and then `[API] openai_api_key` directly, and `TTS/TTS_Backends.py` reads `[API] openai_api_key` from the app config. Probed at the Task 17 head (74b5af9d16) with the real loader on a scratch config with `[API] openai_api_key` set and `[api_settings.openai] credential_source = "none"`: readiness resolved no key and `openai_api.api_key` was None, but `get_api_key("openai")` returned the `[API]` key. So after a Clear in Settings, these features still spend the key the user removed, against ADR-012. The User Guide's settings page names them as still reading the `[API]` line.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With credential_source none recorded for a provider, `config.get_api_key` returns no key even when the provider's `[API]` key, its table's named variable or its conventional variable is set, proved by a real-loader test on a scratch config
- [ ] #2 With credential_source environment, `config.get_api_key` returns only the variable the table names, and with stored only the table's key (or the `[API]` key the bridge copied there), matching readiness for every combination the test covers
- [ ] #3 A provider with no credential_source recorded keeps today's order, pinned by the existing `get_api_key` tests
- [ ] #4 The RAG embedding and TTS readers of `[API] openai_api_key` take their key from the same rule, or the remaining difference is recorded in ADR-012
- [ ] #5 The User Guide's settings page no longer names these features as still reading the `[API]` line once they honour the source
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
