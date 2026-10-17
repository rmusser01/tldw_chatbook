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
- tldw_chatbook/TTS/legacy_bridge.py
- tldw_chatbook/TTS/TTS_Backends.py
- tldw_chatbook/Model_Artifacts/acquisition.py
- tldw_chatbook/UI/LLM_Management/vllm_connection.py
- tldw_chatbook/UI/Screens/llm_screen.py
- backlog/decisions/012-provider-credential-settings-boundary.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the round-1 review of Phase 9 Task 17 (2026-10-17). TASK-33641.11 made the loader's legacy `[API]` bridge honour `credential_source`, and TASK-33001.14 made summaries take the key readiness resolves. Readers that go to the raw `[API]` section themselves were outside that task and still ignore the source. `config.get_api_key` never reads `credential_source`: it reads the table's `api_key_env_var` even under "none", then falls back to `get_cli_setting("API", "<provider>_api_key")` and the conventional variable. Its callers are Console realtime (`UI/Console_Modules/realtime.py`), the OpenAI and ElevenLabs TTS backends, the MCP server's chat tools (`MCP/server.py`, `MCP/tools.py`), and the Models screen's vLLM connection probe (`UI/LLM_Management/vllm_connection.py::_default_credential_resolver`, which sends the key as `Bearer`) and credential flag (`UI/Screens/llm_screen.py`): under "none" `get_api_key("vllm")` returns the variable the shipped `api_key_env_var = "VLLM_API_KEY"` names whenever it is exported, though readiness ignores it. `RAG_Search/simplified/active_config.py` reads `OPENAI_API_KEY` and then `[API] openai_api_key` directly. Text to speech takes its OpenAI key in `TTS/legacy_bridge.py::legacy_provider_config`, which reads the raw TOML's `[API] openai_api_key` (then `OPENAI_API_KEY`) into the projection's `openai_api` dict that `TTS/TTS_Backends.py::_prepare_backend_config` hands to the backend, so fixing `TTS_Backends.py` alone leaves TTS sending the removed key. `Model_Artifacts/acquisition.py::EnvConfigCredentialResolver` reads `HUGGINGFACE_API_KEY`, `HF_TOKEN`, then `[API] huggingface_api_key`; Parakeet, OmniVoice and the Models screen's remote GGUF installs use it by default. Probed at the Task 17 head (74b5af9d16) with the real loader on a scratch config with `[API] openai_api_key` set and `[api_settings.openai] credential_source = "none"`: readiness resolved no key and `openai_api.api_key` was None, but `get_api_key("openai")` returned the `[API]` key. Probed again at 366741be8f the same way: with `[API] huggingface_api_key` set and Hugging Face's source "none", readiness and `huggingface_api.api_key` were None but the artifact resolver returned the `[API]` key; with OpenAI's source "none", `legacy_provider_config("openai", …)` put the `[API]` OpenAI key in `openai_api`; with vLLM's source "none" and `VLLM_API_KEY` exported, readiness resolved no key but the vLLM probe's resolver returned the variable. So after a Clear in Settings, these features still spend the key the user removed, against ADR-012. The User Guide's settings page names them as still reading the `[API]` line, and says the vLLM connection check still sends `VLLM_API_KEY`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With credential_source none recorded for a provider, `config.get_api_key` returns no key even when the provider's `[API]` key, its table's named variable or its conventional variable is set, proved by a real-loader test on a scratch config
- [ ] #2 With credential_source environment, `config.get_api_key` returns only the variable the table names, and with stored only the table's key (or the `[API]` key the bridge copied there), matching readiness for every combination the test covers
- [ ] #3 A provider with no credential_source recorded keeps today's order, pinned by the existing `get_api_key` tests
- [ ] #4 The TTS projection (`TTS/legacy_bridge.py`, and `TTS/TTS_Backends.py` behind it), the RAG embedding reader and the Hugging Face artifact resolver take their key from the same rule as `config.get_api_key`, or the remaining difference is recorded in ADR-012
- [ ] #5 The User Guide's settings page no longer names these features (the `[API]` readers and the vLLM connection check) as ones a Clear does not reach once they honour the source
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
