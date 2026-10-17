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
- tldw_chatbook/UI/Screens/settings_speech_tts.py
- tldw_chatbook/UI/Wizards/first_run_voice_credentials.py
- tldw_chatbook/Model_Artifacts/acquisition.py
- tldw_chatbook/UI/LLM_Management/vllm_connection.py
- tldw_chatbook/UI/Screens/llm_screen.py
- backlog/decisions/012-provider-credential-settings-boundary.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the round-1 review of Phase 9 Task 17 (2026-10-17). TASK-33641.11 made the loader's legacy `[API]` bridge honour `credential_source`, and TASK-33001.14 made summaries take the key readiness resolves. Readers that go to the raw `[API]` section themselves were outside that task and still ignore the source. `config.get_api_key` never reads `credential_source`: it reads the table's `api_key_env_var` even under "none", then falls back to `get_cli_setting("API", "<provider>_api_key")` and the conventional variable. Its callers are Console realtime (`UI/Console_Modules/realtime.py`), the OpenAI and ElevenLabs TTS backends, the MCP server's chat tools (`MCP/server.py`, `MCP/tools.py`), and the Models screen's vLLM connection probe (`UI/LLM_Management/vllm_connection.py::_default_credential_resolver`, which sends the key as `Bearer`) and credential flag (`UI/Screens/llm_screen.py`): under "none" `get_api_key("vllm")` returns the variable the shipped `api_key_env_var = "VLLM_API_KEY"` names whenever it is exported, though readiness ignores it. `RAG_Search/simplified/active_config.py` reads `OPENAI_API_KEY` and then `[API] openai_api_key` directly. Text to speech takes its OpenAI key in `TTS/legacy_bridge.py::legacy_provider_config`, which reads the raw TOML's `[API] openai_api_key` (then `OPENAI_API_KEY`) into the projection's `openai_api` dict that `TTS/TTS_Backends.py::_prepare_backend_config` hands to the backend, so fixing `TTS_Backends.py` alone leaves TTS sending the removed key. Two TTS surfaces read the same raw `[API] openai_api_key` themselves, from `COMPREHENSIVE_CONFIG_RAW`, and never check the source: `UI/Screens/settings_speech_tts.py::_credential_states` (its `_CREDENTIAL_LOCAL_LOCATIONS["openai"]` ends in `("API", "openai_api_key")`) sets the Settings ▸ Speech & TTS credential source, which `Widgets/Settings_Widgets/speech_tts_settings_panel.py` shows as "Saved local config" with "Replace credential"; and `UI/Wizards/first_run_voice_credentials.py::find_openai_credential` (its `_SAVED_KEY_LOCATIONS` end the same way) picks the key the first-run Voice step reports as found, sends in its test and lets its save go ahead with. Today both agree with what synthesis sends, because the projection sends the `[API]` key too. Fixing the projection alone would leave the panel showing a saved key that synthesis no longer sends, and the Voice step testing with it: the readiness/spend split ADR-012 forbids. `Model_Artifacts/acquisition.py::EnvConfigCredentialResolver` reads `HUGGINGFACE_API_KEY`, `HF_TOKEN`, then `[API] huggingface_api_key`; Parakeet, OmniVoice and the Models screen's remote GGUF installs use it by default. Probed at the Task 17 head (74b5af9d16) with the real loader on a scratch config with `[API] openai_api_key` set and `[api_settings.openai] credential_source = "none"`: readiness resolved no key and `openai_api.api_key` was None, but `get_api_key("openai")` returned the `[API]` key. Probed again at 366741be8f the same way: with `[API] huggingface_api_key` set and Hugging Face's source "none", readiness and `huggingface_api.api_key` were None but the artifact resolver returned the `[API]` key; with OpenAI's source "none", `legacy_provider_config("openai", …)` put the `[API]` OpenAI key in `openai_api`; with vLLM's source "none" and `VLLM_API_KEY` exported, readiness resolved no key but the vLLM probe's resolver returned the variable. Probed at a3990cd6f3 with `[API] openai_api_key` set and OpenAI's source "none" in `COMPREHENSIVE_CONFIG_RAW` and no `OPENAI_API_KEY`: `_credential_states` gave "Saved local config" with `local_saved` True, and `find_openai_credential` returned the `[API]` key. So after a Clear in Settings, these features still spend the key the user removed, against ADR-012. The User Guide's settings page names them as still reading the `[API]` line, and says the vLLM connection check still sends `VLLM_API_KEY`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With credential_source none recorded for a provider, `config.get_api_key` returns no key even when the provider's `[API]` key, its table's named variable or its conventional variable is set, proved by a real-loader test on a scratch config
- [ ] #2 With credential_source environment, `config.get_api_key` returns only the variable the table names, and with stored only the table's key (or the `[API]` key the bridge copied there), matching readiness for every combination the test covers
- [ ] #3 A provider with no credential_source recorded keeps today's order, pinned by the existing `get_api_key` tests
- [ ] #4 The TTS projection (`TTS/legacy_bridge.py`, and `TTS/TTS_Backends.py` behind it), the RAG embedding reader and the Hugging Face artifact resolver take their key from the same rule as `config.get_api_key`, and the Settings ▸ Speech & TTS credential state (`settings_speech_tts.py::_credential_states`) and the first-run Voice step's key lookup (`first_run_voice_credentials.py::find_openai_credential`) follow the same rule as the TTS projection, or each remaining difference is recorded in ADR-012
- [ ] #5 The User Guide's settings page no longer names these features (the `[API]` readers and the vLLM connection check) as ones a Clear does not reach once they honour the source
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
