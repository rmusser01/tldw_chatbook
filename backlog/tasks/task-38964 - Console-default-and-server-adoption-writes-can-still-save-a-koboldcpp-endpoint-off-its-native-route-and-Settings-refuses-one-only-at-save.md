---
id: TASK-38964
title: "Console default and server-adoption writes can still save a koboldcpp endpoint off its native route, and Settings refuses one only at save"
status: To Do
created_date: 2026-11-06 01:21
labels:
- follow-up
- providers
- console
- settings
priority: medium
references:
- tldw_chatbook/UI/Wizards/first_run_setup_state.py
- tldw_chatbook/Provider_Contract/provider_endpoint_contract.py
- tldw_chatbook/Chat/console_settings_defaults.py
- tldw_chatbook/UI/Console_Modules/model_switcher.py
- tldw_chatbook/UI/Screens/settings_screen.py
- tldw_chatbook/UI/Wizards/first_run_provider_detection.py
- Tests/UI/test_settings_provider_save_scope.py
dependencies:
- TASK-36516
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down. Checked on dev 29592ecc9b and on the TASK-36516/36517 branch (the state after it merges), 2026-11-05; line numbers are from that branch. TASK-36516 made Settings refuse a koboldcpp endpoint that is not on the native route. The predicate it uses, `first_run_endpoint_refusal`, lives in the wizard module (`tldw_chatbook/UI/Wizards/first_run_setup_state.py:357-365`, `endswith("/api/v1/generate")`), and Settings imports it from there lazily (`settings_screen.py:16132-16146`). Two Console write paths never call it:

- **Console ▸ Settings modal ▸ "Make default for new chats".**
  - `_endpoint_patch_is_authorized` (`tldw_chatbook/Chat/console_settings_defaults.py:1095-1110`) checks the action, the field mask, the dirty and checked flags, the provider, and `parse_console_endpoint_preview` (authority and network class).
  - `:1254-1257` then writes the raw `patch.value.strip()` to the koboldcpp section of `api_settings`. Nothing in the file checks the route.
- **Model-switcher server adoption.**
  - `model_switcher.py:959` writes `provider_values["api_url"] = server.base_url` with no route check.
  - koboldcpp discovery candidates come only from the koboldcpp section's own endpoints (`local_server_discovery.py:431-445`).
  - `normalize_probe_base_url("http://127.0.0.1:5001/api/v1/generate")` is None, while a bare root normalizes to itself. So adoption can only ever write back a root that is not the native route.

A non-native endpoint then fails at send. `chat_with_kobold` (`LLM_API_Calls_Local.py:965`) posts the native payload to the saved URL. An OpenAI-style route answers with `choices`, and the send fails with "Unexpected response structure" (`:1216-1224`).

Settings also shows the refusal later than the wizard does:
- Its endpoint `Input.Changed` handler (`settings_screen.py:30621-30653`) only stages the value, and the refusal runs only inside Save's `connection_dirty` branch (`:33380-33392`).
- The wizard's `_refresh_endpoint_status` (`first_run_provider_detection.py:366-411`) shows "✗ KoboldCpp needs its native /api/v1/generate address…" while the user types.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One koboldcpp native-route rule lives in Provider_Contract and the first-run wizard, Settings, the Console defaults write and the model-switcher adoption all use it; no copy of the /api/v1/generate check remains outside it
- [ ] #2 Console "Make default for new chats" with a koboldcpp endpoint off the native route is refused with the native-route copy and writes nothing; a test is red on dev (the root is written)
- [ ] #3 Model-switcher server adoption never writes a non-native koboldcpp api_url (it keeps the saved endpoint or uses the native route); a test is red on dev
- [ ] #4 Settings shows the native-route refusal on the endpoint row while the user types, as the wizard does, and still refuses at save; a test types a bare root and reads the row before any save, and is red on dev
- [ ] #5 A koboldcpp endpoint already on /api/v1/generate saves unchanged on every path, and other providers' writes are unchanged
<!-- AC:END -->
