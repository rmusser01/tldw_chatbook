---
id: TASK-433
title: Fix llama.cpp prefilled endpoint default to server root
status: Done
assignee: []
created_date: '2026-07-21 09:38'
updated_date: '2026-07-23 14:11'
labels:
  - settings
  - llm-calls
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the RP/character-card UX review (Docs/superpowers/qa/rp-ux-review-2026-07-21/report.md). The Settings default/hint for llama.cpp is http://localhost:8080/completion, but the chat caller always appends v1/chat/completions unless the URL already ends with it (LLM_API_Calls_Local.py:213-222), producing .../completion/v1/chat/completions. A user who keeps the suggested default gets failures after a green-looking setup. Either fix the default to the server root or make the caller strip/tolerate the legacy /completion suffix.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A user who accepts the prefilled llama.cpp endpoint and a running llama-server on that port gets working chat completions
- [x] #2 Endpoint guidance text matches what the caller actually does with the value
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: the config default [api_settings.llama_cpp].api_url was http://localhost:8080/completion (llama.cpp's native endpoint), but the legacy caller (chat_with_llama -> _chat_with_openai_compatible_local_server) appends /v1/chat/completions unless the URL already ends with it -> http://localhost:8080/completion/v1/chat/completions. The caller was also fragile for any partial path (the /v1 placeholder form doubled to /v1/v1/...). The native-Console path already tolerated this via console_provider_gateway.normalize_llamacpp_base_url.

Fix (robust, chosen over minimal): 
- AC#1/AC#2 (Task 1): config.py [api_settings.llama_cpp].api_url -> server root 'http://localhost:8080' + honest comment (only that line; llama_api_IP and [providers] llama_cpp left alone). Settings placeholders (llama_cpp/local_llamacpp) -> 'http://127.0.0.1:9099' (dropped /v1). Updated one hardcoded placeholder assertion in test_settings_configuration_hub.
- AC#1 for the legacy path (Task 2): chat_with_llama now normalizes its base URL via the EXISTING public normalize_llamacpp_base_url (deferred local import, after the empty-guard, non-empty only) before the shared caller appends the path. Now /completion, /v1, bare root, and the full path all resolve to http://localhost:8080/v1/chat/completions. Retroactively fixes users who saved /completion (no migration). Scoped to chat_with_llama (covers llama_cpp/local_llamacpp/local_llamafile); shared caller and other providers untouched.

Design decisions/trade-offs: reused the gateway's existing normalizer (not a rename — it was already public; the console_session_settings.py duplicate is a noted follow-up, not deduped here); deferred import to keep the gateway's heavy deps out of app startup + avoid cycles; server-root default over full /v1/chat/completions form (both work; root matches gateway semantics).

Tests: config-default (server root, no /completion); normalizer-contract (strip-to-root incl scheme-less + reverse-proxy-prefix left unchanged); parametrized chat_with_llama URL-capture (/completion, /v1, root -> exactly /v1/chat/completions) mutation-verified to fail without the fix. Full SDD: per-task implement+review + whole-branch review, all clean.

Files: config.py, UI/Screens/settings_screen.py, LLM_Calls/LLM_API_Calls_Local.py, Tests/Chat/test_chat_functions.py, Tests/Chat/test_console_provider_gateway.py, Tests/UI/test_settings_configuration_hub.py.
<!-- SECTION:NOTES:END -->
